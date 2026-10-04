# Plan: model switching — POST /v1/models/switch

Goal: one running server can hold several *configured* models and swap the resident one on
request. Today the config only supports aliases of ONE resident model (`set_aliases`,
`model_for`); a request for another name is still served by the same weights. This feature
adds real switching: unload the current engine + vision, load another model's engine,
tokenizer and chat template, and swap them on the `Service` atomically.

Decisions settled earlier (kept here):
- One model resident at a time (the GPU holds one MoE anyway); switching = unload + load.
- Loading takes minutes — a synchronous switch request matches the existing `/load`
  behaviour (the client waits; no job/poll API in v1).
- The switch endpoint is a *control* endpoint: it takes `svc.fifo` NON-blocking, re-checks
  `svc.status` busy/queued under `status_lock`, and raises `ModelBusy` (409) when
  contended — exactly the `/v1/load` / `/v1/unload` pattern (server.py ~3118).
- CSRF: `_own_page(...)` like `/load`, `/unload`, `/v1/vram`. API-key auth alone is not a
  CSRF defense.

## 1. Config schema (run config, written by setup.py / edited by the Settings view)

New optional key `"models"`: a list of model entries. The existing top-level keys
(`exe`, `args`, `cwd`, `tokenizer`, `model_name`, `vision`, `sampling`, `log`, `gpu`,
`backend`, ...) stay exactly as they are and describe the *default* model. Each entry in
`"models"` is the same shape — a complete per-model config — except the shared
network/security keys (`host`, `api_key`, `cors_origins`, `trusted_origins`,
`allowed_hosts`, `mcp_servers`, `before_load`, `api_monitor`, `idle_unload_s`,
`min_free_vram_mib`, `open_browser`, `lazy_load`), which stay global.

```json
{
  "exe": "...", "args": ["--native", "pack/full", ...], "tokenizer": "pack/full/tokenizer",
  "model_name": "qwen3.8-flash-next-q4", "log": "strata.log",
  "models": [
    { "model_name": "qwen3.8-flash-next-coder-q4",
      "args": ["--native", "pack-coder/full", ...],
      "tokenizer": "pack-coder/tokenizer", "exe": "...", "cwd": "...", "log": "strata.log" }
  ]
}
```

Validation at start (`main()`), before any minutes-long engine start:
- every entry needs `model_name`; names must be unique and must not collide with the
  default model's name or its aliases (ValueError → `SystemExit`, like the other config
  errors).
- relative `exe`/paths resolve against the entry's `cwd` (else the config's `cwd`), same
  rule as today (server.py ~3990).
- an entry whose `tokenizer/vocab.json` is missing fails at start with the same message
  shape as line 3950.
- `"models"` without `--engine strata` (mock): allowed — entries become mock model specs
  (see tests).

## 2. Extract the shared loader pieces (pure refactor first, no behaviour change)

- `load_tokenizer(tpath: Path) -> tokenizer | None` — the block at server.py 3948–3960
  (vocab.json + merges.txt + token_type.json → `strata_tokenizer.Tokenizer`, else
  `ByteTokenizer`). `main()` calls it; the switch path calls it per model.
- `chat_template_for(tpath: Path) -> ChatTemplate` — `tpath/chat_template.jinja` if it
  exists, else `ROOT/serve/chat_template.jinja` (the rule at 4013–4015).
- `engine_start_checks(cfg, tok) -> (exe, silence_s, effort_end_args)` +
  `build_engine(cfg, env, lazy, checked) -> StrataEngine` — the per-model part of
  3962–4004, split so the ValueError→SystemExit boundary stays exactly where it was:
  only the config checks (layer split, silence, effort_position) raise ValueError; the
  engine's own errors (bad engine option, exit before READY) escape as before. For a
  switch target, always `lazy=True` (the engine is spawned right after, see 3.).
- `build_vision(cfg_entry, env)` — the vision-config path resolution + `Vision(...)`
  from 3974–3981.

Keep `main()` calling these so there is exactly one tokenizer/template/engine code path.

## 3. Service changes

`Service.__init__` gains `self.model_specs: dict[str, dict] = {}` — name → config entry,
**including the default model's own spec** (review finding: after switching away, the
default must stay switchable back to, and its spec is what the rollback path rebuilds).
The running state (`engine/tok/template/model/vision/sampling_defaults/aliases/stop_ids/...`)
stays as-is.

New method `switch_model(name: str) -> dict` on `Service`:

1. `name` unknown → `KeyError`/ValueError → 404 "model not found" (same body as the
   `/v1/load` model check).
2. `name == self.model` → no-op, return `{"status": "already", "model": name}`.
3. `self.fifo.acquire(blocking=False)` fails → `ModelBusy` (409). Under `status_lock`,
   `busy or queued` → `ModelBusy` — parallel requests do not hold the fifo, so this
   check is required (copy 3129–3134).
4. Inside the fifo:
   a. `self.engine.close()` (ends the process; `EngineStuck` → 503, server unchanged),
      and `self.vision.close()` if present. Do NOT reuse `restart()` — the spawn tuple
      must change.
   b. **min_free_vram check AFTER the unload, before the new engine spawns** (decided
      after review): the old model's VRAM is what the replacement needs, so checking
      before the unload would count the current model's allocation against its own
      replacement (a switch that fits would be refused). Use the same
      wait-until-deadline loop `ensure_loaded` uses (1785–1793); on GpuBusy (503),
      restart the OLD engine (the rollback path below) — the current model comes back.
   c. Build the new pieces from the spec: `tok = load_tokenizer(...)`,
      `template = chat_template_for(...)`, `checked = engine_start_checks(spec, tok)`
      (ValueError → 400/503, nothing else touched),
      `vision = build_vision(spec)` if configured (vision starts before the engine so a
      GPU encoder takes its VRAM first — same order as `ensure_loaded`), then
      `engine = build_engine(spec, env, lazy=True, checked)` and spawn it
      (`engine.restart()` — `StrataEngine(lazy=True)` does not spawn; lazy first lets a
      failed tokenizer load fail before any process spawns).
   d. On any failure (engine exits before READY, vision fails): the old model is already
      gone — rebuild the OLD engine from its old spec and report the failure (503 with
      the engine's start hint). Keep the old spec around for exactly this rollback.
   e. Only after the new engine is running and READY: swap on self — `engine, tok,
      template, model, vision`, plus per-model `sampling_defaults`, `aliases` (from the
      entry; empty if absent), `stop_ids` (recomputed from the new tokenizer,
      1709–1710), `effort_end`, `gpu_index`/`gpu_indices`/`backend` (monitor reads the
      right card), `reasoning_budget_tokens` if the entry sets one. Re-apply
      `self.vram_reserve` via `engine.vram(...)` like `ensure_loaded` does (1807–1811)
      — AFTER the engine runs, else `vram()` raises EngineDied. Publishing a started
      engine also keeps readers outside the fifo from seeing a half-installed model.
   f. Reset per-model runtime state that must not leak: `self.rate.clear()`,
      `self.live_reqs.clear()`, `self.last_timings = None`, `self.conv_log.reset()`,
      `self.totals` keep (they are server-wide). `shared` settings stay (they are
      client settings, not model settings).
5. Print the `[strata] switched to model ...` line; return
   `{"status": "switched", "model": name, **self.v1_status()}`.

`model_names()` / `model_for()` stay as they are (aliases of the *resident* model).
A chat request naming a *switchable* (not resident) model: v1 keeps today's behaviour —
unknown names are served by the resident model, as before. Switching is explicit.
(Open question for review: reject with 404 + hint, or auto-switch. Default: no auto-switch.)

## 4. HTTP endpoint

`do_POST`, next to `/v1/load` / `/v1/unload` (server.py ~3118):

```
POST /v1/models/switch   {"model": "<name>"}
```

- `_own_page("the model can be switched")` (JSON + same-origin) — it mutates server state.
- `req["model"]` must be a string; missing/other type → 400.
- Unknown name → 404 `{"error": {"message": "model not found", "known": [...]}}` —
  `known` only lists switchable names + the resident one.
- Contended → 409 `model_busy` (existing handler at 3170 already maps `ModelBusy`).
- Engine start failure → 503 `server_error` (existing `EngineDied`/`EngineStuck` mapping).
- Success → 200 with the switch result.
- Also accept `POST /switch` as the un-prefixed control twin of `/load`/`/unload`
  (`_control_body()` + `_own_page`), for symmetry with the web app's control calls.

`GET /v1/models` (3048): when `svc.model_specs` is non-empty, list every switchable
model too — resident one `"status": {"value": "loaded"}`, others
`{"value": "switchable"}` with `"switchable": true`. Clients that only understand
loaded/unloaded ignore the extra entries' flag; the id list stays honest.

`GET /v1/status` (`v1_status`, 2075): add `"switchable_models": [...]` so the web app can
draw a picker later.

## 5. Web app (serve/web) — v1 scope: none required

The endpoint works with curl/API clients from day one. A model dropdown in the chat page
is a follow-up (it would call `/v1/models/switch` and re-read `/v1/models`); keep it out
of this PR to keep the diff reviewable.

## 6. Tests (`serve/test_models_switch.py`, MockEngine, no GPU — follow test_server.py shape)

Build a `Service` with a mock default model + two mock specs (distinct `model_name`,
distinct tokenizer/template fixtures — `ByteTokenizer` + the repo template is fine; the
mock "engine" per spec is a `MockEngine` with a different script so the answer proves
which model is resident).

- switch happy path: POST `/v1/models/switch` → 200; `/v1/models` now lists the new
  model as loaded; a chat completion returns the second model's script; `svc.stop_ids`
  and `svc.model` updated.
- switch to the resident name → 200 `"already"`, no engine churn (assert close not called).
- unknown name → 404 with `known` list.
- busy: hold `svc.fifo` in the test (or set `status.busy`) → 409 `model_busy`, server
  unchanged.
- CSRF: POST with `Origin: http://evil.example` → 403; with `Content-Type: text/plain`
  → 415 (mirrors the `/load` tests in test_security.py).
- failed switch rolls back: spec whose engine raises before READY → 503, and the old
  model still answers afterwards.
- `/v1/models` listing: switchable entries present, aliases still listed under the
  resident model.
- config validation: duplicate names / missing `model_name` → start fails (unit-test the
  validator function directly, no server).

Run: `python -m unittest serve.test_models_switch serve.test_server serve.test_security
serve.test_lifecycle -v` — all green, plus the existing suite untouched.

## 7. Docs

- `docs/DETAILS.md`: API section — the endpoint, the `"models"` config key, the
  one-resident-model rule and the minutes-long switch, with the 409/404 semantics.
- `docs/AI_SETUP.md`: one line — a second model can be added to the config and switched
  without restarting the server (only for users with the RAM/VRAM headroom for the
  second pack on disk; only one is ever resident).
- `serve/runconfig.py`: do NOT make `"models"` editable in the Settings view in v1
  (a list of whole configs is not a single key; the view's contract is flat keys).
  Note it in the module docstring.

## 8. PR split (each PR green on its own)

1. Refactor: extract `load_tokenizer` / `chat_template_for` / `build_engine` /
   `build_vision` from `main()`; no behaviour change; existing tests stay green.
2. Config: `"models"` parsing + validation + `model_specs` on `Service` (no endpoint yet);
   validator unit tests.
3. `Service.switch_model` + rollback + state reset; direct unit tests (no HTTP).
4. HTTP: `/v1/models/switch` (+ `/switch` twin), `/v1/models` listing, `/v1/status`
   field; endpoint tests.
5. Docs (DETAILS.md, AI_SETUP.md) — same PR as 4 or separate.

## 9. Risks / open questions

- Long synchronous request: a switch holds the HTTP connection for minutes. Matches
  `/load`; clients with short timeouts may hang — document it. (Job/poll API is the
  fallback if that turns out to matter.)
- VRAM: the min_free_vram check runs AFTER the unload and before the new engine spawns
  (the old model's VRAM is what the replacement needs — checking earlier would count it
  against itself); on GpuBusy the old engine is restarted, so the current model comes
  back. See §3.4b.
- MCP hub, api_monitor, history/metrics: server-wide, untouched.
- `lazy_load` + `"models"`: allowed; the default stays unloaded, switching still works.
- Upstream: this is a fork feature; keep it additive so a future upstream merge stays
  clean (new functions, new endpoint branch, no signature changes to existing helpers).
