"""Explicit model switching, discovery, config validation and rollback; no GPU or model pack needed."""
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from serve import server
from serve.server import (ByteTokenizer, EngineDied, EngineStuck, GpuBusy, MockEngine,
                          Service, validate_model_specs)

ROOT = Path(__file__).resolve().parents[1]


class OtherTokenizer(ByteTokenizer):
    SPECIALS = list(reversed(ByteTokenizer.SPECIALS))


class ResidentMock(MockEngine):
    """A mock with process lifecycle and VRAM controls so start/publication order is observable."""
    def __init__(self, tok, script):
        super().__init__(tok, script, max_context=4096)
        self.running, self.starts, self.closes = True, 0, 0
        self.unloaded, self.info = False, {}
        self.reserves = []

    def alive(self):
        return self.running

    def close(self):
        self.closes += 1
        self.running = False

    def restart(self):
        self.starts += 1
        self.running = True
        self.unloaded = False

    def vram(self, reserve):
        if not self.running:
            raise EngineDied("reserve before READY")
        self.reserves.append(reserve)


class ConfigValidation(unittest.TestCase):
    def specs(self, entries, **cfg):
        return validate_model_specs({"model_name": "default", "models": entries, **cfg}, "mock")

    def test_default_is_always_switchable_and_global_keys_are_excluded(self):
        specs = self.specs([{"model_name": "second"}], host="127.0.0.1", api_key="secret")
        self.assertEqual(list(specs), ["default", "second"])
        self.assertEqual(specs["default"]["_engine"], "mock")
        self.assertNotIn("api_key", specs["default"])
        self.assertNotIn("models", specs["default"])

    def test_duplicate_names_and_missing_or_invalid_names(self):
        for entries in ([{"model_name": "default"}], [{"model_name": "a"}, {"model_name": "a"}],
                        [{}], [{"model_name": ""}], [{"model_name": 1}], [None]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                self.specs(entries)

    def test_models_must_be_a_list(self):
        for entries in (None, {}, "second"):
            with self.subTest(entries=entries), self.assertRaisesRegex(ValueError, "list"):
                self.specs(entries)

    def test_model_names_cannot_collide_with_aliases(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            self.specs([{"model_name": "alias"}], aliases="alias, other")
        with self.assertRaisesRegex(ValueError, "collide"):
            self.specs([{"model_name": "second", "aliases": ["default"]}])

    def test_global_settings_cannot_be_overridden(self):
        for key in server.MODEL_GLOBAL_KEYS:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "stay global"):
                self.specs([{"model_name": "second", key: "override"}])

    def test_paths_use_own_cwd_or_default_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = self.specs([{"model_name": "second", "exe": "engine", "tokenizer": "tok", "log": "log"},
                                {"model_name": "third", "cwd": tmp + "/other", "exe": "engine"}],
                               cwd=tmp, tokenizer="default-tok")
            self.assertEqual(specs["default"]["tokenizer"], str(Path(tmp) / "default-tok"))
            self.assertEqual(specs["second"]["exe"], str(Path(tmp) / "engine"))
            self.assertEqual(specs["second"]["tokenizer"], str(Path(tmp) / "tok"))
            self.assertEqual(specs["second"]["log"], str(Path(tmp) / "log"))
            self.assertEqual(specs["third"]["exe"], str(Path(tmp) / "other/engine"))
            self.assertNotIn("aliases", specs["second"])

    def test_native_tokenizer_is_checked_before_any_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "vocab.json").write_text("{}")
            default = {"model_name": "default", "exe": "engine", "args": [], "tokenizer": tmp}
            self.assertIn("default", validate_model_specs(default))
            with self.assertRaisesRegex(ValueError, "tokenizer is missing.*vocab.json.*run setup again"):
                validate_model_specs({**default, "models": [dict(default, model_name="second", tokenizer=tmp + "/absent")]})

    def test_per_model_settings_are_validated_at_start(self):
        for settings in ({"sampling": {"top_p": 2}}, {"sampling": []}, {"aliases": [1]}, {"reasoning_budget_tokens": True},
                         {"repeat_stop_tokens": -1}, {"anthropic_thinking": "invalid"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.specs([{"model_name": "second", **settings}])

    def test_main_installs_default_and_mock_specs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "config.json")
            path.write_text(json.dumps({"model_name": "default", "models": [{"model_name": "second"}]}))
            with mock.patch("sys.argv", ["server", "--config", str(path), "--port", "0"]), \
                    mock.patch.object(server, "serve") as start, \
                    mock.patch.object(server.time, "sleep", side_effect=KeyboardInterrupt):
                self.assertEqual(server.main(), 0)
            svc = start.call_args.args[0]
            self.assertEqual(list(svc.model_specs), ["default", "second"])
            self.assertIs(svc.current_spec, svc.model_specs["default"])

    def test_main_rejects_bad_entries_before_loading_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "config.json")
            Path(tmp, "vocab.json").write_text("{}")
            cfg = {"model_name": "default", "exe": "engine", "args": [], "tokenizer": tmp,
                   "models": [{"model_name": "default"}]}
            path.write_text(json.dumps(cfg))
            with mock.patch("sys.argv", ["server", "--engine", "strata", "--config", str(path)]), \
                    mock.patch.object(server, "Server"), mock.patch.object(server, "load_tokenizer") as load, \
                    mock.patch.object(server, "build_engine") as build:
                with self.assertRaisesRegex(SystemExit, "duplicate model_name"):
                    server.main()
            load.assert_not_called()
            build.assert_not_called()


class TelemetrySwitching(unittest.TestCase):
    def test_sampler_changes_cards_and_backend_without_a_new_thread(self):
        from serve import telemetry
        readers = {}

        def reader(index, amd):
            gpu = mock.Mock()
            gpu.ok.return_value = True
            gpu.name.return_value = f"GPU {index}"
            gpu.read.return_value = {"mem_used": index * 100, "mem_total": 1000}
            readers[index, amd] = gpu
            return gpu

        with mock.patch.object(telemetry, "gpu_reader", side_effect=reader) as create, \
                mock.patch.object(threading.Thread, "start") as start:
            sampler = telemetry.Telemetry(gpu_index=0)
            sampler.now["gpu_mem_used"] = 1
            sampler.hist["gpu_util"].append(1)
            sampler.set_gpus(2, [2, 3], amd=True)
            self.assertEqual([i for i, _ in sampler.gpus], [2, 3])
            self.assertEqual(sampler.static["gpu_name"], "GPU 2 + GPU 3")
            self.assertEqual(sampler.static["gpu_count"], 2)
            self.assertFalse(sampler.now)
            self.assertFalse(sampler.hist)
            self.assertEqual(sampler.sample()["gpu_mem_used"], 500)
            self.assertEqual(create.call_args_list, [mock.call(0, False), mock.call(2, True), mock.call(3, True)])
            start.assert_called_once()


class ModelSwitching(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tpath = Path(self.tmp.name, "second")
        self.tpath.mkdir()
        source = (ROOT / "serve/chat_template.jinja").read_text()
        (self.tpath / "chat_template.jinja").write_text("{# second model #}\n" + source)
        cfg = {"model_name": "default", "aliases": ["old-alias"], "script": "Default answer.",
               "models": [{"model_name": "second", "script": "Second answer.", "tokenizer": str(self.tpath),
                           "aliases": ["new-alias"], "sampling": {"temperature": 0.7},
                           "reasoning_budget_tokens": 42},
                          {"model_name": "third", "script": "Third answer."}]}
        self.specs = validate_model_specs(cfg, "mock")
        tok = ByteTokenizer()
        self.old = ResidentMock(tok, "Default answer.")
        self.svc = Service(self.old, tok, server.chat_template_for(Path(self.specs["default"]["tokenizer"])),
                           model_name="default")
        self.svc.model_specs, self.svc.current_spec = self.specs, self.specs["default"]
        self.svc.set_aliases(cfg["aliases"])
        self.httpd = server.serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            return response.status, json.loads(response.read())

    def switch(self, name="second", path="/v1/models/switch", headers=None):
        return self.request(path, {"model": name}, headers)

    def chat(self, model="default"):
        code, body = self.request("/v1/chat/completions", {"model": model, "max_tokens": 64,
                                  "reasoning_effort": "none", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, body)
        return body["choices"][0]["message"]["content"]

    def native(self):
        """Exercise the actual native switch loader with observable fake engine processes."""
        for spec in self.specs.values():
            spec.update(_engine="strata", exe="missing-engine", args=[])
        self.engines, self.events = [], []

        def build(spec, env, lazy, checked):
            self.assertTrue(lazy)
            tok = OtherTokenizer() if spec["model_name"] == "second" else ByteTokenizer()
            engine = ResidentMock(tok, spec["script"])
            engine.running = False
            self.engines.append(engine)
            self.events.append("build " + spec["model_name"])
            return engine, spec["model_name"] == "second"

        self.addCleanup(mock.patch.stopall)
        mock.patch.object(server, "build_engine", side_effect=build).start()
        mock.patch.object(server, "load_tokenizer", side_effect=lambda p: OtherTokenizer()
                          if p == self.tpath else ByteTokenizer()).start()

    def test_happy_path_and_switch_back_to_default(self):
        self.native()
        old_stops = self.svc.stop_ids.copy()
        code, result = self.switch()
        self.assertEqual((code, result["status"], result["model"]), (200, "switched", "second"))
        self.assertTrue(result["loaded"])
        self.assertEqual(self.old.closes, 1)
        self.assertNotEqual(self.svc.stop_ids, old_stops)
        self.assertEqual(self.svc.stop_ids, {259, 260})
        self.assertIsInstance(self.svc.tok, OtherTokenizer)
        self.assertIn("second model", self.svc.template.source)
        self.assertTrue(self.svc.effort_end)
        self.assertEqual(self.chat("second"), "Second answer.")
        self.assertEqual(self.switch("default")[0], 200)
        self.assertEqual(self.chat(), "Default answer.")
        self.assertEqual(self.svc.stop_ids, old_stops)
        self.assertEqual(self.svc.aliases, ["old-alias"])
        self.assertFalse(self.svc.effort_end)

    def test_resident_name_is_a_noop_even_when_busy(self):
        with self.svc.fifo:
            self.assertEqual(self.switch("default"), (200, {"status": "already", "model": "default"}))
        self.assertEqual(self.old.closes, 0)

    def test_unknown_and_invalid_names(self):
        code, result = self.switch("absent")
        self.assertEqual(code, 404)
        self.assertEqual(result["error"], {"message": "model not found", "known": ["default", "second", "third"]})
        for body in ({}, {"model": 1}, {"model": None}, {"model": []}, []):
            with self.subTest(body=body):
                self.assertEqual(self.request("/v1/models/switch", body)[0], 400)
        self.assertEqual(self.old.closes, 0)

    def test_busy_fifo_active_and_queued_parallel_requests(self):
        with self.svc.fifo:
            code, result = self.switch()
            self.assertEqual(code, 409)
            self.assertEqual(result["error"]["type"], "model_busy")
        for state in ({"busy": True}, {"queued": 1}):
            self.svc.status = {"busy": False, "queued": 0, **state}
            self.assertEqual(self.switch()[0], 409)
        self.assertEqual(self.svc.model, "default")
        self.assertEqual(self.old.closes, 0)

    def test_security_for_both_routes_even_with_key_and_cors(self):
        self.svc.api_key = "secret"
        self.svc.cors_origins = ["http://evil.example"]
        for path in ("/v1/models/switch", "/switch"):
            with self.subTest(path=path):
                self.assertEqual(self.switch(path=path)[0], 401)
                headers = {"Authorization": "Bearer secret", "Origin": "http://evil.example"}
                self.assertEqual(self.switch(path=path, headers=headers)[0], 403)
                headers = {"Authorization": "Bearer secret", "Content-Type": "text/plain"}
                self.assertEqual(self.switch(path=path, headers=headers)[0], 415)
        self.assertEqual(self.old.closes, 0)

    def test_control_twin_switches_and_validates_names(self):
        self.assertEqual(self.switch(path="/switch")[0], 200)
        self.assertEqual(self.chat("second"), "Second answer.")
        self.assertEqual(self.switch("absent", path="/switch")[0], 404)
        self.assertEqual(self.request("/switch", {})[0], 400)

    def test_discovery_lists_resident_aliases_and_switchable_ids(self):
        for path in ("/v1/models", "/models"):
            code, listing = self.request(path)
            self.assertEqual(code, 200)
            entries = {entry["id"]: entry for entry in listing["data"]}
            self.assertEqual(set(entries), {"default", "old-alias", "second", "third"})
            self.assertEqual(entries["default"]["status"]["value"], "loaded")
            self.assertEqual(entries["old-alias"]["alias_of"], "default")
            self.assertEqual(entries["second"]["status"]["value"], "switchable")
            self.assertTrue(entries["second"]["switchable"])
        self.assertEqual(self.request("/v1/status")[1]["switchable_models"], ["default", "second", "third"])
        self.switch()
        entries = {entry["id"]: entry for entry in self.request("/v1/models")[1]["data"]}
        self.assertEqual(entries["second"]["status"]["value"], "loaded")
        self.assertEqual(entries["new-alias"]["alias_of"], "second")
        self.assertEqual(entries["default"]["status"]["value"], "switchable")
        self.assertNotIn("old-alias", entries)

    def test_chat_does_not_auto_switch(self):
        self.assertEqual(self.chat("second"), "Default answer.")
        self.assertEqual(self.svc.model, "default")
        self.assertEqual(self.old.closes, 0)

    def test_runtime_state_resets_and_shared_settings_and_totals_stay(self):
        self.svc.rate.append((1, 5))
        self.svc.live_reqs[1] = ({}, [])
        self.svc.last_timings = {"predicted_n": 5}
        self.svc.status.update(tail="old reply", generated=5)
        self.svc.shared = {"temperature": 0.2}
        totals = self.svc.totals.copy()
        self.svc.reasoning_budget_tokens = 99
        with mock.patch.object(self.svc.conv_log, "reset") as reset:
            self.switch()
            reset.assert_called_once()
        self.assertFalse(self.svc.rate)
        self.assertFalse(self.svc.live_reqs)
        self.assertIsNone(self.svc.last_timings)
        self.assertNotIn("tail", self.svc.status)
        self.assertEqual(self.svc.shared, {"temperature": 0.2})
        self.assertEqual(self.svc.totals, totals)
        self.assertEqual(self.svc.sampling_defaults, {"temperature": 0.7})
        self.assertEqual(self.svc.reasoning_budget_tokens, 42)
        self.switch("third")
        self.assertEqual(self.svc.reasoning_budget_tokens, 0)
        self.assertEqual(self.svc.sampling_defaults, {})
        self.assertEqual(self.svc.aliases, [])

    def test_failed_start_cleans_up_and_rolls_back(self):
        self.native()
        factory = server.build_engine.side_effect

        def build(*args, **kwargs):
            engine, effort = factory(*args, **kwargs)
            if args[0]["model_name"] == "second":
                engine.restart = mock.Mock(side_effect=RuntimeError("exit before READY: out of RAM"))
            return engine, effort

        server.build_engine.side_effect = build
        code, result = self.switch()
        self.assertEqual(code, 503)
        self.assertEqual(result["error"]["type"], "server_error")
        self.assertIn("out of RAM", result["error"]["message"])
        self.assertIn("restored model default", result["error"]["message"])
        self.assertEqual(self.engines[0].closes, 1)
        self.assertEqual(self.events, ["build second", "build default"])
        self.assertEqual(self.svc.model, "default")
        self.assertIs(self.svc.current_spec, self.specs["default"])
        self.assertEqual(self.chat(), "Default answer.")

    def test_vram_guard_runs_after_unload_and_restores_old_on_gpu_busy(self):
        self.native()

        def guard(spec):
            self.assertFalse(self.old.alive())
            self.assertEqual(self.old.closes, 1)
            self.assertEqual(spec["model_name"], "second")
            raise GpuBusy("another program owns the GPU")

        with mock.patch.object(self.svc, "_wait_free_vram", side_effect=guard):
            code, result = self.switch()
        self.assertEqual(code, 503)
        self.assertIn("another program", result["error"]["message"])
        self.assertEqual(self.events, ["build default"])
        self.assertTrue(self.svc.loaded())
        self.assertEqual(self.chat(), "Default answer.")

    def test_guard_reads_the_target_gpu_after_unload(self):
        self.specs["second"].update(gpu=2, backend="hip")
        self.svc.min_free_vram_mib = 100

        def free(index, amd):
            self.assertFalse(self.old.alive())
            self.assertEqual((index, amd), (2, True))
            return 200

        with mock.patch("serve.telemetry.free_vram_mib", side_effect=free) as check, \
                mock.patch.object(self.svc.telemetry, "set_gpus") as monitor:
            self.assertEqual(self.switch()[0], 200)
            check.assert_called_once()
            monitor.assert_called_once_with(2, [2], amd=True)

    def test_stuck_old_engine_keeps_service_unchanged(self):
        with mock.patch.object(self.old, "close", side_effect=EngineStuck("still releasing")):
            self.assertEqual(self.switch()[0], 503)
        self.assertIs(self.svc.engine, self.old)
        self.assertEqual(self.svc.model, "default")
        self.assertTrue(self.svc.loaded())
        self.assertFalse(self.svc.fifo.locked())

    def test_only_started_engine_is_published_and_reserve_follows_ready(self):
        self.native()
        self.svc.vram_reserve = 512
        factory = server.build_engine.side_effect

        def build(*args, **kwargs):
            engine, effort = factory(*args, **kwargs)

            def ready():
                self.assertIs(self.svc.engine, self.old)
                self.assertEqual(self.svc.model, "default")
                self.assertEqual(engine.reserves, [])
                engine.running = True

            engine.restart = ready
            return engine, effort

        server.build_engine.side_effect = build
        self.assertEqual(self.switch()[0], 200)
        self.assertIs(self.svc.engine, self.engines[0])
        self.assertEqual(self.svc.engine.reserves, [512])

    def test_switch_refuses_a_request_preparing_its_prompt(self):
        preparing, proceed = threading.Event(), threading.Event()
        prepare = self.svc.prepare

        def paused_prepare(*args):
            preparing.set()
            if not proceed.wait(5):
                raise RuntimeError("test did not release prompt preparation")
            return prepare(*args)

        with ThreadPoolExecutor(max_workers=1) as pool, \
                mock.patch.object(self.svc, "prepare", side_effect=paused_prepare):
            request = pool.submit(self.chat)
            try:
                self.assertTrue(preparing.wait(5))
                self.assertEqual(self.switch()[0], 409)
                self.assertEqual(self.old.closes, 0)
            finally:
                proceed.set()
            self.assertEqual(request.result(5), "Default answer.")
        self.assertEqual(self.svc._model_requests, 0)

    def test_request_arriving_during_switch_waits_for_new_model(self):
        self.native()
        starting, proceed, arrived = threading.Event(), threading.Event(), threading.Event()
        factory = server.build_engine.side_effect
        admission = self.svc.model_request

        def build(*args, **kwargs):
            engine, effort = factory(*args, **kwargs)

            def ready():
                starting.set()
                if not proceed.wait(5):
                    raise RuntimeError("test did not release engine start")
                engine.running = True

            engine.restart = ready
            return engine, effort

        def model_request():
            arrived.set()
            return admission()

        server.build_engine.side_effect = build
        with ThreadPoolExecutor(max_workers=2) as pool, \
                mock.patch.object(self.svc, "model_request", side_effect=model_request):
            switch = pool.submit(self.switch)
            try:
                self.assertTrue(starting.wait(5))
                request = pool.submit(self.chat, "second")
                self.assertTrue(arrived.wait(5))
                self.assertFalse(request.done())
                self.assertEqual(self.svc.model, "default")
                self.assertIs(self.svc.engine, self.old)
            finally:
                proceed.set()
            self.assertEqual(switch.result(5)[0], 200)
            self.assertEqual(request.result(5), "Second answer.")

    def test_stuck_failed_replacement_is_kept_until_it_can_exit(self):
        self.native()
        factory = server.build_engine.side_effect

        def build(*args, **kwargs):
            engine, effort = factory(*args, **kwargs)
            engine.restart = mock.Mock(side_effect=RuntimeError("start failed"))
            engine.close = mock.Mock(side_effect=EngineStuck("replacement still exiting"))
            return engine, effort

        server.build_engine.side_effect = build
        code, result = self.switch()
        self.assertEqual(code, 503)
        self.assertIn("rollback must wait", result["error"]["message"])
        self.assertEqual(self.events, ["build second"])
        self.assertIs(self.svc._failed_start_engine, self.engines[0])
        self.engines[0].close.side_effect = None
        self.assertEqual(self.chat(), "Default answer.")
        self.assertIsNone(self.svc._failed_start_engine)
        self.assertEqual(self.old.starts, 1)

    def test_switch_from_lazy_unloaded_default(self):
        self.old.running, self.old.unloaded = False, True
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertEqual(self.switch()[0], 200)
        self.assertEqual(self.chat("second"), "Second answer.")

    def test_bad_tokenizer_rolls_back_before_new_engine_is_built(self):
        self.native()
        server.load_tokenizer.side_effect = lambda path: None if path == self.tpath else ByteTokenizer()
        code, result = self.switch()
        self.assertEqual(code, 503)
        self.assertIn("tokenizer is missing", result["error"]["message"])
        self.assertEqual(self.events, ["build default"])
        self.assertEqual(self.chat(), "Default answer.")

    def test_vision_starts_first_and_closes_on_failed_engine_start(self):
        self.native()
        self.specs["second"]["vision"] = {"exe": "fake"}
        vision = mock.Mock()
        factory = server.build_engine.side_effect

        def build(*args, **kwargs):
            self.assertEqual(self.events[0], "vision")
            engine, effort = factory(*args, **kwargs)
            if args[0]["model_name"] == "second":
                engine.restart = mock.Mock(side_effect=RuntimeError("bad engine"))
            return engine, effort

        def start_vision(*args):
            self.events.append("vision")
            return vision

        server.build_engine.side_effect = build
        with mock.patch.object(server, "build_vision", side_effect=start_vision):
            self.assertEqual(self.switch()[0], 503)
        vision.close.assert_called_once()
        self.assertEqual(self.chat(), "Default answer.")

    def test_vision_failure_rolls_back_without_building_target_engine(self):
        self.native()
        self.specs["second"]["vision"] = {"exe": "fake"}
        with mock.patch.object(server, "build_vision", side_effect=RuntimeError("encoder exited before READY")):
            code, result = self.switch()
        self.assertEqual(code, 503)
        self.assertIn("encoder exited", result["error"]["message"])
        self.assertEqual(self.events, ["build default"])
        self.assertEqual(self.chat(), "Default answer.")

    def test_successful_switch_replaces_and_closes_old_vision(self):
        self.native()
        self.specs["second"]["vision"] = {"exe": "fake"}
        old_vision, new_vision = mock.Mock(), mock.Mock()
        self.svc.vision = old_vision
        with mock.patch.object(server, "build_vision", return_value=new_vision):
            self.assertEqual(self.switch()[0], 200)
        old_vision.close.assert_called_once()
        new_vision.close.assert_not_called()
        self.assertIs(self.svc.vision, new_vision)

    def test_failed_encoder_start_ends_its_process(self):
        proc = mock.Mock()
        proc.stdout.readline.return_value = "ERR encoder failed\n"
        with mock.patch.object(server.subprocess, "Popen", return_value=proc), \
                mock.patch.object(server, "contain"), \
                mock.patch.object(server.tempfile, "mkdtemp", return_value=self.tmp.name):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                server.Vision({"exe": "fake", "mmproj": "fake", "model": "fake"})
        proc.kill.assert_called_once()
        proc.wait.assert_called_once_with(timeout=10)

    def test_failed_rollback_preserves_old_spawn_for_next_request(self):
        self.native()
        server.build_engine.side_effect = RuntimeError("start failed")
        code, result = self.switch()
        self.assertEqual(code, 503)
        self.assertIn("restoring model default also failed", result["error"]["message"])
        self.assertIs(self.svc.engine, self.old)
        self.assertFalse(self.svc.loaded())
        entries = {entry["id"]: entry for entry in self.request("/v1/models")[1]["data"]}
        self.assertEqual(entries["default"]["status"]["value"], "unloaded")
        self.assertEqual(self.chat(), "Default answer.")
        self.assertEqual(self.old.starts, 1)


if __name__ == "__main__":
    unittest.main()
