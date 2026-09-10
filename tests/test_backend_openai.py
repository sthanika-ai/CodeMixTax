"""OpenAI-compatible backend tests against a real local HTTP server.

The default backend talks HTTP, so mocking the client would leave the actual
request/response handling untested. These tests stand up a tiny stdlib server
that speaks enough of the OpenAI API to exercise the real code path: chat and
completion endpoints, usage accounting, concurrency, retry/backoff, and the
guarantee that a persistently failing row degrades to an error rather than
killing the sweep.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cmb_indic.config import ModelConfig

# State shared with the handler; reset per test by the fixture.
STATE: dict = {}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence test output
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send(200, {"data": [{"id": STATE.get("model_id", "test/model")}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")

        with STATE["lock"]:
            STATE["calls"] += 1
            n = STATE["calls"]
            STATE["seen"].append(req)

        # Fail the first `fail_first` calls to exercise retry/backoff.
        if n <= STATE.get("fail_first", 0):
            self._send(503, {"error": {"message": "service unavailable, loading model"}})
            return
        if STATE.get("always_fail"):
            self._send(500, {"error": {"message": "boom"}})
            return

        reply = STATE.get("reply", "B")
        usage = {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}

        if "chat/completions" in self.path:
            self._send(200, {
                "id": "c1", "object": "chat.completion", "model": req.get("model", ""),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": STATE.get("finish_reason", "stop"),
                }],
                "usage": usage,
            })
        elif "completions" in self.path:
            self._send(200, {
                "id": "c2", "object": "text_completion", "model": req.get("model", ""),
                "choices": [{
                    "index": 0, "text": reply,
                    "finish_reason": STATE.get("finish_reason", "stop"),
                }],
                "usage": usage,
            })
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture
def server():
    STATE.clear()
    STATE.update({"calls": 0, "seen": [], "lock": threading.Lock(), "reply": "B"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    STATE["base_url"] = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield STATE
    srv.shutdown()
    srv.server_close()


def _cfg(server, **kw) -> ModelConfig:
    params = {
        "key": "http-test",
        "hf_id": "test/model",
        "backend": "openai",
        "base_url": server["base_url"],
        "max_retries": 2,
        "concurrency": 4,
        "request_timeout": 15.0,
    }
    params.update(kw)
    return ModelConfig(**params)


def _backend(cfg):
    from cmb_indic.backends.openai_compat import OpenAICompatBackend

    return OpenAICompatBackend(cfg)


class TestChat:
    def test_single_request(self, server):
        with _backend(_cfg(server)) as be:
            out = be.generate([[{"role": "user", "content": "hi"}]])
        assert len(out) == 1
        assert out[0].text == "B"
        assert out[0].ok
        assert out[0].finish_reason == "stop"
        assert out[0].prompt_tokens == 11
        assert out[0].completion_tokens == 3
        assert out[0].latency_s > 0

    def test_order_preserved_under_concurrency(self, server):
        # 12 prompts, 4 in flight: results must still map 1:1 to inputs by position.
        prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(12)]
        with _backend(_cfg(server, concurrency=4)) as be:
            out = be.generate(prompts)
        assert len(out) == 12
        assert all(g.text == "B" for g in out)
        assert server["calls"] == 12

    def test_sampling_params_forwarded(self, server):
        cfg = _cfg(server, temperature=0.0, top_p=1.0, max_tokens=48, seed=7, stop=["END"])
        with _backend(cfg) as be:
            be.generate([[{"role": "user", "content": "hi"}]])
        sent = server["seen"][0]
        assert sent["max_tokens"] == 48
        assert sent["temperature"] == 0.0
        assert sent["seed"] == 7
        assert sent["stop"] == ["END"]
        assert sent["model"] == "test/model"

    def test_chat_template_kwargs_go_in_extra_body(self, server):
        # This is how thinking gets disabled on Qwen3.x via vLLM.
        cfg = _cfg(server, chat_template_kwargs={"enable_thinking": False})
        with _backend(cfg) as be:
            be.generate([[{"role": "user", "content": "hi"}]])
        assert server["seen"][0]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_truncation_is_reported(self, server):
        server["finish_reason"] = "length"
        with _backend(_cfg(server)) as be:
            out = be.generate([[{"role": "user", "content": "hi"}]])
        assert out[0].truncated


class TestCompletion:
    def test_base_model_uses_completions_endpoint(self, server):
        cfg = _cfg(server, prompt_style="completion")
        with _backend(cfg) as be:
            out = be.generate(["plain text prompt"])
        assert out[0].text == "B"
        assert "prompt" in server["seen"][0]
        assert "messages" not in server["seen"][0]

    def test_chat_prompt_flattened_when_style_is_completion(self, server):
        cfg = _cfg(server, prompt_style="completion")
        with _backend(cfg) as be:
            be.generate([[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}]])
        sent = server["seen"][0]["prompt"]
        assert "SYS" in sent and "USR" in sent


class TestResilience:
    def test_retries_then_succeeds(self, server):
        server["fail_first"] = 2  # two 503s, then OK
        with _backend(_cfg(server, max_retries=3)) as be:
            out = be.generate([[{"role": "user", "content": "hi"}]])
        assert out[0].ok
        assert out[0].text == "B"
        assert server["calls"] == 3

    def test_persistent_failure_becomes_row_error_not_exception(self, server):
        # The property that protects a long sweep: one dead row must not raise.
        server["always_fail"] = True
        with _backend(_cfg(server, max_retries=1)) as be:
            out = be.generate([[{"role": "user", "content": "hi"}]])
        assert len(out) == 1
        assert not out[0].ok
        assert out[0].text == ""
        assert out[0].error

    def test_failure_isolated_to_its_own_row(self, server):
        server["fail_first"] = 1
        with _backend(_cfg(server, max_retries=3, concurrency=1)) as be:
            out = be.generate([[{"role": "user", "content": f"q{i}"}] for i in range(3)])
        assert all(g.ok for g in out)


class TestProbe:
    def test_probe_reports_served_models(self, server):
        server["model_id"] = "google/gemma-3-12b-it"
        with _backend(_cfg(server)) as be:
            info = be.probe()
        assert info["server_reachable"] is True
        assert info["server_models"] == ["google/gemma-3-12b-it"]

    def test_probe_on_dead_endpoint_reports_unreachable(self):
        cfg = ModelConfig(
            key="dead", hf_id="x/y", backend="openai",
            base_url="http://127.0.0.1:9/v1", request_timeout=2.0,
        )
        with _backend(cfg) as be:
            info = be.probe()
        assert info["server_reachable"] is False
        assert info["server_probe_error"]

    def test_describe_includes_provenance(self, server):
        with _backend(_cfg(server, max_tokens=64)) as be:
            info = be.describe()
        assert info["backend"] == "openai-compat"
        assert info["max_tokens"] == 64
        assert info["base_url"] == server["base_url"]


class TestRunnerIntegration:
    def test_full_split_over_http(self, server, fake_splits, tmp_path):
        """A real split scored end to end through the HTTP backend."""
        from cmb_indic.config import RunConfig
        from cmb_indic.registry import DATASETS
        from cmb_indic.runner import run_split

        server["reply"] = "Answer: B"
        cfg = _cfg(server, key="http-model")
        run_cfg = RunConfig(
            data_dir=str(fake_splits),
            out_dir=str(tmp_path / "runs"),
            prompt_file="prompt.json",
        )
        res = run_split(DATASETS["mmlu_hineng"], cfg, run_cfg, run_id="http")

        assert res.n_rows == 4
        assert res.n_errors == 0
        # Gold cycles A,B,C,D; predicting B for all four gives exactly 25%.
        assert res.metrics.scores["accuracy"] == pytest.approx(25.0)
        assert res.metrics.diagnostics["unparsed_rate"] == 0.0

        meta = json.loads((res.out_dir / "run_meta.json").read_text())
        assert meta["backend"]["server_reachable"] is True
