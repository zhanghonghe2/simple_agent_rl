#!/usr/bin/env python3
"""The engine as an HTTP service: no Ray, no cluster, no handle shipping.

vLLM ships an RL online entrypoint that exposes the weight-transfer routes, and a
client that posts to them. That is the whole protocol: five URLs. The tensors ride
along as CUDA IPC handles, pickled into the JSON body, which is why the server has
to run with VLLM_ALLOW_INSECURE_SERIALIZATION=1.

So the framework is two process groups: a vLLM server and the trainer. Nothing
crosses a process boundary except HTTP, which is the one thing that always works.

    python3 -m zero_ppo.engine.http --self-test
"""
from __future__ import annotations

import argparse
import time


class HttpEngine:
    """Same verbs as the local engine, spoken over HTTP."""

    backend_name = "http"

    def __init__(self, base_url: str, *, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---- inference ---------------------------------------------------------
    def generate(self, prompts, temperature=0.7, max_tokens=1024):
        import requests
        out = []
        for p in prompts:
            r = requests.post(f"{self.base_url}/v1/completions",
                              json={"model": "default", "prompt": p,
                                    "temperature": temperature,
                                    "max_tokens": max_tokens},
                              timeout=self.timeout)
            r.raise_for_status()
            c = r.json()["choices"][0]
            out.append({"text": c["text"], "tokens": c.get("token_ids") or []})
        return out

    # ---- cache control -----------------------------------------------------
    # The server owns its own memory; there is nothing for the trainer to free or
    # re-take across the process boundary, so these are deliberate no-ops that keep
    # the engine interface uniform.
    def sleep(self):
        return None

    def wake(self):
        return None

    def pause(self):
        self._post("pause")

    def resume(self):
        self._post("resume")

    # ---- weight transfer ---------------------------------------------------
    def _post(self, path: str, body=None):
        import requests
        r = requests.post(f"{self.base_url}/{path}", json=body,
                          timeout=self.timeout)
        r.raise_for_status()

    def init_weight_transfer_engine(self, init_info):
        self._post("init_weight_transfer_engine", {"init_info": init_info})

    def start_weight_update(self):
        self._post("start_weight_update")

    def update_weights(self, request):
        self._post("update_weights", {"update_info": request})

    def finish_weight_update(self, weight_version=None):
        body = ({"weight_version": weight_version}
                if weight_version is not None else None)
        self._post("finish_weight_update", body)

    def shutdown_weight_transfer_engine(self):
        self._post("shutdown_weight_transfer_engine")

    # ---- liveness ----------------------------------------------------------
    def wait_ready(self, timeout: float = 600.0, interval: float = 2.0) -> bool:
        import requests
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if requests.get(f"{self.base_url}/health",
                                timeout=5).status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False


def _self_test() -> None:
    e = HttpEngine("http://127.0.0.1:8000/")
    assert e.base_url == "http://127.0.0.1:8000", e.base_url
    verbs = ("generate", "wait_ready", "sleep", "wake", "pause", "resume",
             "init_weight_transfer_engine",
             "start_weight_update", "update_weights", "finish_weight_update",
             "shutdown_weight_transfer_engine")
    missing = [v for v in verbs if not hasattr(e, v)]
    assert not missing, missing
    print("  engine.http self-test OK")
    print(f"    {len(verbs)} verbs, base url normalised, five POST routes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", dest="self_test")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        ap.print_help()
