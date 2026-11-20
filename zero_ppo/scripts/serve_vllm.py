#!/usr/bin/env python3
"""Serve vLLM with the RL weight-transfer routes mounted, the way verl does it.

verl does not shell out to `vllm serve`: it builds the engine in its own process
from AsyncEngineArgs, passing `weight_transfer_config` straight into the engine
config, then mounts vLLM's own FastAPI app and runs uvicorn
(`verl/workers/rollout/vllm_rollout/vllm_async_server.py`, run_server). That is the
only way the transfer config reaches the engine, which is why the CLI flag looked
ignored.

    VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        python3 -m zero_ppo.scripts.serve_vllm --model /data/... --port 8000
"""
from __future__ import annotations

import argparse
import inspect
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tensor-parallel-size", type=int, default=2,
                    dest="tensor_parallel_size")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.30,
                    dest="gpu_memory_utilization")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    dest="max_model_len")
    ap.add_argument("--served-model-name", default="default",
                    dest="served_model_name")
    ap.add_argument("--backend", default="ipc",
                    help="weight transfer backend: ipc for colocated cards")
    ap.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    a = ap.parse_args()

    os.environ.setdefault("VLLM_SERVER_DEV_MODE", "1")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    import uvicorn
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import build_app, init_app_state
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    # The transfer config goes into the engine config directly; that is the piece
    # `vllm serve` never forwards.
    engine_args = AsyncEngineArgs(
        model=a.model,
        served_model_name=a.served_model_name,
        tensor_parallel_size=a.tensor_parallel_size,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_model_len=a.max_model_len,
        enable_prefix_caching=True,
        disable_custom_all_reduce=True,
        enable_sleep_mode=True,
        enforce_eager=a.enforce_eager,
        trust_remote_code=True,
        weight_transfer_config={"backend": a.backend},
    )

    usage = UsageContext.OPENAI_API_SERVER
    vllm_config = engine_args.create_engine_config(usage_context=usage)
    print(f"  engine config: tp={a.tensor_parallel_size} "
          f"weight_transfer={a.backend} max_len={a.max_model_len}", flush=True)

    fn_args = set(inspect.signature(AsyncLLM.from_vllm_config).parameters)
    kwargs = {}
    for name, value in (("usage_context", usage), ("log_requests", False),
                        ("disable_log_stats", True)):
        if name in fn_args:
            kwargs[name] = value
    engine = AsyncLLM.from_vllm_config(vllm_config=vllm_config, **kwargs)

    app = build_app(engine_args)
    try:
        init_app_state(engine, app.state, engine_args)
    except TypeError as e:                       # signature differs by version
        print(f"  ! init_app_state: {e}\n"
              f"    signature: {inspect.signature(init_app_state)}", flush=True)
        raise

    print(f"  serving on {a.host}:{a.port} "
          f"(RL routes at /init_weight_transfer_engine …)", flush=True)
    try:
        uvicorn.run(app, host=a.host, port=a.port, log_level="info")
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
