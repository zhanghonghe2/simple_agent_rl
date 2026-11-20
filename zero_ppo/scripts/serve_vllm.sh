#!/usr/bin/env bash
# vLLM as a service, with the RL weight-transfer routes mounted.
#
# The routes are registered only when VLLM_SERVER_DEV_MODE is set (see
# vllm/entrypoints/launchers/api_server/routers.py), and the update route
# deserialises pickled CUDA IPC handles, which needs the insecure-serialisation
# opt-in. That is the whole configuration.
#
#   bash zero_ppo/scripts/serve_vllm.sh <model-path> [port] [tp]
set -euo pipefail

MODEL=${1:?usage: serve_vllm.sh <model-path> [port] [tp]}
PORT=${2:-8000}
TP=${3:-2}

export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

exec vllm serve "$MODEL" \
    --served-model-name default \
    --tensor-parallel-size "$TP" \
    --port "$PORT" \
    --gpu-memory-utilization 0.30 \
    --max-model-len 4096 \
    --disable-custom-all-reduce \
    --enable-prefix-caching
