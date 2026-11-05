#!/bin/bash
# Host side. One container, one GPU: /dev/kfd plus a single render node.
#   radeon/start_container.sh <workdir> <render-node> [name]
#   radeon/start_container.sh /data/splatsim /dev/dri/renderD128
# <workdir> is mounted at /workspace; the repo is expected at <workdir>/src.
set -euo pipefail
WORK=$(realpath "$1")
NODE=$2
NAME=${3:-radeon-splatsim}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker run -d --name "$NAME" \
    --device=/dev/kfd \
    --device="$NODE" \
    --group-add "$(getent group video | cut -d: -f3)" \
    --group-add "$(getent group render | cut -d: -f3)" \
    --ipc=host \
    --shm-size=16g \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -v "$WORK":/workspace \
    -w /workspace/src \
    --entrypoint bash \
    "$IMAGE" -lc 'sleep infinity'
fi
docker start "$NAME" >/dev/null
docker exec "$NAME" python -c \
  "import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
