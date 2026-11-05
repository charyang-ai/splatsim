#!/bin/bash
# Inside the container, from anywhere. Installs SplatSim on top of the image's ROCm PyTorch.
#   ROCM_HEADERS=/workspace/rocm-headers/include bash radeon/install.sh
# Optional: PIP_INDEX_URL, APT_MIRROR (e.g. http://mirrors.aliyun.com/ubuntu/), PYTORCH_ROCM_ARCH (default gfx1100), MAX_JOBS.
set -euo pipefail
cd "$(dirname "$0")/.."
HDR=${ROCM_HEADERS:?run radeon/fetch_rocm_headers.sh and point ROCM_HEADERS at its include directory}
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx1100}
export MAX_JOBS=${MAX_JOBS:-8}
PIP="python -m pip install"

# Xvfb stands in for the monitor that launch_nodes.py (PyBullet GUI) and run_env_sim.py (cv2.imshow) expect.
if [ -n "${APT_MIRROR:-}" ]; then
  sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#${APT_MIRROR}#g" /etc/apt/sources.list.d/ubuntu.sources
fi
apt-get update
apt-get install -y --no-install-recommends \
  xvfb libgl1 libglx-mesa0 libgl1-mesa-dri libegl1 libglib2.0-0 libsm6 libice6 libxext6 libxrender1 \
  libfontconfig1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
  libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 ffmpeg build-essential swig

TORCH=$(python -c "import torch; print(torch.__version__)")
echo "torch before: $TORCH"

# hipcc finds hipCUB/rocPRIM/rocThrust through CPATH; the wheel ships libamdhip64.so.N without the link name.
export CPATH="$HDR${CPATH:+:$CPATH}"
LINKDIR="$PWD/build/rocm-link"
mkdir -p "$LINKDIR"
HIP_SO=$(python - <<'PY'
import glob, os, site
pats = [os.path.join(p, "_rocm_sdk_core", "lib", "libamdhip64.so*") for p in site.getsitepackages()]
pats.append("/opt/rocm/lib/libamdhip64.so*")
hits = sorted(h for p in pats for h in glob.glob(p))
print(hits[0] if hits else "")
PY
)
[ -n "$HIP_SO" ] || { echo "libamdhip64 not found"; exit 1; }
ln -sfn "$HIP_SO" "$LINKDIR/libamdhip64.so"
export LIBRARY_PATH="$LINKDIR${LIBRARY_PATH:+:$LIBRARY_PATH}"

python radeon/patch_submodules.py

# Same order as the upstream README. The two CUDA extensions build against the installed torch.
$PIP --no-build-isolation submodules/gaussian-splatting-wrapper/gaussian_splatting/submodules/diff-gaussian-rasterization
$PIP submodules/pybullet-URDF-models submodules/pybullet-playground-wrapper/ submodules/ghalton
$PIP -e submodules/gaussian-splatting-wrapper
$PIP -e submodules/gello_software
$PIP --no-build-isolation submodules/simple-knn/

# requirements.txt minus 3dgsconverter (a GitHub-only tool for CloudCompare exports, not on the replay path),
# plus what the replay path imports but the file does not list.
grep -v '^git+' requirements.txt > build/requirements-radeon.txt
$PIP -r build/requirements-radeon.txt mujoco pyzmq tyro termcolor pyyaml
$PIP -e .

AFTER=$(python -c "import torch; print(torch.__version__)")
[ "$AFTER" = "$TORCH" ] || { echo "torch changed: $TORCH -> $AFTER"; exit 1; }
python - <<'PY'
import torch
import diff_gaussian_rasterization, simple_knn._C, urdf_models, pybullet, gaussian_splatting, gello, splatsim
print("imports ok", torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))
PY
