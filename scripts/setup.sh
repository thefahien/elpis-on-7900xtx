#!/bin/bash
# Build exllamav3-rocm + patched TabbyAPI for an RX 7900 XTX (gfx1100) on Ubuntu 24.04, ROCm 7.2.4.
# Tested in an Ubuntu 24.04 container with /dev/kfd and /dev/dri/renderD* passed through.
# usage: sudo bash setup.sh [install_dir=/opt/elpis-xtx] [models_dir=/models]
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-/opt/elpis-xtx}"
MODELS="${2:-/models}"
REPO=https://github.com/phoenixhaxor/exllamav3-rocm
REPO_COMMIT=360c9025188810a0ac9d434c837dc3cb6b8e39cd   # main as of 2026-09-26

apt-get update
apt-get install -y wget curl gnupg ca-certificates git build-essential python3.12 python3.12-venv python3.12-dev \
  libnuma-dev pciutils kmod zstd

# ROCm 7.2.4 from AMD's apt repo
mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor > /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2.4 noble main" \
  > /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' > /etc/apt/preferences.d/rocm-pin-600
apt-get update
apt-get install -y rocm-hip-sdk rocminfo rocm-smi-lib
echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf; ldconfig
/opt/rocm/bin/rocminfo | grep -E 'Marketing|gfx' | head -6 || true

mkdir -p "$DIR"; cd "$DIR"
[ -d exllamav3-rocm ] || git clone "$REPO"
cd exllamav3-rocm
git checkout -q "$REPO_COMMIT"
# 0001: host-side rsqrtf does not compile with ROCm 7.2.4's clang 22
# 0002: accept DFlash2 drafts that store codebooks as nn.Embedding (<key>.weight), e.g. r0b0tlab's
for p in "$HERE"/patches/*.patch; do git apply --check "$p" 2>/dev/null && git apply "$p" || echo "already applied: $p"; done
rm -f exllamav3/exllamav3_ext/attention.hip   # stale hipified copy would hide patch 0001

python3.12 -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"
pip install --upgrade pip
pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install "huggingface_hub[hf_transfer]" tokenizers "numpy>=2.1" rich typing_extensions safetensors ninja \
            pillow pyyaml marisa_trie pydantic "llguidance>=1.7.0"
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"

# ~10 min on 6 jobs; each compiler process can take 3-4 GB RAM, lower MAX_JOBS on small boxes
MAX_JOBS="${MAX_JOBS:-6}" ROCM_HOME=/opt/rocm bash rocm/scripts/build.sh
python -c "from exllamav3.ext import exllamav3_ext; print('exllamav3_ext ok')"

bash rocm/scripts/install_tabbyapi.sh "$DIR/tabbyAPI" "$MODELS"
cp "$HERE/config/config.yml" "$DIR/tabbyAPI/config.elpis-xtx.yml"
echo SETUP-DONE
