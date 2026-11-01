#!/bin/bash
# Fetch the elpis weight pair at pinned revisions (~17.8 GB) and verify sha256.
# usage: bash download_models.sh [models_dir=/models]
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:-/models}"
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download r0b0tlab/Qwen3.8-27B-EXL3-4.00bpw --revision 3f1771b8c21f83cbb8e82169559ced9f38ca04e5 \
  --local-dir "$M/qwen38-27b-exl3"
hf download r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw --revision 265b5240592907d2d55ff0dc4d5f66569692604d \
  --local-dir "$M/dflash2-exl3"
cd "$M" && sha256sum -c "$HERE/config/models.sha256"
