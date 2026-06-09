#!/usr/bin/env bash
###############################################################################
# SoulX-FlashHead + Gemini  ::  automated pod setup
#
# Target pod template:
#   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
#   (Python 3.11, CUDA 12.8, RTX 4090)
#
# What this does:
#   1. system packages (ffmpeg, libgl, git-lfs, micro)
#   2. clone SoulX-FlashHead
#   3. pin torch 2.7.1+cu128 (matches xformers 0.0.31 + flash-attn wheel)
#   4. install requirements (with the nvidia-nccl pin filtered out)
#   5. fix the blinker distutils conflict
#   6. install flash-attn from the prebuilt cp311/torch2.7/cu12 wheel
#   7. build SageAttention 2.x from source (the 96fps speed unlock)
#   8. install google-genai for the Gemini bridge
#   9. download FlashHead-1.3B + wav2vec2 weights (~15GB)
#  10. set up a persistent torch.compile cache on the volume
#  11. verify the whole stack
#
# YOU PROVIDE (place these in the project dir BEFORE or AFTER running):
#   - flashhead_gemini_server.py
#   - client.html
#
# Usage:
#   chmod +x setup_soulx.sh
#   ./setup_soulx.sh
#
# Re-runnable: steps that are already done are skipped where practical.
###############################################################################

set -e  # exit on error
set -o pipefail

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/SoulX-FlashHead"
REPO_URL="https://github.com/Soul-AILab/SoulX-FlashHead.git"
SAGE_URL="https://github.com/thu-ml/SageAttention.git"
FLASH_ATTN_WHEEL="flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/${FLASH_ATTN_WHEEL}"
COMPILE_CACHE_DIR="${WORKSPACE}/torch_compile_cache"

# Colors for readability
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; NC="\033[0m"
log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
err()  { echo -e "${RED}[setup]${NC} $*"; }

# ----------------------------------------------------------------------------
# Step 0 - sanity checks
# ----------------------------------------------------------------------------
log "Checking environment..."
echo "  nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'NOT FOUND')"
echo "  python: $(python --version 2>&1)"
python -c "import torch; print('  torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpu:', torch.cuda.is_available())" 2>/dev/null || warn "torch not importable yet (ok, base image should have it)"

if ! nvcc --version 2>/dev/null | grep -q "release 12.8"; then
    warn "nvcc is not 12.8 - this script targets CUDA 12.8. Proceeding anyway, but flash-attn/sage may fail if CUDA differs."
fi

# ----------------------------------------------------------------------------
# Step 1 - system packages
# ----------------------------------------------------------------------------
log "Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 git-lfs micro wget \
    > /dev/null
git lfs install
log "System packages done."

# ----------------------------------------------------------------------------
# Step 2 - clone the repo
# ----------------------------------------------------------------------------
cd "${WORKSPACE}"
if [ -d "${PROJECT_DIR}/.git" ]; then
    log "Repo already cloned at ${PROJECT_DIR}, skipping clone."
else
    log "Cloning SoulX-FlashHead..."
    git clone "${REPO_URL}" "${PROJECT_DIR}"
fi
cd "${PROJECT_DIR}"

# ----------------------------------------------------------------------------
# Step 3 - pin torch 2.7.1 + cu128 (matches xformers 0.0.31 & flash-attn wheel)
# ----------------------------------------------------------------------------
TORCH_VER="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)"
if [[ "${TORCH_VER}" == 2.7.1+cu128* ]]; then
    log "torch already 2.7.1+cu128, skipping torch install."
else
    log "Installing torch 2.7.1 + torchvision 0.22.1 (cu128)... (this replaces the base nightly)"
    pip install --quiet torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128
fi

# ----------------------------------------------------------------------------
# Step 4 - install requirements WITHOUT the nvidia-nccl pin
#          (requirements pins nccl 2.27.3 which conflicts with torch 2.7.1's 2.26.2)
# ----------------------------------------------------------------------------
log "Installing FlashHead requirements (nccl pin filtered out)..."
grep -v "nvidia-nccl-cu12" requirements.txt | sed "s/mediapipe==0.10.9/mediapipe==0.10.14/" > requirements_fixed.txt
# Try install; if blinker distutils error, fix and retry.
if ! pip install --quiet -r requirements_fixed.txt; then
    warn "requirements install hit an error (likely blinker distutils). Applying fix and retrying..."
    pip install --quiet --ignore-installed blinker
    pip install --quiet -r requirements_fixed.txt
fi
log "Requirements done."

# ----------------------------------------------------------------------------
# Step 5 - flash-attn from prebuilt wheel (skip the 30-min source compile)
# ----------------------------------------------------------------------------
if python -c "import flash_attn" 2>/dev/null; then
    log "flash_attn already installed, skipping."
else
    log "Installing flash-attn from prebuilt wheel..."
    pip install --quiet ninja
    if [ ! -f "${FLASH_ATTN_WHEEL}" ]; then
        wget -q "${FLASH_ATTN_URL}" -O "${FLASH_ATTN_WHEEL}" || {
            err "Failed to download flash-attn wheel. URL may have changed."
            err "Check https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2"
            err "and find the cp311 / torch2.7 / cu12 / cxx11abiFALSE wheel."
            exit 1
        }
    fi
    pip install --quiet "${FLASH_ATTN_WHEEL}" --no-build-isolation
fi

# ----------------------------------------------------------------------------
# Step 6 - SageAttention 2.x from source (the 96fps speed unlock)
# ----------------------------------------------------------------------------
if python -c "from sageattention import sageattn" 2>/dev/null; then
    log "SageAttention already installed, skipping."
else
    log "Building SageAttention from source (~5-15 min compile)..."
    cd "${WORKSPACE}"
    if [ ! -d "${WORKSPACE}/SageAttention/.git" ]; then
        git clone "${SAGE_URL}" "${WORKSPACE}/SageAttention"
    fi
    cd "${WORKSPACE}/SageAttention"
    export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
    python setup.py install
    cd "${PROJECT_DIR}"
fi

# ----------------------------------------------------------------------------
# Step 7 - google-genai for the Gemini bridge
# ----------------------------------------------------------------------------
if python -c "from google import genai" 2>/dev/null; then
    log "google-genai already installed, skipping."
else
    log "Installing google-genai..."
    pip install --quiet google-genai
fi

# ----------------------------------------------------------------------------
# Step 8 - download model weights (~15GB)
# ----------------------------------------------------------------------------
cd "${PROJECT_DIR}"
mkdir -p models
if [ -f "models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors" ]; then
    log "FlashHead weights already present, skipping download."
else
    log "Downloading SoulX-FlashHead-1.3B weights (~14GB, few minutes)..."
    huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
        --local-dir ./models/SoulX-FlashHead-1_3B
fi
if [ -f "models/wav2vec2-base-960h/pytorch_model.bin" ]; then
    log "wav2vec2 already present, skipping download."
else
    log "Downloading wav2vec2-base-960h (~1GB)..."
    huggingface-cli download facebook/wav2vec2-base-960h \
        --local-dir ./models/wav2vec2-base-960h
fi

# ----------------------------------------------------------------------------
# Step 9 - persistent torch.compile cache (so restarts skip the ~273s compile)
# ----------------------------------------------------------------------------
mkdir -p "${COMPILE_CACHE_DIR}"
log "torch.compile cache dir: ${COMPILE_CACHE_DIR}"

# Write a launch helper that sets all the right env vars then runs the server.
cat > "${PROJECT_DIR}/run_server.sh" << RUNEOF
#!/usr/bin/env bash
# Launch helper for the FlashHead+Gemini server.
# Sets the persistent compile cache + your API key, then runs the server.
cd "${PROJECT_DIR}"

# Persistent torch.compile cache (first run still compiles ~273s; later restarts reuse it)
export TORCHINDUCTOR_CACHE_DIR="${COMPILE_CACHE_DIR}"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

# Make sure cuDNN/CUDA libs from pip wheels are findable (matches Ditto-era lesson)
export LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:\${LD_LIBRARY_PATH}"

# >>> SET YOUR GEMINI KEY HERE (or export it before running) <<<
if [ -z "\${GEMINI_API_KEY}" ]; then
    echo "ERROR: GEMINI_API_KEY not set. Run:  export GEMINI_API_KEY='your_key'  then re-run."
    exit 1
fi

python flashhead_gemini_server.py
RUNEOF
chmod +x "${PROJECT_DIR}/run_server.sh"
log "Wrote launch helper: ${PROJECT_DIR}/run_server.sh"

# ----------------------------------------------------------------------------
# Step 10 - verify the stack
# ----------------------------------------------------------------------------
log "Verifying installed stack..."
python - << 'PYEOF'
def check(name, fn):
    try:
        v = fn()
        print(f"  OK   {name}: {v}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

import importlib
check("torch",        lambda: __import__('torch').__version__)
check("torch.cuda",   lambda: __import__('torch').cuda.is_available())
check("numpy",        lambda: __import__('numpy').__version__)
check("xformers",     lambda: __import__('xformers').__version__)
check("flash_attn",   lambda: __import__('flash_attn').__version__)
check("sageattention",lambda: (importlib.import_module('sageattention'), 'importable')[1])
check("diffusers",    lambda: __import__('diffusers').__version__)
check("transformers", lambda: __import__('transformers').__version__)
check("gradio",       lambda: __import__('gradio').__version__)
check("google-genai", lambda: (__import__('google.genai', fromlist=['genai']), 'importable')[1])
check("cv2",          lambda: __import__('cv2').__version__)
PYEOF

# ----------------------------------------------------------------------------
# Step 11 - check for the two user-provided files
# ----------------------------------------------------------------------------
echo ""
log "=============================================================="
log "Setup complete. Checking for YOUR two files in ${PROJECT_DIR}:"
MISSING=0
if [ -f "${PROJECT_DIR}/flashhead_gemini_server.py" ]; then
    log "  FOUND: flashhead_gemini_server.py"
else
    warn "  MISSING: flashhead_gemini_server.py  (place it in ${PROJECT_DIR})"
    MISSING=1
fi
if [ -f "${PROJECT_DIR}/client.html" ]; then
    log "  FOUND: client.html"
else
    warn "  MISSING: client.html  (place it in ${PROJECT_DIR})"
    MISSING=1
fi
log "=============================================================="
echo ""
if [ "${MISSING}" -eq 1 ]; then
    warn "Place the missing file(s) above, then launch with:"
else
    log "All files present. Launch with:"
fi
echo ""
echo "    export GEMINI_API_KEY='your_key_here'"
echo "    cd ${PROJECT_DIR} && ./run_server.sh"
echo ""
echo "    Then open RunPod HTTP port 7860 in your browser."
echo "    (First launch compiles ~273s at 'Priming...'; that's normal. Wait for 'Server ready'.)"
echo ""
log "Don't forget: set COND_IMAGE in flashhead_gemini_server.py to your avatar photo,"
log "and place that photo in ${PROJECT_DIR}/examples/ (or wherever COND_IMAGE points)."