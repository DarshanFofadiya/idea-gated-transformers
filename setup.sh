#!/bin/bash
# Stops the script if any command fails
set -e 

echo "==================================================="
echo "   REPAIRING ENVIRONMENT FOR IDEA-GATED MODEL      "
echo "   (Mistral-7B + QLoRA + Triton 2.1.0)             "
echo "==================================================="

# 1. CLEANUP PHASE
# We remove everything that might be mismatched or corrupted.
echo ">>> [1/5] Uninstalling conflicting libraries..."
pip uninstall -y transformers peft accelerate bitsandbytes tokenizers triton pyarrow pandas wandb scipy datasets

# 2. SYSTEM UPDATE PHASE
# Ensure pip is new enough to find the correct binary wheels
echo ">>> [2/5] Upgrading pip..."
pip install --upgrade pip

# 3. BINARY DEPENDENCY PHASE
# We install PyArrow/Pandas first with strictly binary flags to avoid the CMake build error
echo ">>> [3/5] Installing Data Libraries (Binary Only)..."
pip install pyarrow pandas --only-binary=:all: --upgrade --no-cache-dir

# 4. DEEP LEARNING STACK PHASE
# The "Modern Stable" Stack we verified:
# - Transformers 4.41.2: Supports new Mistral Tokenizers
# - Peft 0.11.1: Compatible with 4.41.2
# - BitsAndBytes 0.43.1: Fast 4-bit quantization
# - Triton 2.1.0: CRITICAL. Newer versions break bitsandbytes.
echo ">>> [4/5] Installing ML Stack..."
pip install \
    transformers==4.41.2 \
    peft==0.11.1 \
    accelerate==0.31.0 \
    bitsandbytes==0.43.1 \
    tokenizers==0.19.1 \
    triton==2.1.0 \
    datasets \
    scipy \
    wandb \
    --no-cache-dir

# 5. VERIFICATION PHASE
echo ">>> [5/5] Verifying Installation..."
python -c "import torch; print(f'Torch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import triton; print(f'Triton: {triton.__version__} (Should be 2.1.0)')"
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'); print('Tokenizer Check: SUCCESS')"

echo "==================================================="
echo "   SETUP COMPLETE. YOU CAN RUN TRAIN.PY NOW.       "
echo "==================================================="
