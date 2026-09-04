#!/usr/bin/env bash
# ============================================================
# ✅ CHEMIN 1B NPU PROPRE — Llama-3.2-1B Q4_0 — 5.2 t/s
# Binaire : repo_0808 (18b583a) per-op — TEXTE PROPRE vérifié
# ============================================================
# Ce dossier est autonome : binaire + DLLs + kernels + modèle.
# Résultat attendu (04/09) : « The capital of France is Paris. »
# Prompt ~150-170 t/s | Generation ~5.2 t/s (texte propre)
cd "$(dirname "$0")"

export PATH="$(pwd)/bin:$PATH"
export GGML_XDNA_CACHE_DIR="$(pwd)/kernels"
export GGML_XDNA_NUM_COLS="8"

# Chemin per-op complet (garbage = QKV/flowkv sur couches hybrides — pas ici, 1B = Llama pur)
export XDNA_ENABLE_GEMV=1
export XDNA_ENABLE_SWIGLU=1
export XDNA_ENABLE_QKV=1
export XDNA_ENABLE_DECODE_BATCH=1
export XDNA_ENABLE_TRANSFORMER_BLOCK=1
export XDNA_ENABLE_FLOWKV_DECODE=1
export XDNA_ENABLE_GEMV_INT4=1
export XDNA_ENABLE_SWIGLU_INT4=1
export XDNA_ENABLE_FUSED_LAYER=0
export XDNA_LAYER_FUSED=0
export XDNA_ENABLE_LAYER_F3BEST=0

MODEL="models/Llama-3.2-1B-Instruct-Q4_0.gguf"

# -c 2048 : le ctx par défaut exige ~30 Go RAM host. 2048 → ~1,2 Go.
./bin/llama-cli.exe -m "$MODEL" -n 48 -c 2048 \
  -p "What is the capital of France?" \
  --no-mmap -fa off -ngl 100 --single-turn
