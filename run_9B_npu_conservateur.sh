#!/usr/bin/env bash
# ============================================================
# ✅ CHEMIN 9B NPU CONSERVATEUR — Qwen3.5-9B Q4_0 — texte PROPRE
# GEMV INT4 + SwiGLU INT4 sur NPU, attention CPU
# Prompt 8.9 t/s | Generation 1.8 t/s (bound attention CPU)
# ============================================================
# POURQUOI CE MODE : le Qwen3.5-9B est HYBRIDE = 24 couches full
# attention (attn_qkv) + 8 couches linear attention (ssm_conv1d /
# ssm_a / attn_q). Les chemins QKV/flowkv/transformer_block ne
# gèrent pas les couches lin_attn → garbage immédiat. En ne gardant
# que GEMV INT4 + SwiGLU INT4 sur NPU, le texte devient PROPRE.
# Résultat vérifié : « Thinking Process: ... Capital City: Paris ... »
cd "$(dirname "$0")"

export PATH="$(pwd)/bin:$PATH"
export GGML_XDNA_CACHE_DIR="$(pwd)/kernels"
export GGML_XDNA_NUM_COLS="8"
export XDNA_DEBUG=1
export XDNA_ENABLE_GEMV_INT4=1
export XDNA_ENABLE_SWIGLU_INT4=1
unset XDNA_ENABLE_GEMV XDNA_ENABLE_SWIGLU XDNA_ENABLE_QKV XDNA_ENABLE_QKV_INT4_FUSED
unset XDNA_ENABLE_DECODE_BATCH XDNA_ENABLE_TRANSFORMER_BLOCK XDNA_ENABLE_FLOWKV_DECODE
unset XDNA_ENABLE_FUSED_LAYER XDNA_LAYER_FUSED XDNA_ENABLE_LAYER_F3BEST

MODEL="models/Qwen3.5-9B-q40-lmhead-f16.gguf"

./bin/llama-cli.exe -m "$MODEL" -n 96 -c 512 -ub 32 \
  -p "What is the capital of France?" \
  --no-mmap -fa off -ngl 100 --single-turn
