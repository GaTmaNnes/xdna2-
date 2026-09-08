# [CORRIGÉ 03/09/2026] FLOWKV_DECODE passé à 0 : le run 23:26 (02/09) routait vers FlowKV
# (XDNA_ENABLE_FLOWKV_DECODE=1), dont le JIT compile échoue (exit -1), et decode_layer_f3best
# n'était JAMAIS appelé (0 occurrence dans npu9b_gate_run.log). Le kernel full-K validé au replay
# (C:\t17fk, COMPLETED 4218 µs) n'était même pas consulté.
# Backup de l'original : npu9b_env_fullk.sh.bak_flowkv_on
# [CORRIGÉ 03/09 07:35] GGML_XDNA_PYTHON_CMD : le python par défaut (miniconda) n'a PAS
# le module iron → post-attn-fused JIT échoue exit 1 (stderr avalé). Bundle prebuilt
# placé dans le cache : post_attn_fused_v7_e4096_h12288_c4_g32 (combined.xclbin + main.insts).
export GGML_XDNA_PYTHON_CMD="E:\\trixdna_test\\ironenv_devel\\Scripts\\python.exe"
export GGML_XDNA_NUM_COLS=8
export XDNA_ENABLE_GEMV=1 XDNA_ENABLE_SWIGLU=1 XDNA_ENABLE_QKV=1 XDNA_ENABLE_DECODE_BATCH=1 XDNA_ENABLE_TRANSFORMER_BLOCK=1 XDNA_ENABLE_FLOWKV_DECODE=0 XDNA_ENABLE_GEMV_INT4=1 XDNA_ENABLE_SWIGLU_INT4=1 XDNA_ENABLE_FUSED_LAYER=1 XDNA_LAYER_FUSED=1 XDNA_ENABLE_LAYER_F3BEST=1 XDNA_ATTN_SUPPORTS=1 XDNA_LAYER_F3BEST_LIVE=1 XDNA_F3BEST_LOOP=1
export F3BEST_LEGACY_KEY=1 F3BEST_NUM_KV=4 F3BEST_SEQ=128 F3BEST_TRIPLE_B=1 F3BEST_HANDASM_RR=1
# [AJOUTÉ 03/09 23:15] AM-102 : kernel gate compilé (C:\t17gk, _ag) + runtime patché
# (suffixe _ag dans la clé cache + levée de _full_attn_excl). Active la fusion des
# 8 layers full-attn (blk.3/7/.../31) avec gate sigmoid on-chip.
export F3BEST_GATE_FULLATTN=1
export XDNA_DEBUG=1