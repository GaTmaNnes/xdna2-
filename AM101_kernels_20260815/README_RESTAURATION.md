# BACKUP KERNELS AM-101 — 15/08/2026

## KERNEL BINAIRE SAIN (VALIDE_0800, head0=0.80)
- decode_layer_f3best_tb_qdump7.xclbin : 190303 B, SHA256 CEB9D7F1F4D294DE5A03E167DBFA94DA6F821D7002BE1FDB6E921F7C63501C22
- decode_layer_f3best_tb_qdump7.insts : 3172 B
- Dé-entrelacement SEUL (kernel base kc256.s), head0=0.8009 propre, head4=NaN 488 (accumulateur)
- Clé cache : decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_qdump7

## RESTAURATION
Copy-Item ...\xclbin_VALIDE_0800\decode_layer_f3best_tb_qdump7.xclbin %LOCALAPPDATA%\ggml-xdna\xclbin\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_qdump7.xclbin
Copy-Item ...\xclbin_VALIDE_0800\decode_layer_f3best_tb_qdump7.insts  %LOCALAPPDATA%\ggml-xdna\xclbin\decode_layer_f3best_4096x12288_d256_g32_s256_a2_kv8_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_qdump7.insts
Flags run : GGML_XDNA_NUM_COLS=8 + XDNA_ENABLE_QKV/FUSED_LAYER/LAYER_FUSED/ENABLE_LAYER_F3BEST/ATTN_SUPPORTS/ENABLE_GEMV_INT4/ENABLE_SWIGLU_INT4/ML_DEBUG + XDNA_LAYER_F3BEST_LIVE=1 + F3BEST_QDUMP=1 (SANS F3BEST_HANDASM_RR, SANS F3BEST_HANDASM_CMLFIX)
Modèle : C:\Users\videl\Desktop\Qwen3.5-9B-Claude-4.6-HighIQ-THINKING-HERETIC-UNCENSORED.i1-Q4_K_S.gguf

## SOURCES
- layer_fused_bcast_kc256.s : ORIGINAL, hash BBACE4E3, 4930 B (jamais modifié)
- layer_fused_bcast_kc256.s.ORIG_AM101_20260815 : backup de l'original (identique BBACE4E3)
- layer_fused_bcast_kc256_rr.s : kernel RR (élimine NaN head4 quand bien placé), hash BE1B4966, 6122 B
- layer_fused_bcast_kc256_rr.s.orig_cml1 : version d'origine du RR, hash D62A0FC1, 6052 B
- layer_fused_bcast_kc256_fix.s : copie test (seed cml1) INVALIDÉE (4/4 mauvais placement), hash 6874CA99, 5184 B
- layer_fused_bcast_wrapper.cc : wrapper C++ (accumulation cross-chunk)

## FAITS
1. Dé-entrelacement q_proj = RÉSOLU (head0 339/72M → 0.80)
2. NaN head4 = accumulateur cml1/cmh1 (scalechk 1024/1024 finies, 7/8 heads OK)
3. Non-déterminisme compilation = placement BDs (99 octets adresses sur 190303)
4. Runtime = commit c9c5b75 (dé-entrelacement, sans gardes TUTO e515141)
