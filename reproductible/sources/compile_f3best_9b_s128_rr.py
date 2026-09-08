#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compile_f3best_9b_s128_rr.py — T-17 (02/09/2026).

Recompile le f3best 9B s128 a2_kv4 avec le code COURANT de repos_iron_windows
(HEAD 09-01 : lignée fixes #188 pure-C++ bcast, #208 decouple dropped par defaut,
#210 silu2, #211 single-B par defaut rock-solid, #272 prod tail sh7).

Contexte T-1/T-9 : le design s128 a2_kv4 `decouple_tb` du 16/08 (compile avec
F3BEST_MT_DECOUPLE=1, code PRE-fixes) sortait NO-OP NaN (3 valeurs uniques) alors
que le s256 a2_kv8 `tb_rr` (15/08, kernel RR, sans decouple) ecrit 510 valeurs.
Le code courant contient les fixes de fond (#208 decouple OFF par defaut,
single-B, C++ bcast) -> recompile AM-100 avec les DEFAUTS du code courant.

Geometrie AM-100 : embed_dim=4096 hidden_dim=12288 head_dim=256 attn_group=2
num_kv_heads=4 seq_len=128 num_q_heads=16 (= E/head_dim, regle #211 AM-100).

Usage: python compile_f3best_9b_s128_rr.py [build_dir]
"""
import sys
import logging
from pathlib import Path

repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("t17-am100")

from iron.common import AIEContext
from iron.operators.decode_layer_f3best.op import AIEDecodeLayerF3Best

build_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
    Path("build_t17_am100_s128").resolve())
build_dir.mkdir(parents=True, exist_ok=True)
log.info("build_dir: %s", build_dir)

ctx = AIEContext(build_dir=build_dir)

# AM-100 (Qwen3.5-9B full-attention) — defaults du code courant, PAS de decouple,
# PAS de triple-B (env non poses => single-B + C++ bcast + fixes #188/#208/#210).
op = AIEDecodeLayerF3Best(
    embed_dim=4096, hidden_dim=12288, K_gemv=4096, head_dim=256,
    group_size=32, attn_group=2, num_kv_heads=4, m_input=4, seq_len=128,
    num_q_heads=16, with_npu_kv=False, context=ctx,
)
log.info("op name prefix: decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4")

import traceback
try:
    op.set_up_artifacts()
    op.compile()
    for a in op.artifacts.dfs():
        fn = getattr(a, "filename", "?")
        if isinstance(fn, str) and fn.endswith((".xclbin", ".bin", ".mlir")):
            log.info("artifact: %s", fn)
    log.info("COMPILE OK")
except Exception as e:
    log.error("COMPILE FAILED: %s", e)
    traceback.print_exc()
    sys.exit(2)

log.info("DONE")
