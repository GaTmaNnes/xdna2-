# -*- coding: utf-8 -*-
"""b1_diag_allaa.py — Diagnostic B1.2 (UNE variable) : les 5 BO data remplis de 0xAA.

Hypothèse : xclbin_replay.exe (A2) a chargé le fichier 0xAA dans TOUS les BO data
(son csum étant probablement seedé par BO, les pre-csums bo1..4 diffèrent malgré
un contenu identique). Le kernel lit W/x dans plusieurs BO → sortie 0xA0CE-pattern.
Test : bo0..bo4 = 0xAA → SHA attendu = témoin bc99cc09...

Preuves d'appui (A2 + B1.1) :
  - témoin A2 : région 73 728 octets != 0xAA remplie de mots 0xA0CE (bf16), 2-octets gaps
  - B1.1 (bo1..4=0) : même région = zéros → la sortie dépend du CONTENU de bo1..4
"""
import hashlib
import time

import numpy as np
import pyxrt

XCLBIN = r"C:\t17rr_0906\decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.xclbin"
INSTS = r"C:\t17rr_0906\decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.bin"
WITNESS_SHA = "bc99cc09f320c9ea694b57776339db420a8add518691f4ab80386fbb618f5025"
BO_SIZE = 4 * 1024 * 1024

insts = open(INSTS, "rb").read()
dev = pyxrt.device(0)
xclbin = pyxrt.xclbin(XCLBIN)
dev.register_xclbin(xclbin)
hwctx = pyxrt.hw_context(dev, xclbin.get_uuid())
kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
gids = [kernel.group_id(i) for i in range(8)]
print("gids =", gids, flush=True)

bo_insts = pyxrt.bo(dev, len(insts), pyxrt.bo.flags.cacheable, gids[1])
bos = [pyxrt.bo(dev, BO_SIZE, pyxrt.bo.flags.host_only, gids[3 + j]) for j in range(5)]
bo_insts.write(insts, 0)
aa = b"\xAA" * BO_SIZE
for b in bos:
    b.write(aa, 0)                           # ← LA variable du test : TOUS les BO = 0xAA
bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
for b in bos:
    b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

run = pyxrt.run(kernel)
run.set_arg(0, 3)
run.set_arg(1, bo_insts)
run.set_arg(2, len(insts))
for j, b in enumerate(bos):
    run.set_arg(3 + j, b)
t0 = time.perf_counter()
run.start()
st = run.wait(60000)
t1 = time.perf_counter()
print(f"state={st} npu={(t1-t0)*1e6:.1f}us", flush=True)

bos[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
out = np.frombuffer(bos[0].read(BO_SIZE, 0), dtype=np.uint8).tobytes()
sha = hashlib.sha256(out).hexdigest()
open(r"E:\trixdna_test\docs\B1_DIAG_ALLAA_OUT.bin", "wb").write(out)
print("sha_out =", sha)
print("temoin  =", WITNESS_SHA)
print("MATCH   =", sha == WITNESS_SHA)

w = np.frombuffer(out, dtype=np.uint8)
nz = np.flatnonzero(w != 0xAA)
print("octets != 0xAA :", nz.size)
if nz.size:
    print("premiers octets:", w[:16].tolist())
