# -*- coding: utf-8 -*-
"""b1_diag_zerobo.py — Diagnostic B1.1 (UNE variable) : bo1..bo4 remplis de ZÉROS.

Hypothèse : le témoin A2 (xclbin_replay.exe) avait des bo1..bo4 = zéros (pages fraîches) ;
le harnais B1 les laissait non initialisés → sortie ≠ témoin.
Test : 1 dispatch, bo1..4 = 0x00, bo0 = 0xAA (inchangé) → comparer SHA au témoin.
"""
import hashlib
import sys
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
bos[0].write(b"\xAA" * BO_SIZE, 0)          # bo0 = 0xAA (protocole gelé)
zeros = b"\x00" * BO_SIZE
for b in bos[1:]:
    b.write(zeros, 0)                        # ← LA variable du test
bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bos[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
for b in bos[1:]:
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
open(r"E:\trixdna_test\docs\B1_DIAG_ZEROBO_OUT.bin", "wb").write(out)
print("sha_out =", sha)
print("temoin  =", WITNESS_SHA)
print("MATCH   =", sha == WITNESS_SHA)

# structure de la région modifiée vs 0xAA
w = np.frombuffer(out, dtype=np.uint8)
nz = np.flatnonzero(w != 0xAA)
print("octets != 0xAA :", nz.size)
if nz.size:
    breaks = np.flatnonzero(np.diff(nz) > 1)
    starts = np.concatenate(([0], breaks + 1)); ends = np.concatenate((breaks, [nz.size - 1]))
    segs = [(int(nz[s]), int(nz[e])) for s, e in zip(starts, ends)]
    print("segments:", segs[:8], "total =", len(segs))
    print("premiers octets:", w[:16].tolist())
