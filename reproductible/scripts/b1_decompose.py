# -*- coding: utf-8 -*-
"""b1_decompose.py — B1 : décomposition temporelle du témoin gelé C:\\t17rr_0906 (f3best 9B RR full-K).

Gouvernance (08/09) : A = CLOSED/FROZEN ; B = ACTIVE ; B1 = décomposition temporelle du témoin.
Interdits : aucune modification du xclbin/ctrlcode/replay ; buffers protocole A2 inchangés
(5 BO 4 Mo, bo0 = 4 Mo de 0xAA) ; allocation normale (pas de pooling).

ABI (retrouvée : /e/tmp/flowkv_direct_hw.py + RAPPORT_REPRODUCTIBLE_02_09.md §3.5) :
    arg0 = opcode (3)            arg1 = BO instructions (cacheable)
    arg2 = insts size (octets)   arg3..arg7 = 5 BO data 4 Mo (bo0 chargé depuis fichier 0xAA)
Kernel : "MLIR_AIE" (un seul kernel dans le xclbin).

Validation du harnais : la sortie bo0 doit reproduire EXACTEMENT le témoin gelé A2 :
    post-csum bo0 = 578da392cd6c188a   SHA256 = bc99cc09...
Phases mesurées par dispatch :
    t0 entrée | t1 prep buffers | t2 sync H2D | t3 submit | t4 completion (NPU)
    t5 sync D2H | t6 lecture output
Sortie : CSV phase par phase + verdict NPU/total, transport/total, host_wait/total.
"""
import faulthandler
faulthandler.enable()
import csv
import hashlib
import os
import statistics
import sys
import time

import numpy as np
import pyxrt

# ---- Témoin gelé (aucun paramètre modifiable) ----
XCLBIN = r"C:\t17rr_0906\decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.xclbin"
INSTS = r"C:\t17rr_0906\decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.bin"
BO0_FILE = r"E:\trixdna_test\repos_soket1\build\bin\aa_4MB.bin"  # 4 MiB de 0xAA (SHA 9f0d861b...)
WITNESS_SHA = "bc99cc09f320c9ea694b57776339db420a8add518691f4ab80386fbb618f5025"
WITNESS_CSUM = "578da392cd6c188a"

BO_SIZE = 4 * 1024 * 1024
N_WARMUP = 3
N_MEASURED = 10

# csum "adler-like" 64-bit de xclbin_replay (fletcher64, pour comparabilité) :
#   xclbin_replay affiche pre/post-csum ; on recalcule le même fletcher-64.
def fletcher64(data: bytes) -> str:
    lo = 0xFFFFFFFF
    hi = 0xFFFFFFFF
    n = len(data)
    i = 0
    # traiter par blocs de 8 octets (2 mots u32, little-endian)
    while i + 8 <= n:
        w1 = int.from_bytes(data[i:i + 4], "little")
        w2 = int.from_bytes(data[i + 4:i + 8], "little")
        lo = (lo + w1) % 0xFFFFFFFF
        hi = (hi + lo) % 0xFFFFFFFF
        lo = (lo + w2) % 0xFFFFFFFF
        hi = (hi + lo) % 0xFFFFFFFF
        i += 8
    if i < n:  # reste (ne devrait pas arriver ici, 4 Mo multiple de 8)
        while i < n:
            lo = (lo + data[i]) % 0xFFFFFFFF
            hi = (hi + lo) % 0xFFFFFFFF
            i += 1
    return f"{(hi << 32) | lo:016x}"

WARMUP_CSUM = fletcher64(b"\xAA" * BO_SIZE)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(*a):
    print(*a, flush=True)


def main():
    log("=" * 74)
    log("B1 — décomposition temporelle du témoin gelé (f3best 9B RR full-K, 0xAA)")
    log("=" * 74)

    # Preuves d'entrée (identité du témoin)
    sha_x = sha256_file(XCLBIN)
    sha_i = sha256_file(INSTS)
    sha_b0 = sha256_file(BO0_FILE)
    log(f"xclbin      : {sha_x[:16]}... ({os.path.getsize(XCLBIN)} B)")
    log(f"ctrlcode    : {sha_i[:16]}... ({os.path.getsize(INSTS)} B)  attendu 02580f41... (gel A)")
    assert sha_i.startswith("02580f41"), "ctrlcode != témoin gelé A — ABANDON"
    log(f"input 0xAA  : {sha_b0[:16]}... ({os.path.getsize(BO0_FILE)} B)  attendu 9f0d861b... (A2)")
    assert sha_b0.startswith("9f0d861b"), "input 0xAA modifié — ABANDON"

    # Vérifier que notre fletcher64 reproduit le checksum affiché par xclbin_replay
    # sur l'input (pre-csum bo0 = 524d3927699d0383, observé aux runs A2)
    calc = fletcher64(b"\xAA" * BO_SIZE)
    log(f"fletcher64(0xAA 4MB) = {calc}  (replay A2 affichait pre-csum 524d3927699d0383)")
    fletcher_ok = (calc == "524d3927699d0383")
    log(f"fletcher64 == replay csum : {fletcher_ok}")

    # ---- Setup XRT (hors chronométrage) ----
    insts = open(INSTS, "rb").read()
    dev = pyxrt.device(0)
    xclbin = pyxrt.xclbin(XCLBIN)
    dev.register_xclbin(xclbin)
    hwctx = pyxrt.hw_context(dev, xclbin.get_uuid())
    kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
    gids = [kernel.group_id(i) for i in range(8)]
    log("gids =", gids)

    # ---- Boucle de dispatch (t0..t6 par dispatch) ----
    rows = []
    for it in range(N_WARMUP + N_MEASURED):
        tag = "warmup" if it < N_WARMUP else f"run{it - N_WARMUP + 1:02d}"

        t0 = time.perf_counter()
        bo_insts = pyxrt.bo(dev, len(insts), pyxrt.bo.flags.cacheable, gids[1])
        bos = [pyxrt.bo(dev, BO_SIZE, pyxrt.bo.flags.host_only, gids[3 + j]) for j in range(5)]
        bo_insts.write(insts, 0)
        data0 = open(BO0_FILE, "rb").read()
        bos[0].write(data0, 0)
        t1 = time.perf_counter()

        bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        t2 = time.perf_counter()

        run = pyxrt.run(kernel)
        run.set_arg(0, 3)
        run.set_arg(1, bo_insts)
        run.set_arg(2, len(insts))
        for j, b in enumerate(bos):
            run.set_arg(3 + j, b)
        run.start()
        t3 = time.perf_counter()

        state = run.wait(60000)
        t4 = time.perf_counter()

        bos[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t5 = time.perf_counter()

        out = np.frombuffer(bos[0].read(BO_SIZE, 0), dtype=np.uint8).tobytes()
        t6 = time.perf_counter()

        if state != 4:
            log(f"{tag}: ÉTAT INATTENDU state={state} — arrêt")
            sys.exit(2)

        rec = {
            "tag": tag,
            "state": int(state),
            "sha_out": hashlib.sha256(out).hexdigest(),
            "csum_out": fletcher64(out),
            "us_prepare": (t1 - t0) * 1e6,
            "us_h2d": (t2 - t1) * 1e6,
            "us_submit": (t3 - t2) * 1e6,
            "us_npu": (t4 - t3) * 1e6,
            "us_d2h": (t5 - t4) * 1e6,
            "us_readout": (t6 - t5) * 1e6,
            "us_total": (t6 - t0) * 1e6,
        }
        rows.append(rec)
        log(f"{tag}: state=4 total={rec['us_total']:9.1f}us  prep={rec['us_prepare']:8.1f}  "
            f"h2d={rec['us_h2d']:8.1f}  submit={rec['us_submit']:8.1f}  npu={rec['us_npu']:8.1f}  "
            f"d2h={rec['us_d2h']:8.1f}  read={rec['us_readout']:7.1f}  csum={rec['csum_out']}")
        del run, bos, bo_insts

    # ---- Validation témoin : toutes les sorties mesurées == témoin gelé ----
    meas = rows[N_WARMUP:]
    all_match = all(r["sha_out"] == WITNESS_SHA for r in meas)
    csum_match = all(r["csum_out"] == WITNESS_CSUM for r in meas)
    log("-" * 74)
    log(f"VALIDATION TÉMOIN : SHA256 sortie == bc99cc09... : {all_match}")
    log(f"                    csum == 578da392cd6c188a     : {csum_match}")
    log(f"                    fletcher64 calculé == replay : {fletcher_ok}")

    # ---- Agrégats ----
    def agg(key):
        vals = [r[key] for r in meas]
        return statistics.mean(vals), statistics.median(vals), min(vals), max(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    tot_m, tot_med, tot_min, tot_max, tot_sd = agg("us_total")
    log("-" * 74)
    log(f"{'phase':<12}{'moy(us)':>10}{'med(us)':>10}{'min(us)':>10}{'max(us)':>10}{'sd':>8}{'%total':>8}")
    parts = ["us_prepare", "us_h2d", "us_submit", "us_npu", "us_d2h", "us_readout"]
    shares = {}
    for k in parts:
        m, med, mn, mx, sd = agg(k)
        shares[k] = m / tot_m * 100.0
        log(f"{k:<12}{m:>10.1f}{med:>10.1f}{mn:>10.1f}{mx:>10.1f}{sd:>8.1f}{shares[k]:>7.1f}%")
    log(f"{'us_total':<12}{tot_m:>10.1f}{tot_med:>10.1f}{tot_min:>10.1f}{tot_max:>10.1f}{tot_sd:>8.1f}{'100.0':>8}")

    # ---- CSV (preuve brute) ----
    out_csv = r"E:\trixdna_test\docs\B1_DECOMPOSE_T17RR0906_08_09_2026.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"CSV écrit : {out_csv}")

    # ---- Cas de décision (gouvernance B3) ----
    npu_s = shares["us_npu"]
    transport_s = shares["us_h2d"] + shares["us_d2h"]
    wait_s = shares["us_submit"] + shares["us_readout"]
    log("-" * 74)
    log(f"NPU/total       = {npu_s:.1f}%")
    log(f"transport/total = {transport_s:.1f}%  (H2D {shares['us_h2d']:.1f}% + D2H {shares['us_d2h']:.1f}%)")
    log(f"host_wait/total = {wait_s:.1f}%")
    if npu_s >= 80:
        case = "CAS 1 : NPU ≈ total → cible = kernel/AIE"
    elif transport_s >= 50:
        case = "CAS 3 : DMA ≈ total → cible = transport/DDR"
    elif wait_s >= 50:
        case = "CAS 4 : submit/wait ≈ total → orchestration XRT/driver"
    else:
        case = "CAS 2 : NPU << total, wait/staging >> NPU → runtime/XRT/BO"
    log("DÉCISION :", case)

    # ---- Statut de la preuve ----
    ok = all_match and csum_match
    log("=" * 74)
    log(f"STATUT PREUVE : {'VALIDÉE — témoin reproduit bit-exact par le harnais' if ok else 'PARTIELLE — harnais ne reproduit pas le témoin (voir csum/SHA)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
