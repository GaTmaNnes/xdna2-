# -*- coding: utf-8 -*-
"""b1_decompose_v2.py — B1 : décomposition temporelle, protocole ENTIÈREMENT SPÉCIFIÉ.

Contexte preuve (B1.1/B1.2, logs 08/09) :
  - sortie = f(contenu bo1..4) — PROUVÉ : zeros → 68029117..., all-AA → df78300f...
  - le témoin A2 (bc99cc09...) dépendait de contenus bo1..4 NON spécifiés par l'outil
    ( malloc déterministe par process mais pas un contrat ) → qualifié, pas reproductible tel quel.
Protocole B1 v2 (gelé pour ce test) :
  - xclbin/ctrlcode : témoin gelé A (t17rr_0906, SHA 02580f41...)
  - 5 BO data 4 Mo TOUS remplis de 0xAA (protocole 0xAA sans ambiguïté), insts BO cacheable
  - BOs PERSISTANTS (alloués 1×, sémantique boucle steady de xclbin_replay)
  - 3 warm-up + 10 dispatches mesurés ; phases par dispatch : submit, npu(wait), d2h, readout
  - préparation (alloc+fill+H2D) mesurée séparément (coût par-layer du runtime 9B)
Sortie : CSV + agrégats + NPU/total, transport/total, host_wait/total + cas de décision B3.
"""
import csv
import hashlib
import statistics
import sys
import time

import numpy as np
import pyxrt

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

XCLBIN = r"C:\t17rr_0906\decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.xclbin"
INSTS = r"C:\t17rr_0906\decode_layer_f3best_4096x12288_d256_g32_s128_a2_kv4_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac_nokv_tb_rr.bin"
BO_SIZE = 4 * 1024 * 1024
N_WARMUP = 3
N_MEASURED = 10
REF_ALLAA_SHA = "df78300f8b6364169706285ee1cc9aa0f6da921f722cc8d97a8ef5d6546d54f4"  # B1.2


def log(*a):
    print(*a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log("=" * 74)
    log("B1 v2 — décomposition temporelle, protocole spécifié (5 BO = 0xAA, BOs persistants)")
    log("=" * 74)
    sha_i = sha256_file(INSTS)
    log(f"ctrlcode: {sha_i[:16]}... (attendu 02580f41... = gel A)")
    assert sha_i.startswith("02580f41"), "ctrlcode != témoin gelé A"

    insts = open(INSTS, "rb").read()
    dev = pyxrt.device(0)
    xclbin = pyxrt.xclbin(XCLBIN)
    dev.register_xclbin(xclbin)
    hwctx = pyxrt.hw_context(dev, xclbin.get_uuid())
    kernel = pyxrt.kernel(hwctx, "MLIR_AIE")
    gids = [kernel.group_id(i) for i in range(8)]
    log("gids =", gids)

    # ---- T_prepare (mesurée une fois : alloc + fill + H2D) ----
    t0 = time.perf_counter()
    bo_insts = pyxrt.bo(dev, len(insts), pyxrt.bo.flags.cacheable, gids[1])
    bos = [pyxrt.bo(dev, BO_SIZE, pyxrt.bo.flags.host_only, gids[3 + j]) for j in range(5)]
    bo_insts.write(insts, 0)
    aa = b"\xAA" * BO_SIZE
    for b in bos:
        b.write(aa, 0)
    t1 = time.perf_counter()
    bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    for b in bos:
        b.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    t2 = time.perf_counter()
    log(f"T_alloc+fill = {(t1-t0)*1e6:.1f} us   T_H2D_sync = {(t2-t1)*1e6:.1f} us (1x, hors boucle)")

    run = pyxrt.run(kernel)
    run.set_arg(0, 3)
    run.set_arg(1, bo_insts)
    run.set_arg(2, len(insts))
    for j, b in enumerate(bos):
        run.set_arg(3 + j, b)

    rows = []
    for it in range(N_WARMUP + N_MEASURED):
        tag = "warmup" if it < N_WARMUP else f"run{it - N_WARMUP + 1:02d}"
        t3 = time.perf_counter()
        run.start()
        t4 = time.perf_counter()
        st = run.wait(60000)
        t5 = time.perf_counter()
        bos[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        t6 = time.perf_counter()
        out = np.frombuffer(bos[0].read(BO_SIZE, 0), dtype=np.uint8).tobytes()
        t7 = time.perf_counter()
        if int(st) != 4:
            log(f"{tag}: state={st} INATTENDU — arrêt")
            return 2
        rows.append({
            "tag": tag,
            "state": int(st),
            "sha_out": hashlib.sha256(out).hexdigest(),
            "us_submit": (t4 - t3) * 1e6,
            "us_npu_wait": (t5 - t4) * 1e6,
            "us_d2h_sync": (t6 - t5) * 1e6,
            "us_readout": (t7 - t6) * 1e6,
            "us_dispatch_cycle": (t7 - t3) * 1e6,
        })
        r = rows[-1]
        log(f"{tag}: cycle={r['us_dispatch_cycle']:9.1f}us  submit={r['us_submit']:7.1f}  "
            f"npu_wait={r['us_npu_wait']:8.1f}  d2h={r['us_d2h_sync']:7.1f}  read={r['us_readout']:7.1f}")

    meas = rows[N_WARMUP:]
    shas = set(r["sha_out"] for r in meas)
    deterministic = (len(shas) == 1)
    log("-" * 74)
    log(f"DETERMINISME : {len(shas)} SHA distinct sur {len(meas)} runs -> {deterministic}")
    log(f"SHA protocole specifie : {shas.pop() if deterministic else 'MULTIPLE'}")
    log(f"SHA attendu (B1.2)     : {REF_ALLAA_SHA}")

    def agg(key):
        v = [r[key] for r in meas]
        return (statistics.mean(v), statistics.median(v), min(v), max(v),
                statistics.stdev(v) if len(v) > 1 else 0.0)

    log("-" * 74)
    log(f"{'phase':<18}{'moy(us)':>10}{'med(us)':>10}{'min(us)':>10}{'max(us)':>10}{'sd':>8}{'%cycle':>8}")
    tot_m = agg("us_dispatch_cycle")[0]
    parts = ["us_submit", "us_npu_wait", "us_d2h_sync", "us_readout"]
    shares = {}
    for k in parts:
        m, med, mn, mx, sd = agg(k)
        shares[k] = m / tot_m * 100.0
        log(f"{k:<18}{m:>10.1f}{med:>10.1f}{mn:>10.1f}{mx:>10.1f}{sd:>8.1f}{shares[k]:>7.1f}%")
    m, med, mn, mx, sd = agg("us_dispatch_cycle")
    log(f"{'us_dispatch_cycle':<18}{m:>10.1f}{med:>10.1f}{mn:>10.1f}{mx:>10.1f}{sd:>8.1f}{'100.0':>8}")

    out_csv = r"E:\trixdna_test\docs\B1_DECOMPOSE_V2_08_09_2026.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"CSV écrit : {out_csv}")

    # Décomposition du cycle decode par layer (vue runtime 9B) :
    prep_m = (t1 - t0) * 1e6
    h2d_m = (t2 - t1) * 1e6
    npu_s = shares["us_npu_wait"]
    transport_s = shares["us_d2h_sync"]
    host_s = shares["us_submit"] + shares["us_readout"]
    log("-" * 74)
    log(f"COUTS 1x (par layer runtime) : alloc+fill={prep_m:.0f}us  H2D={h2d_m:.0f}us")
    log(f"PARTS du cycle dispatch      : NPU={npu_s:.1f}%  D2H={transport_s:.1f}%  host(submit+read)={host_s:.1f}%")
    if npu_s >= 80:
        case = "CAS 1 : NPU ~ cycle -> cible kernel/AIE"
    elif transport_s >= 50:
        case = "CAS 3 : DMA ~ cycle -> cible transport/DDR"
    elif host_s >= 50:
        case = "CAS 4 : submit/wait ~ cycle -> orchestration XRT/driver"
    else:
        case = "CAS 2 : NPU << cycle, wait/staging >> NPU -> runtime/XRT/BO"
    log("DECISION B3 :", case)
    log("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
