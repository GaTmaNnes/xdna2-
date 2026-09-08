# MANIFEST — Reproductibility package (08/09/2026)

> **Goal** : allow anyone to re-execute the frozen-witness chain (angle A) and the temporal
> decomposition (B1) on the same Strix Point / XDNA2 machine, and verify every number by hash.
> Companion of the upstream conversation: the binaries-first layout was a deliberate choice
> (working-config reproducibility); the source side is provided here at **exact revisions**.

## 1. Provenance — exact source revisions

| Component | Repo / worktree | Commit (verified 08/09) | Role |
|---|---|---|---|
| f3best emitter (full-K backport) | `E:\tmp\iron_am100` (era worktree) | `42178241db02648c3d54438c5f58b4b265809662` (13/08/2026) | generates the ctrlcode `…rr.bin` |
| ggml-xdna backend (host dispatch) | `E:\trixdna_test\repos_soket1` | `dae0a1cefe6287ab65d49ebe94e5dbe5e36773cf` (02/09/2026, "fix(xdna): f3best dispatch TIMEOUT — ninstr=0 (format aie2txn transaction)") | host-side, replay tool build |
| IRON windows (compile script) | `E:\trixdna_test\repos_iron_windows` | `0efa9e9fd2e53b3cb4fb3bfd179c739bf684718a` (01/09/2026) | MLIR-AIE / aiecc pipeline |
| Compile script | `sources/compile_f3best_9b_s128_rr.py` | copied from `repos_iron_windows` | rebuild the .mlir/.xclbin/.bin |
| Emitter source | `sources/f3best_emit_nokv.py` | copied from `E:\tmp\iron_am100` worktree (commit above, **with full-K backport applied**) | weight formulas GEMV_T=256, NC_Q=16, DN=6 |
| pyxrt harness scripts | `scripts/b1_*.py` | session 08/09 | B1 measurement/diagnostics |
| Environment | Python 3.13.14, pyxrt (C:\Python313), XRT/Windows NPU driver | see B1 report §0 | runtime stack |

## 2. Frozen witness (angle A) — `temoin/`

| File | SHA256 (verified) |
|---|---|
| `temoin/decode_layer_f3best_…_rr.xclbin` | `a2c290faa839b2472d8135d46bed47b17228ae0e87b578431d19c1baeab9512b` (185 375 B) |
| `temoin/decode_layer_f3best_…_rr.bin` | `02580f411439e8dc81cd401b8bb235df1a48fcc73ad55b0632ce8bc21ac8bdb9` (3 172 B = 793 words) |
| `temoin/xclbin_replay.exe` | `cd0ca6bb0d139e5b0974d6e1c2d447a191b7b47648a5978282777372808fc404` |

Identity chain (angle A1): the frozen `.bin` is byte-identical to the 02/09 archive
`RUNTIME_FONCTIONNEL_02_09/kernels_s128_fullk/…rr.bin` → the 06/09 `t17rr_0906` build is the
full-K baseline (f3best 9B s128 a2_kv4 RR, `layer_fused_bcast_kc256_rr`).

## 3. Protocol — A2 replay (original tool, `xclbin_replay.exe`)

```
cd <dir with aa_4MB.bin>
X=<path to decode_layer_f3best_…_rr>
./xclbin_replay.exe "$X.xclbin" "$X.bin" 4194304 4194304 4194304 4194304 4194304 3 3172 3 bo0=aa_4MB.bin
```
Expected (3/3 runs, logs in `donnees/replay_run{1,2,3}.log`): first dispatch **state=4**,
steady ≈ 4905 µs/dispatch, bo0 post-csum `578da392cd6c188a`.
⚠️ **Scope amendment (B1 finding)**: the tool does not initialize bo1..bo4 → its output bytes
(`bc99cc09…`) are process-dependent (malloc leftovers), reproducible within one process only.
For a byte-level reproducibility contract use §4.

## 4. Protocol — B1 reproducible output contract (pyxrt harness, `scripts/b1_diag_allaa.py`)

- Inputs: frozen xclbin+bin above; **5 data BOs (4 MiB each) ALL filled with 0xAA**; insts BO cacheable.
- ABI: `arg0=3 (opcode), arg1=insts BO, arg2=3172, arg3..arg7=data BOs`; kernel `"MLIR_AIE"`.
- Expected output (bo0), **inter-process verified** (B1.2 vs B1 v2, 10/10 runs):
  - SHA256 = `df78300f8b6364169706285ee1cc9aa0f6da921f722cc8d97a8ef5d6546d54f4`
  - content = bf16 pattern `0x7593` over 71 680 bytes, rest 0xAA
  - state=4, npu_wait ≈ 4 890 µs (sd ≈ 24 µs)
- Negative controls (also reproducible): bo1..4 = zeros → all-zero output 73 728 B
  (SHA `68029117…`, `donnees/B1_DIAG_ZEROBO_OUT.bin`).
- Raw data: `donnees/B1_DECOMPOSE_V2_08_09_2026.csv` (13 dispatches, phase timings),
  `donnees/B1_V2_RUN_LOG_08_09_2026.txt`, `donnees/B1_DIAG_ALLAA_OUT.bin`.

## 5. B1 verdict (summary, full report in `rapports/B1_RAPPORT_DECOMPOSITION_08_09_2026.md`)

| Phase | mean (µs) | % cycle |
|---|---|---|
| submit (XRT) | 54.5 | 0.9% |
| **NPU_WAIT (host wait until completion — proxy of NPU time, driver/DMA not isolated)** | **4 889.8** (sd 23.9) | **78.5%** |
| D2H sync | 58.4 | 0.9% |
| readout (4 MiB forced) | 1 222.5 | 19.6% (≈20 µs useful in a real runtime) |
| dispatch cycle | 6 225.2 | 100% |

**Decision B3 = CAS 1 (corrected)**: NPU_WAIT dominates (78.5%) → the "XRT/BO staging" track is
closed as the main cause; priority goes to kernel / internal DMA / geometry. Cross-check:
npu_wait ≈ A2 steady 4905 µs (0.3%). Note: the 4 MB recipe used here is *under-fed* versus the
02/09 full-K recipe (arg5 = 190 000 000 B, else state=8 — see B3); the +16.3% historical gap
(4218 vs 4905 µs) is the B3 experimental hypothesis.

## 6. Verify everything

```bash
cd reproductible
sha256sum temoin/* donnees/B1_DIAG_ALLAA_OUT.bin donnees/B1_DECOMPOSE_V2_08_09_2026.csv
# puis exécuter §3 (tool) et/ou §4 (harness) — NPU must be idle
```
Full SHA256 list: `SHA256SUMS.txt` (regenerate: `sha256sum $(find . -type f | sort) > SHA256SUMS.txt`).

## 7. Not included (known gaps, same as the upstream repo discussion)

- ggml-xdna host-side source tree (backend dispatch, classifier, kernel cache) — only the
  revision hash is given; publication in progress upstream (see user's GitHub answer).
- MLIR build tree of `t17rr_0906` (194 files; regenerate via `sources/compile_f3best_9b_s128_rr.py`
  at commit `4217824` + `F3BEST_HANDASM_RR=1`, worktree `E:\tmp\iron_am100`).
- The 9B E2E runtime path is NOT covered by this package (angles C/D pending).
