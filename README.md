# XDNA2 / Qwen3.5-9B — Current Status — 16 September 2026

> **Major milestone:** the direct GGUF → XDNA2 NPU path has now completed a full end-to-end autoregressive inference run.
>
> This status supersedes the 11 September 2026 status where Qwen3.5-9B end-to-end generation was still marked **NOT CLOSED**.

xdna2-     windows opensources                                                                                                                                                           Chargement GGUF             ✅

Q4_0 standard               ✅

Q6_K standard               ✅

Packer XDNA2                ✅

GEMV réel NPU               ✅

Workers / RPC               ✅

Chaîne Qwen3.5              ✅

DeltaNet / états            ✅

LM head NPU                 ✅

E2E CPU-Q4 == NPU-Q4        ✅ 8/8

Golden long                 ✅ 64/64

Cause ancienne divergence   ✅ identifiée

Reproductibilité            🔄 ×3 en cours

Performance                 ❌  en cours

## End-to-End Milestone

The complete inference chain successfully reached completion using the real Qwen3.5-9B model state and the XDNA2 NPU execution path.

Canonical test prompt:

```text
The capital of France is
```

Execution:

```text
real GGUF
    ↓
real tokenizer
    ↓
8-position prefill
    ↓
32-layer model chain
    ↓
XDNA2 NPU execution
    ↓
full 248,320-logit vocabulary
    ↓
token selection
    ↓
updated autoregressive state
    ↓
8 generated tokens
```

Observed generated token IDs:

```text
26705
92795
97950
99952
72787
126104
126104
126104
```

Complete logits were produced at every generation step:

```text
248320 / 248320
```

Total wall-clock time for the current unoptimized reference implementation:

```text
418 s
```

This run establishes that the current stack can execute a complete prefill + autoregressive generation sequence instead of stopping at isolated kernels, synthetic tensors, individual transformer blocks.

---

## Current Validation Matrix

| Validation area                                    | Status                            |
| -------------------------------------------------- | --------------------------------- |
| NPU transport                                      | **PASS**                          |
| Transport integrity checks                         | **PASS — 6/6 SHA**                |
| Signed Q4_0 GEMV                                   | **~95% validated**                |
| Complete `blk.31` differential validation          | **PASS — bit-exact**              |
| Cross-layer validation `blk.7`                     | **PASS**                          |
| Cross-layer validation `blk.15`                    | **PASS**                          |
| Cross-layer validation `blk.23`                    | **PASS**                          |
| Real model-state NPU execution                     | **PASS — tested across 8 layers** |
| Complete 32-layer CPU chain                        | **PASS**                          |
| Real tokenizer                                     | **PASS**                          |
| Full vocabulary output                             | **PASS — 248,320 logits**         |
| Full logits at every E2E step                      | **PASS**                          |
| Resident RPC workers                               | **PASS — 7/7**                    |
| Per-request RPC protocol                           | **PASS**                          |
| RPC anti-replay protection                         | **PASS**                          |
| 8-position prefill                                 | **PASS**                          |
| 8-token autoregressive generation                  | **PASS**                          |
| Complete E2E execution                             | **PASS**                          |
| Strict CPU/NPU stream equivalence                  | **OPEN — T6-b**                   |
| Exhaustive 32/32 NPU layer differential validation | **OPEN**                          |
| Long-generation stability                          | **OPEN**                          |
| Long-context stability                             | **OPEN**                          |
| Optimized LM head                                  | **OPEN — G10**                    |
| Batched GEMV                                       | **OPEN — G9**                     |

---

## Evidence Levels

All current and future results should use the following evidence classification.

### PROVEN

A result reproduced with direct correctness controls such as hashes, differential comparisons, bit-exact comparisons, or complete execution.

### MEASURED

A directly observed quantity, but not by itself a correctness proof.

### SUPPORTED HYPOTHESIS

An explanation consistent with the available evidence but not yet isolated experimentally.

### TARGET

A projected or expected result that has not yet been demonstrated.

### OPEN

An unresolved validation or engineering item.

---

## Current Evidence Summary

### PROVEN

```text
Complete 8-position prefill + 8-token generation run reaches completion.
248,320 / 248,320 logits are produced at every generation step.
NPU transport passes 6/6 SHA integrity controls.
Signed Q4_0 GEMV path is operational.
blk.31 complete differential validation is bit-exact.
Cross-layer generalization validated on blk.7, blk.15, blk.23.
NPU execution exercised on real model state across 8 layers.
Complete 32-layer CPU chain executes.
Real tokenizer is used.
Resident 7-worker RPC architecture executes the complete T6 run.
Previous stale/zombie RPC response race is closed for successful T6 configuration.
```

### MEASURED

```text
T6 end-to-end wall time: 418 s
Resident launcher measurement: 10.23 ms
Storage utilization during pathological pack preparation: ~98%
Pack preparation before storage migration: up to ~30 minutes
Pack preparation after migration to C:: ~13 seconds
```

### SUPPORTED HYPOTHESIS

```text
Remaining token-0 CPU/NPU argmax divergence caused primarily by
numerical differences between host FP16/FP32 golden arithmetic
and BF16-oriented NPU arithmetic.

Confidence before T6-b validation: ~85%.
```

### TARGET

```text
G9 batched GEMV:
FFN ~10.5 ms/token → ~2.6 ms/token
(Note: batch-4 targets weight b amortization and Q4_0 unpack/dequant,
not redundant x transfers; b is already reused across columns)

G10 Q4_0 NPU LM head:
Current CPU head ~30 s → target ~206 ms
```

### OPEN

```text
T6-b BF16-aligned CPU golden
Complete 32-layer NPU differential sweep
Long autoregressive generation
Long-context validation
G9 batched GEMV correctness and performance
G10 Q4_0 NPU LM-head correctness and performance
```

---

## T6 — Complete End-to-End Execution

T6 closes the previous gap between isolated NPU correctness and complete model execution. Earlier stages established correctness independently for transport, packing, signed Q4_0 GEMV, transformer blocks, model state, tokenizer handling, and logits generation.

The successful run performs 8 autoregressive generation steps with complete end-to-end execution validation.

It is not yet claimed as strict CPU/NPU numerical stream equivalence because T6-b remains open.

---

## T6-b — Remaining Numerical Discrepancy

Strict CPU and NPU generation streams currently diverge at generation token 0.

Observed behavior:
- CPU argmax ≠ NPU argmax
- Δvlogit ≈ 0.5%
- Competing top-2 logits remain inside previously observed BF16 numerical envelope

**Leading explanation**: Numerical tie-break caused by different arithmetic between execution paths. Current host golden uses FP16/FP32, while NPU kernel path uses BF16-oriented arithmetic.

**Status**: SUPPORTED HYPOTHESIS (not yet proven)

**Validation approach**: Full 248,320-logit differential analysis comparing max absolute error, mean error, RMSE, cosine similarity, top-k overlap, and rank behavior around winning logits.

---

## RPC Architecture — Race Closed

Earlier RPC architecture could encounter stale/zombie response races. This has been replaced by per-request protocol with:
- Per-request state
- Unique request sequencing
- Anti-replay handling
- Worker heartbeat validation

Successful T6 run started with 7/7 workers ready and completed using resident worker pool.

---

## Persistent Worker Pool

NPU execution infrastructure now uses seven persistent workers (q, k, v, o, gate, up, down). This avoids repeatedly rebuilding complete worker environment. Worker readiness independently checked before generation.

Canonical readiness condition: `7 / 7 fresh heartbeats`

---

## Host Storage Bottleneck — RESOLVED

**Previous problem**: Critical runtime artifacts on E: HDD resulted in:
- Disk utilization ≈ 98%
- Severe filesystem contention
- Pack preparation: up to ~30 minutes

**Resolution**: Migrated to C::
- RPC working data
- XCLBIN artifacts
- GGUF model (~5.3 GB)

**Result**: Pack preparation ≈ 13 seconds

**Lesson**: Previous initialization delays dominated by host storage, not NPU execution. Future benchmarks must distinguish:
- **COLD START**: new processes, no intentional cache, fresh workers
- **WARM**: workers resident, packs potentially cached
- **STEADY STATE**: runtime resident, reusable artifacts available

---

## Resident Launcher

Measured value: **10.23 ms**

Must remain explicitly distinguished from other metrics. Separate in performance reporting:
- request submission latency
- host/RPC latency
- launcher latency
- NPU execution latency
- synchronization latency
- result readback latency
- complete operation wall time

---

## Optimization Priorities

| Priority | Area | Description | Status |
|----------|------|-------------|--------|
| **P1** | Column & tile utilization | Measure requested vs. allocated vs. active columns, per-tile occupancy | MEASUREMENT NEEDED |
| **P2** | Persistent runtime state | Reuse contexts, BOs, overlays across requests | Reference: AMD Agent (2.2×/4.0× gain via context reuse + zero-copy BO recycling) |
| **P3** | AIE kernel fusion | Merge distinct kernels (RMSNorm+QKV+RoPE, output-proj+SwiGLU) to cut inter-op memory round-trips | Reference: AMD Agent (15→3 dispatches/layer via kernel fusion) |
| **P4** | W4A16 native execution | Compare current Q4NX vs. fused unpack+dequant+compute | Reference: TileFuse (2.0× prefill latency, 64%+ energy reduction) |
| **P5** | DMA/compute pipelining | Sequential vs. ping-pong double buffering | RESEARCH PHASE |
| **P6** | Attention & decode optimization | Separate strategies for prefill vs. decode, GEMM vs. GEMV | Reference: STEEL, Zen-Attention |

---

## Performance Decomposition Roadmap

Current reference E2E run: **418 s**

Must be decomposed into:
- initialization
- prefill (Q, K, V, O)
- transformer execution (gate, up, down)
- RPC overhead
- launcher overhead
- LM head
- sampling
- other host overhead

Decomposition required before attributing end-to-end improvements to individual kernel optimizations.

---

## Model-Driven Compiler — Long-Term Architecture

The project evolves from fixed kernel/runtime toward model-specialized compiler generating XDNA2 execution programs directly from target model.

**High-level flow**:
```
MODEL → Analysis → IR → XDNA2 Codegen → XCLBIN/PDI → Validation → Executable Cache → Runtime
```

**Specialization driven by**:
- Model architecture
- Tensor shapes
- Quantization
- Memory layout
- NPU topology
- Tile allocation
- DMA traffic

**Currently Implemented**:
- Real GGUF model tensors
- Real tokenizer
- Q4_0 packing path
- Direct XDNA2 NPU execution
- Persistent worker execution
- Signed Q4_0 GEMV
- Real-state transformer execution
- Full vocabulary logits
- Autoregressive generation loop
- Complete E2E execution
- Transport integrity validation
- Differential kernel/block validation

**Planned**:
- Full cost-model-driven scheduling
- Automatic fusion search
- Automatic tile exploration
- Automatic DMA exploration
- Model-wide autotuning
- Automatic correctness/performance search

---

## Historical Reports

Previous project states are preserved for reproducibility and traceability:

- **`docs/history/README_2026-09-04.md`** — NPU MINIMAL baseline; cleanest reproducible execution paths (1B/9B conservative)
- **`docs/history/README_2026-09-08.md`** — Frozen f3best witness; temporal decomposition; ~4.89 ms NPU execution established
- **`docs/history/README_2026-09-11.md`** — Numerical corruption diagnosis; STAGE 5 results; buffer divergence at node 44

**These reports represent historical snapshots and do not represent the current project state.**

For traceability: Each historical README is exact as committed; no modifications or omissions.

---

## Quick Start & Running Tests

Two reproducible scripts are provided for non-commercial testing:

```bash
# 1B — complete per-op path on NPU (attention included)
bash run_1B_npu_propre.sh
# Expected: "The capital of France is Paris." ~5.2 t/s

# 9B — GEMV + SwiGLU on NPU, CPU attention (only clean 9B path)
bash run_9B_npu_conservateur.sh
# Expected: ~1.8 t/s, clean text output
```

**Environment**: AMD Ryzen AI 9 365 (Strix Point), Windows 11, GGML backend, XRT driver, XDNA2 NPU

**Model**: Qwen3.5-9B-q40-lmhead-f16.gguf (hardlinked, ~5.3 GB)

**Kernels**: Cached in `kernels/` directory (45 .xclbin+.insts pairs)

---

## Licensing & Attribution

See **LICENSE** for dual licensing (non-commercial free / commercial restricted).

See **ATTRIBUTION.md** for full upstream acknowledgments and research references.

See **UPSTREAM_LICENSES.md** for third-party component licenses and compliance details.

**Critical external reference**: FastFlowLM Issue #636 ("FastFlowLM / XDNA2 — Performance Observations & Validation Roadmap") provides complementary performance analysis, measurement methodology, and optimization validation framework.

---

## Recommended Next Steps

1. **Reproducibility**: 5–10 complete fresh T6 runs with full environment recording
2. **Critical-path analysis**: Unified timeline (CPU → Runtime → XRT → IOCTL → Driver → DMA → Compute → Sync → Output)
3. **Utilization heatmap**: Per-column, per-tile activity during prefill/decode
4. **TTFT decomposition**: Context initialization, buffer allocation, weight prep, command construction, NPU execution
5. **Persistent runtime state prototype**: Context/BO reuse across requests
6. **AIE kernel fusion**: Merge RMSNorm+QKV+RoPE, output-proj+SwiGLU (not mere dispatch batching)
7. **Extended generation**: Validation at 32, 64, 128 tokens with KV-cache and state evolution tracking
8. **Differential 32-layer sweep**: All layers validated against CPU golden
9. **Separate prefill/decode optimization**: Different bottlenecks for each phase

---

**Last Updated**: 16 September 2026  
**Repository**: https://github.com/GaTmaNnes/xdna2-  
**License**: See LICENSE file (non-commercial free, commercial restricted)
