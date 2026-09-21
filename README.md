

# XDNA2 — Adaptive Compilation & Runtime Research on Windows

Experimental research project for **AMD XDNA2 / AIE2P NPUs on Windows**.

The project explores the complete path from generated AIE programs to real NPU execution:

```text
Workload / Model
       ↓
Execution planning
       ↓
Kernel generation
       ↓
IRON / MLIR-AIE
       ↓
Peano / AIE2P
       ↓
XCLBIN + instruction stream
       ↓
Windows runtime
       ↓
XDNA2 hardware
       ↓
Correctness validation
       ↓
Hardware measurements
       ↓
Adaptive planning
```

The current work has moved beyond basic NPU bring-up.

The stack can generate execution plans, compile them into real XDNA2 programs, execute them on hardware, verify their outputs, measure their performance, and use previous hardware observations to guide future execution choices.

---

## Current Status

| Area                                   | Status          |
| -------------------------------------- | --------------- |
| Windows XDNA2 execution                | ✅               |
| AIE2P kernel compilation               | ✅               |
| IRON / MLIR-AIE generation             | ✅               |
| Peano microkernels                     | ✅               |
| XCLBIN generation                      | ✅               |
| Real hardware correctness gates        | ✅               |
| Multi-column execution                 | ✅               |
| Tile exploration                       | ✅               |
| ExecutionPlan search                   | ✅               |
| Hardware-backed cost model             | ✅               |
| Prospective unseen-workload validation | ✅               |
| Adaptive planner                       | ✅               |
| Large GEMM multi-TOPS execution        | ✅               |
| Predictive kernel selection            | 🔬 Experimental |
| Global multi-shape cost model          | 🔬 Research     |

---

# Latest Hardware Results

Platform:

* AMD Ryzen AI 9 365
* XDNA2 / AIE2P
* Windows 11
* native AIE2P kernels
* real XCLBIN execution

## 2048³ INT8 GEMM

All measurements below passed the hardware correctness check.

| Tile     | Columns | Median kernel time |     Throughput |
| -------- | ------: | -----------------: | -------------: |
| 32×32×32 |       8 |          10,209 µs | **1.683 TOPS** |
| 64×32×64 |       8 |           3,502 µs | **4.905 TOPS** |
| 64×64×64 |       8 |           2,859 µs | **6.009 TOPS** |

The measured tile transition is substantial:

```text
32×32×32
1.683 TOPS
    │
    │ ×2.91
    ▼
64×32×64
4.905 TOPS
    │
    │ ×1.23
    ▼
64×64×64
6.009 TOPS
```

Overall:

```text
32³ → 64³ = 3.57×
```

This demonstrates that the earlier low throughput observed on small workloads was **not a ~0.1-TOPS limitation of the Windows execution path**.

Workload geometry, tiling and data movement strongly determine how much of the XDNA2 array can actually be exploited.

---

# Comparison with a Public Linux Reference

For the directly comparable 2048³ INT8 configurations:

| Configuration           | This project — Windows | Public Linux reference |
| ----------------------- | ---------------------: | ---------------------: |
| tile32                  |         **1.683 TOPS** |              1.97 TOPS |
| tile64                  |         **6.009 TOPS** |              6.65 TOPS |
| tile32 → tile64 scaling |              **3.57×** |                  3.38× |

The Windows path reaches:

```text
tile32 : 85.4% of the public Linux result
tile64 : 90.4% of the public Linux result
```

More importantly, both environments exhibit a similar large performance transition when moving from the smaller to the larger tile configuration.

These numbers should not be interpreted as a strict cross-platform benchmark: the software stacks, compilation paths and runtime environments differ.

They provide an external reference point for the observed XDNA2 scaling behavior.

---

# Adaptive Execution Planning

A major part of the project is no longer a fixed collection of kernels.

Execution configurations are represented as plans containing hardware-relevant choices such as:

```text
workload shape
tile geometry
column allocation
parallelism
data movement
working set
execution strategy
```

Candidate plans pass through feasibility checks before hardware execution.

A cost model then ranks viable candidates and allows the system to concentrate compilation and hardware measurements on promising regions of the search space.

```text
WorkloadSpec
     ↓
Feasibility
     ↓
Candidate ExecutionPlans
     ↓
Physical cost model
     ↓
Top-K
     ↓
Compile
     ↓
XDNA2
     ↓
Correctness
     ↓
Measurement
```

Simpler predictors were deliberately retained as reproducible baselines.

The current physical model was introduced only after controlled hardware experiments exposed limitations in those simpler approaches.

---

# Prospective Prediction

The planner has been tested using a **freeze-before-measurement protocol**.

For a previously unmeasured workload:

1. the model and training dataset were frozen;
2. all candidate predictions were recorded;
3. the predicted ranking was committed before compilation;
4. all candidates were then compiled;
5. every candidate was executed on the real NPU;
6. correctness was required;
7. an exhaustive hardware oracle was constructed.

On the prospective K=1344 experiment:

| Metric          |    Result |
| --------------- | --------: |
| Candidate plans |         6 |
| Predicted Top-1 |  `c4_t64` |
| Hardware oracle |  `c4_t64` |
| Top-1 regret    |  **0.0%** |
| Oracle rank     | **1 / 6** |
| Spearman        |   **1.0** |
| MAPE            |  **3.8%** |

The predicted best execution plan was therefore the measured hardware optimum.

This is a prospective result: the workload had not been measured when the ranking was produced.

---

# Predictive Kernel Research

The project also contains experimental work on **predictive kernel and execution-plan selection**.

The objective is to use accumulated hardware behavior to identify promising kernel configurations before paying the full cost of compilation and exhaustive benchmarking.

This work is intentionally experimental.

The current implementation is not presented as a universal XDNA2 performance predictor. Some models are reliable primarily as rankers inside their validated domain, while absolute latency prediction across very different workload scales remains an active research area.

An interesting result is that a frozen planner trained on substantially smaller workloads ranked the high-performance `tile64 / 8-column` region first before the 2048³ configurations were compiled.

The resulting hardware execution reached:

```text
6.009 TOPS
```

This motivates continued work on physically informed predictive kernel selection, while broader generalization remains under validation.

---

# Why Physical Features Matter

The hardware experiments show that nominal operation count alone is insufficient to predict XDNA2 performance.

For the same 2048³ INT8 GEMM and the same 8-column configuration:

```text
tile32³  → 1.683 TOPS
tile64³  → 6.009 TOPS
```

A change in execution geometry alone corresponds to a **3.57× throughput difference**.

Current planning research therefore considers quantities related to:

* work distribution;
* number of active cores;
* work per core;
* tile geometry;
* number of tiles;
* data movement;
* working set;
* transfer count;
* reuse and parallelism.

The goal is not to encode rules such as "tile64 is faster", but to learn and model the physical consequences of execution-plan choices.

---

# Hardware-in-the-Loop Methodology

Performance claims in this repository follow a hardware validation pipeline.

```text
Generate
   ↓
Compile
   ↓
XCLBIN
   ↓
Execute on XDNA2
   ↓
Correctness gate
   ↓
Benchmark
   ↓
Record observation
```

A candidate that fails correctness is not considered a performance result.

For prediction experiments:

```text
TRAIN
  ↓
freeze model
  ↓
freeze predictions
  ↓
compile
  ↓
hardware execution
  ↓
correctness
  ↓
exhaustive oracle
  ↓
regret / ranking evaluation
```

This separation is important because it prevents hardware results from leaking back into a supposedly prospective prediction.

---

# Evidence Levels

Results are classified using explicit evidence levels.

### PROVEN

Correctness or behavior established using direct controls such as differential validation, hashes, bit-exact comparisons, causal experiments or complete hardware execution.

### MEASURED

A quantity directly observed on hardware.

### SUPPORTED HYPOTHESIS

An interpretation consistent with measurements but not yet experimentally isolated.

### TARGET

A performance or engineering objective that has not yet been demonstrated.

### OPEN

An unresolved research or engineering question.

---

# End-to-End Model Execution

The project originally started from direct model execution on XDNA2.

That work established:

* GGUF loading;
* standard Q4_0 handling;
* XDNA2-specific packing;
* real NPU GEMV;
* transformer execution;
* state evolution;
* LM-head execution;
* autoregressive generation;
* CPU-Q4 / NPU-Q4 stream validation;
* long golden sequences;
* persistent runtime infrastructure.

These experiments provided the correctness and runtime foundation used by the newer adaptive compiler/planner work.

The project has since expanded from:

```text
"Can this model execute correctly on XDNA2?"
```

toward:

```text
"Which execution program should be generated for this workload?"
```

---

# Compiler / Runtime Direction

The longer-term architecture is:

```text
MODEL / WORKLOAD
       ↓
Analysis
       ↓
Operation extraction
       ↓
ExecutionPlan search
       ↓
Predictive planning
       ↓
Kernel generation
       ↓
IRON / MLIR-AIE
       ↓
Peano
       ↓
XCLBIN / instruction artifacts
       ↓
Windows runtime
       ↓
XDNA2
       ↓
Correctness + performance
       ↓
Hardware knowledge
```

The intent is to progressively replace manually selected configurations with hardware-informed execution planning.

This is still a research system, not a production compiler.

---

# Current Research Directions

Current work focuses on extending the validated search domain beyond the initial GEMM experiments.

Important directions include:

* larger M/N/K domains;
* additional tile geometries;
* 1 / 2 / 4 / 8-column execution;
* DMA scheduling;
* buffering strategies;
* data placement;
* memory reuse;
* multi-kernel execution;
* kernel fusion;
* persistent runtime state;
* physically constrained cost models;
* ranking versus absolute-latency calibration;
* automatic challenger/oracle evaluation.

The next generation of the cost model is intended to preserve the demonstrated ranking behavior while improving latency calibration across very different workload scales.

---

# Reproducibility

Important experiments preserve:

* workload description;
* ExecutionPlan;
* generated MLIR;
* compiled objects;
* XCLBIN;
* instruction streams;
* compiler/runtime provenance;
* hashes;
* frozen predictions;
* hardware measurements;
* correctness results;
* oracle results.

This allows planner decisions to be traced back to the actual hardware program that was executed.

Historical reports are retained rather than rewritten so previous project states remain reproducible.

---

# Scope and Limitations

Current results should be interpreted within their demonstrated scope.

The project does **not** currently claim:

* universal optimality across XDNA2 workloads;
* peak theoretical XDNA2 throughput;
* a globally calibrated performance model;
* superiority over every Linux or AMD runtime;
* a production-ready autonomous compiler.

The demonstrated results are narrower:

* real generated AIE2P programs execute correctly on XDNA2 under Windows;
* execution-plan choices can change measured performance by several times;
* the Windows path has reached **6.009 TOPS** on 2048³ INT8 GEMM;
* directly comparable public Linux results provide an external performance reference;
* hardware-derived physical features can successfully rank execution plans;
* a frozen cost model selected the exact hardware-optimal plan on a prospective unseen workload;
* predictive kernel-selection experiments can identify promising regions of a larger execution space before exhaustive hardware exploration.

---

# Historical Reports

Earlier project states are preserved under `docs/history/`.

They document the progression from minimal NPU execution and numerical validation through complete model execution and toward adaptive compilation.

Historical reports are snapshots and should not be interpreted as the current performance state.

---

# Licensing & Attribution

See:

* `LICENSE`
* `ATTRIBUTION.md`
* `UPSTREAM_LICENSES.md`

for licensing terms, upstream projects and research references.

This project builds on and studies components and concepts from the broader AMD/Xilinx AIE ecosystem, including MLIR-AIE, IRON, Peano, XRT and related open research.

---

## Current Research Milestone

The current milestone can be summarized as:

```text
correct execution
        ↓
generated kernels
        ↓
hardware search
        ↓
physical cost model
        ↓
prospective prediction
        ↓
adaptive execution planning
        ↓
multi-TOPS XDNA2 execution
```

The next objective is broader generalization: predicting both **which execution plan will win** and **its approximate hardware cost** across substantially different workload regimes.

---

**Last updated:** 21 September 2026
**Platform:** AMD Ryzen AI 9 365 / XDNA2 / AIE2P / Windows 11

















# XDNA2 / Qwen3.5-9B — Windows NPU Runtime

> Direct GGUF → XDNA2 execution is correctness-validated against the quantization-matched CPU Q4_0 reference path.
>
> The project has moved beyond basic NPU execution. Current work focuses on **end-to-end decode performance, data movement, kernel geometry, fusion, runtime orchestration, scheduling, and model-specialized XDNA2 code generation**.

**Current status: 20 September 2026**

---

## Current Result

The project now runs a real Qwen3.5-9B inference path on the AMD XDNA2 NPU under Windows, including:

* real GGUF tensors;
* real tokenizer;
* Q4_0 weight packing;
* XDNA2 GEMV execution;
* complete 32-layer model execution;
* recurrent/state evolution;
* full 248,320-logit LM head;
* autoregressive generation;
* quantization-matched CPU/NPU stream validation;
* long 64-token correctness validation;
* persistent in-process execution;
* batched GEMV;
* fused SwiGLU path;
* per-operation performance attribution;
* host/runtime profiling;
* XDNA2 kernel generation and compilation experiments.

The main engineering question is no longer:

> **Can the model execute on XDNA2?**

It is now:

> **How close can a generated, model-specialized XDNA2 execution plan get to the useful memory and execution limits of the hardware?**

---

# Current Validation Matrix

| Area                                 | Status                                   |
| ------------------------------------ | ---------------------------------------- |
| GGUF loading                         | **PASS**                                 |
| Standard Q4_0                        | **PASS**                                 |
| Standard Q6_K reference              | **PASS**                                 |
| XDNA2 packing                        | **PASS**                                 |
| Real NPU GEMV                        | **PASS**                                 |
| Complete Qwen3.5 model chain         | **PASS**                                 |
| DeltaNet / recurrent state evolution | **PASS**                                 |
| Full 248,320-logit LM head           | **PASS**                                 |
| CPU-Q4 == NPU-Q4 generation stream   | **PASS — 8/8**                           |
| 64-token Q4_0 golden                 | **PASS — 64/64**                         |
| Batched GEMV                         | **PASS**                                 |
| Batched GEMV acceleration            | **MEASURED — ~42.1 → 17–18 ms**          |
| Multi-XCLBIN execution               | **PASS with context lifetime contract**  |
| Dual / fused SwiGLU                  | **PASS with corrected packing layout**   |
| 64-token production baseline         | **VALIDATED**                            |
| F-796 host-I/O optimization          | **MEASURED — 24-token candidate result** |
| Physical DDR ceiling                 | **OPEN**                                 |
| Automatic model-wide scheduling      | **IN DEVELOPMENT**                       |
| Model-specialized kernel generation  | **IN DEVELOPMENT**                       |

---

# Performance Progress

The runtime has moved from an early approximately **1.86 s/token** execution path to a current candidate around **0.667 s/token**.

The important milestones are:

```text
~1860 ms/token   early complete runtime
      ↓
 ~925 ms/token   LM-head path improvement
      ↓
 ~893 ms/token   output-projection improvement
      ↓
845–872 ms/token clean 64-token baseline campaigns
      ↓
 ~804 ms/token   host/environment path cleaned
      ↓
 ~773 ms/token   in-process runtime / IPC removed
      ↓
 ~730 ms/token   fused SwiGLU path
      ↓
 ~667 ms/token   F-796 persistent logging handles
```

Overall observed improvement from the early runtime:

```text
~1860 → ~667 ms/token
≈ 2.79×
```

### Important measurement note

The **~667 ms/token** F-796 result was measured on a shorter 24-token campaign and is therefore treated as a **candidate record**, not yet as the new long-run 64-token baseline.

Performance decisions are made from longer controlled runs whenever possible.

---

# Correctness

Performance results are not promoted unless the relevant correctness contract passes.

The project has progressively closed several correctness layers:

```text
GGUF
  ↓
Q4_0 decoding
  ↓
packing
  ↓
buffer layout
  ↓
XDNA2 kernel
  ↓
layer output
  ↓
model state
  ↓
LM head
  ↓
token stream
```

A key validation established that the quantization-matched CPU Q4_0 and NPU Q4_0 paths produce the same generation stream:

```text
CPU-Q4 == NPU-Q4
8 / 8 tokens
PASS
```

A longer 64-token golden run is also validated.

Earlier CPU/NPU divergence was traced to comparison of different numerical paths rather than a failure of the tested NPU Q4_0 execution path.

---

# Batched GEMV

Batching multiple output rows inside the kernel was one of the first major architectural improvements.

Representative result:

```text
single / earlier path : ~42.1 ms
batched B=4           : ~17–18 ms

speedup ≈ 2.46×
```

Correctness:

```text
49,145 / 49,152 BF16 outputs matched
max ULP ≤ 2
```

The important lesson is not simply that batching is faster.

Batching changes how weight traffic, unpacking, dequantization and compute are amortized across outputs.

This is now treated as an execution-plan decision rather than a fixed kernel constant.

---

# Data Movement Is the Dominant Device Problem

The current decode path is globally dominated by **weight/data movement**, not raw MAC throughput.

A recent causal decomposition attributes approximately:

```text
DEVICE ≈ 590 ms/token
```

with roughly:

| Family             |    Time |   Effective BW |
| ------------------ | ------: | -------------: |
| gate/up fused path | ~221 ms |     ~8.29 GB/s |
| down               | ~127 ms |     ~7.26 GB/s |
| q                  |  ~98 ms |     ~6.30 GB/s |
| o                  |  ~65 ms |     ~4.77 GB/s |
| head               |  ~58 ms | **~9.80 GB/s** |
| k+v                |  ~20 ms |      ~7.9 GB/s |

The exact values depend on the tested runtime configuration.

The key result is the large spread in effective bandwidth between operation families.

Therefore:

```text
effective bandwidth != one universal machine constant
```

It depends on the execution plan:

```text
BW_eff = f(
    operation,
    layout,
    packing,
    tile geometry,
    DMA organization,
    batching,
    runtime boundaries,
    synchronization
)
```

---

# 9.8 GB/s Is Not Claimed as the Physical DDR Limit

The LM-head path currently reaches approximately:

```text
572 MB / 58.4 ms
≈ 9.8 GB/s
```

This is the highest effective bandwidth observed so far in the current execution stack.

It is **not** claimed to be the physical DDR ceiling.

Earlier `BO.sync()` microbenchmarks produced values above 100 GB/s, but those measurements primarily characterized host-side coherency/cache behavior on the tested unified-memory path and were therefore rejected as measurements of NPU DDR bandwidth.

The physical DDR limit remains:

```text
OPEN
```

Only timings produced by real NPU kernels are used for current effective-bandwidth claims.

---

# Host Runtime Matters Too

The device is not the whole token wall-clock.

A token crosses a large number of runtime boundaries.

Current measurements show roughly:

```text
~175 dispatches / token
```

Host profiling identified several significant costs.

Before F-796, representative host-chain costs included:

```text
logging / file open / exists    ~80 ms/token
decode_q40_rows                 ~21 ms/token
attention                       ~16 ms/token
NumPy/layout conversions        ~21 ms/token
```

The logging overhead was particularly important because diagnostic files were repeatedly opened and closed around dispatches.

F-796 changed these paths to persistent file handles while preserving the live metrics stream.

This produced the current short-run candidate around:

```text
~667 ms/token
~1.45 token/s
```

The result demonstrates that runtime engineering remains relevant even when the NPU kernels themselves are memory-bound.

---

# Runtime Cost Model

The current mental model is:

```text
T_token
    =
      T_device
    + T_host
    + T_boundaries
```

More explicitly:

```text
T_token(P)
    =
      T_memory(P)
    + T_compute(P)
    + N_boundaries(P) × H_boundary(P)
    + T_host(P)
```

where `P` is the complete execution plan.

This is an important change from the original project model.

Optimizing a kernel in isolation is not sufficient if the surrounding execution plan introduces excessive:

* memory traffic;
* packing;
* synchronization;
* dispatch boundaries;
* conversions;
* runtime state reconstruction.

---

# Fused SwiGLU

The gate/up path has also been tested as a fused execution problem.

A critical finding was that packing/layout is part of the kernel contract.

A minimal single-column test showed the internal GEMV/SwiGLU computation was correct, while scaling exposed a packing-layout issue.

After correcting the banded packing layout:

```text
NaN                  0
deterministic         yes
tanh-aware error      ~0.64%
```

The fused path subsequently contributed approximately:

```text
~65 ms/token
```

of improvement in the tested runtime.

The broader lesson is:

> **A kernel is not defined only by its arithmetic. Its input packing and memory layout are part of its executable contract.**

---

# XRT Context Lifetime Is Part of Correctness

Multi-XCLBIN experiments exposed an important runtime rule.

An apparent multi-XCLBIN corruption problem was eventually isolated to the lifetime of XRT/Python objects.

Keeping the relevant:

```text
hardware context
kernel handle
buffer objects
```

alive restores deterministic execution.

Therefore context lifetime is now treated as a correctness requirement.

It is not considered an optional runtime implementation detail.

---

# Column Splitting / NPU Concurrency

Splitting gate/up across separate column groups was tested to determine whether independent NPU work could overlap.

Representative measurements:

```text
8-column kernel       ~7.9 ms
4-column kernel      ~12.4 ms

gate4 || up4
makespan             ~11.9 ms
overlap               ~0.5 ms
```

The tested configuration therefore showed essentially no useful NPU↔NPU overlap.

Verdict:

```text
CHANNEL-LIMITED
```

This killed the assumption that simply dividing the array into independent column groups would provide near-linear concurrent execution.

CPU↔NPU pipelining remains a separate optimization problem.

---

# Compiler / ISA Findings

Generated source structure is not sufficient to predict the final hardware schedule.

The project now audits the compiler chain through:

```text
source
  ↓
LLVM IR
  ↓
Peano
  ↓
MIR
  ↓
post-RA scheduling
  ↓
AIE2P ISA
```

Several compiler-control hypotheses were tested.

### Register rewrite modes

Different register-rewrite modes produced bit-identical objects/binaries in the tested case.

They were therefore eliminated as a useful optimization knob for that problem.

### Post-pipeliner

Several post-pipeliner controls were tested.

A schedule could be made correctness-safe but was slower than the B2 reference.

Result:

```text
correct
but
no performance gain
```

This reinforced an important rule:

> Compiler transformations must be evaluated from final ISA + correctness + real NPU timing, not from source-level intent alone.

---

# What We Fa




Direct GGUF → XDNA2 execution is now correctness-validated against the CPU Q4_0 reference path. Current work has moved from basic execution correctness to adaptive kernel generation, scheduling, memory-path characterization and performance.



F-773 — Unified-memory / BO synchronization characterization

Two BO.sync() bandwidth microbenchmarks were invalidated as NPU DDR measurements. On the tested host_only unified-memory path, their ~100+ GB/s results primarily characterize host-side coherency/cache behavior and must not be interpreted as NPU DDR bandwidth.

Consequently, DDR_CAP remains OPEN. Effective NPU bandwidth is derived from real kernel execution. The current measured kernel-effective bandwidth reaches 9.08 GB/s, while the observed compute activity is ~66%; 9.08 GB/s is therefore not claimed as the physical DDR ceiling.

The next discriminating experiment is a generated column-split gate/up execution: sequential gate@8 + up@8 versus concurrent gate@4 || up@4.


GGUF loading                         PASS
Standard Q4_0                       PASS
XDNA2 packing                       PASS
Real NPU GEMV                       PASS
Full Qwen3.5 execution chain        PASS
DeltaNet/state evolution            PASS
LM-head causality                    PASS
CPU-Q4 == NPU-Q4 stream             PASS — 8/8
64-token Q4_0 golden                PASS
Batched GEMV correctness            PASS — 99.99%, maxUlp≤2
Batched GEMV acceleration           MEASURED — ~42.1 → 17–18 ms
Host overhead                       MEASURED — ~12.6 ms/token
Kernel-effective BW                 MEASURED — up to 9.08 GB/s
Compute activity                    MEASURED — ~66%
Physical DDR_CAP                    OPEN
BO.sync DDR benchmark               INVALIDATED — F-773
Adaptive kernel generator           IN DEVELOPMENT
F-774 clean 64-token KPI            VALIDATION IN PROGRESS


README actuel du dépôt xdna2-







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
