

# XDNA2 / Qwen3.5-9B — Current Status — 16 September 2026

> **Major milestone:** the direct GGUF → XDNA2 NPU path has now completed a full end-to-end autoregressive inference run.
>
> This status supersedes the 11 September 2026 status where Qwen3.5-9B end-to-end generation was still marked **NOT CLOSED**.

---

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

This run establishes that the current stack can execute a complete prefill + autoregressive generation sequence instead of stopping at isolated kernels, synthetic tensors, individual transformer blocks, or partial model execution.

---

# Current Validation Matrix

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

# Evidence Levels

All current and future results should use the following evidence classification.

## PROVEN

A result reproduced with direct correctness controls such as hashes, differential comparisons, bit-exact comparisons, or complete execution.

## MEASURED

A directly observed quantity, but not by itself a correctness proof.

## SUPPORTED HYPOTHESIS

An explanation consistent with the available evidence but not yet isolated experimentally.

## TARGET

A projected or expected result that has not yet been demonstrated.

## OPEN

An unresolved validation or engineering item.

---

# Current Evidence Summary

## PROVEN

```text
Complete 8-position prefill + 8-token generation run reaches completion.

248,320 / 248,320 logits are produced at every generation step.

NPU transport passes 6/6 SHA integrity controls.

Signed Q4_0 GEMV path is operational.

blk.31 complete differential validation is bit-exact.

Cross-layer generalization has been validated on blk.7, blk.15 and blk.23.

NPU execution has been exercised on real model state across 8 layers.

The complete 32-layer CPU chain executes.

The real tokenizer is used.

The resident seven-worker RPC architecture executes the complete T6 run.

The previous stale/zombie RPC response race is closed for the successful T6 configuration.
```

## MEASURED

```text
T6 end-to-end wall time: 418 s

Resident launcher measurement: 10.23 ms

Previous E: storage utilization during pathological pack preparation:
approximately 98%

Pack preparation before storage migration:
up to approximately 30 minutes

Pack preparation after migration to C::
approximately 13 seconds
```

## SUPPORTED HYPOTHESIS

```text
The remaining token-0 CPU/NPU argmax divergence is caused primarily
by numerical differences between the host FP16/FP32 golden arithmetic
and the BF16-oriented NPU arithmetic.

Confidence before T6-b validation: approximately 85%.
```

## TARGET

```text
G9 batched GEMV:
FFN approximately 10.5 ms/token → approximately 2.6 ms/token

G10 Q4_0 NPU LM head:
current CPU head approximately 30 s → target approximately 206 ms
```

## OPEN

```text
T6-b BF16-aligned CPU golden

Complete 32-layer NPU differential sweep

Long autoregressive generation

Long-context validation

G9 batched GEMV correctness and performance

G10 Q4_0 NPU LM-head correctness and performance
```

---

# T6 — Complete End-to-End Execution

T6 closes the previous gap between isolated NPU correctness and complete model execution.

Earlier stages established correctness independently for transport, packing, signed Q4_0 GEMV, transformer blocks, model state, tokenizer handling and logits generation.

T6 connects these components into one complete execution path.

The successful run performs:

```text
prompt
 ↓
tokenization
 ↓
8 prefill positions
 ↓
model state
 ↓
transformer execution
 ↓
NPU operations
 ↓
complete LM output
 ↓
248,320 logits
 ↓
token selection
 ↓
state update
 ↓
next generation step
```

for eight autoregressive generation steps.

The current result is therefore an **end-to-end execution validation**.

It is not yet claimed as strict CPU/NPU numerical stream equivalence because T6-b remains open.

---

# T6-b — Remaining Numerical Discrepancy

Strict CPU and NPU generation streams currently diverge at generation token 0.

Observed behavior:

```text
CPU argmax != NPU argmax

Δvlogit ≈ 0.5%

The competing top-2 logits remain inside the previously observed
BF16 numerical envelope.
```

The leading explanation is a numerical tie-break caused by different arithmetic between the two execution paths.

Current host golden arithmetic includes FP16/FP32 operations, while the NPU kernel path uses BF16-oriented arithmetic.

Two almost equal logits can therefore reverse ordering without requiring a kernel correctness failure.

Current status:

```text
SUPPORTED HYPOTHESIS
```

This explanation is **not yet considered proven**.

## T6-b validation

The next golden should reproduce the arithmetic boundaries used by the NPU path as closely as possible.

The comparison must not be limited to argmax.

For the complete 248,320-logit vector, record:

```text
max absolute error
mean absolute error
RMSE
cosine similarity
top-1 margin
top-k overlap
rank behavior around the winning logits
```

The experiment should also identify where BF16 conversion occurs:

```text
input
 ↓
normalization
 ↓
GEMV input
 ↓
accumulation
 ↓
output conversion
 ↓
residual
 ↓
next operation
```

The goal is to distinguish:

```text
kernel error
```

from:

```text
expected divergence caused by different floating-point arithmetic
```

without using argmax equality as the sole criterion.

---

# RPC Architecture — Race Closed

The earlier RPC architecture could encounter stale/zombie response races.

This has been replaced by a per-request protocol.

Current architecture:

```text
generator
   │
   ├── q worker
   ├── k worker
   ├── v worker
   ├── o worker
   ├── gate worker
   ├── up worker
   └── down worker
```

Seven workers remain resident.

Each request carries request-specific identity information.

The protocol now includes:

```text
per-request state
unique request sequencing
anti-replay handling
worker heartbeat validation
```

The successful T6 run started with:

```text
7 / 7 workers ready
```

and completed using the resident worker pool.

The previous shared-state race is therefore considered closed for the validated T6 configuration.

---

# Persistent Worker Pool

The NPU execution infrastructure now uses seven persistent workers:

```text
q
k
v
o
gate
up
down
```

This avoids repeatedly rebuilding the complete worker environment for every operation.

Worker readiness is independently checked before generation.

Canonical readiness condition:

```text
7 / 7 fresh heartbeats
```

The complete T6 run was executed with the resident worker architecture active.

---

# Host Storage Bottleneck

A major performance problem was found outside the NPU itself.

The previous configuration placed critical runtime artifacts on the `E:` HDD.

Observed behavior included:

```text
disk utilization ≈ 98%

severe filesystem contention

pack preparation taking up to approximately 30 minutes
```

The following were migrated to `C:`:

```text
RPC working data
XCLBIN artifacts
GGUF model (~5.3 GB)
```

After migration:

```text
pack preparation ≈ 13 s
```

This demonstrates that previous initialization delays could be dominated by host storage rather than NPU execution.

Future benchmarks must distinguish at least:

```text
COLD START
- new processes
- no intentionally retained pack cache
- workers freshly started

WARM
- workers resident
- packs potentially cached

STEADY STATE
- runtime resident
- reusable artifacts already available
```

Cold-start and steady-state numbers must not be compared as equivalent measurements.

---

# Resident Launcher

The resident launcher currently has a measured value of:

```text
10.23 ms
```

This metric must remain explicitly distinguished from historical measurements such as kernel-only NPU wait time.

The following quantities must remain separate in performance reporting:

```text
request submission latency
host/RPC latency
launcher latency
NPU execution latency
synchronization latency
result readback latency
complete operation wall time
```

No two values should be compared unless they measure the same interval.

---

# Autoregressive State Validation

The successful T6 run demonstrates repeated generation over updated model state.

Additional validation will explicitly track state evolution.

For each generation step, future validation should record hashes for:

```text
input hidden state
K-cache state
V-cache state
final hidden state
logits
```

Expected invariant:

```text
token N
   ↓
state update
   ↓
KV update
   ↓
token N+1 consumes updated state
```

The repeated final token IDs observed in the initial T6 run:

```text
126104
126104
126104
```

are not by themselves classified as an error.

They will be investigated through top-k logits, margins and state hashes during extended-generation validation.

---

# Full-Layer Validation

Current differential validation includes:

```text
blk.7
blk.15
blk.23
blk.31
```

with complete `blk.31` validation reaching bit-exact agreement.

This provides strong cross-layer evidence but does not yet constitute exhaustive validation of all 32 layers.

The next exhaustive control will record:

```text
L00 PASS/FAIL
L01 PASS/FAIL
L02 PASS/FAIL
...
L31 PASS/FAIL
```

for a canonical real model state.

Inputs and outputs should be hashed so that every layer can be associated with a reproducible witness.

---

# Transport Contract

SHA equality validates byte transport.

Future manifests should additionally record interpretation metadata.

For each transported tensor or packed artifact:

```text
SHA256
byte count
dtype
shape
strides
layout
endianness
quantization type
scale metadata
```

This distinguishes byte identity from tensor interpretation correctness.

---

# Pack Cache Validation

Persistent pack caching is now part of the resident execution architecture.

Cache correctness must be independently verified.

A cache entry must be uniquely determined by every property that can change the generated packed representation, including where applicable:

```text
weight identity
weight hash
shape
quantization
layout
kernel specialization
XCLBIN
instruction configuration
```

Required differential test:

```text
cache disabled → output A

cache enabled  → output B

A == B
```

Caching must never change numerical output.

---

# G10 — Q6_K CPU LM Head → Q4_0 NPU

The LM head is currently one of the largest remaining end-to-end bottlenecks.

Current path:

```text
final hidden state
 ↓
Q6_K output.weight
 ↓
CPU
 ↓
248,320 logits
```

Current CPU head cost is approximately:

```text
~30 s
```

The proposed G10 path is:

```text
final hidden state
 ↓
Q4_0 output.weight
 ↓
XDNA2 NPU
 ↓
248,320 logits
```

Target:

```text
~206 ms
```

Current confidence:

```text
~80%
```

This value is a **TARGET**, not a validated end-to-end measurement.

## Required G10 A/B/C validation

Backend and quantization changes must be isolated.

Three paths will be compared:

```text
A — Q6_K CPU

B — Q4_0 CPU

C — Q4_0 NPU
```

Interpretation:

```text
A ↔ B
= quantization effect

B ↔ C
= backend / arithmetic effect
```

For all three paths, compare:

```text
complete 248,320 logits
max absolute error
mean error
top-k overlap
argmax
wall time
```

Only after B ↔ C correctness is established should the NPU performance result be considered validated.

---

# G9 — Batched GEMV

The next major transformer-side optimization is batched GEMV.

Current FFN cost:

```text
~10.5 ms/token
```

Target:

```text
~2.6 ms/token
```

Current confidence:

```text
~75%
```

This value remains a **TARGET**.

Correctness must be established before throughput.

Required test:

```text
same weights
same input
same quantization

unbatched GEMV
        vs
batched GEMV
```

Compare:

```text
complete output tensor
max absolute error
mean absolute error
bit-exact equality where applicable
```

Only after the differential gate passes should performance measurements be accepted.

---

# Performance Decomposition

The current reference E2E run takes:

```text
418 s
```

This value should not be treated as a single NPU performance metric.

Future profiling will decompose it into:

```text
initialization
pack preparation
tokenization
prefill
Q
K
V
O
gate
up
down
RPC overhead
launcher overhead
NPU execution
CPU execution
LM head
logit readback
sampling
idle/wait
```

The objective is to obtain:

```text
418 s
│
├── initialization
├── prefill
├── transformer execution
├── RPC / launcher
├── LM head
├── sampling
└── other host overhead
```

This decomposition is required before attributing end-to-end improvements to an individual kernel optimization.

---

# Long-Generation Validation

The current T6 witness uses:

```text
8 prefill positions
8 generated tokens
```

This validates integration but does not yet stress long-context behavior.

Planned progression:

```text
8 tokens
 ↓
32 tokens
 ↓
64 tokens
 ↓
128 tokens
 ↓
512 tokens where practical
```

The extended tests will monitor:

```text
KV-cache correctness
position handling
RoPE behavior
memory growth
buffer reuse
worker stability
RPC stability
output drift
state hashes
latency evolution
```

Long-generation correctness is therefore currently classified as:

```text
OPEN
```

---

# Reproducibility Campaign

A single successful complete run establishes existence of the execution path.

The next reproducibility gate requires repeated fresh executions.

Target:

```text
5–10 complete runs
```

For every run record:

```text
source commit
GGUF SHA256
XCLBIN SHA256
instruction SHA256
tokenizer SHA256
worker configuration
storage location
cache state
prompt
generated IDs
logit hashes
latencies
worker failures
memory behavior
```

A reproducible run should produce consistent correctness evidence independently of process lifetime and stale filesystem state.

---

# T6 Reproduction Contract

## Environment

Record:

```text
Windows version/build
XRT version
XDNA driver version
Python version
NPU/device identification
system RAM
model storage location
runtime storage location
```

## Required artifacts

Record exact SHA256 values for:

```text
GGUF
tokenizer
XCLBIN artifacts
instruction artifacts
generated packs where applicable
```

Record the exact source commit used for the run.

## Startup

Start the seven resident workers:

```text
q
k
v
o
gate
up
down
```

Verify:

```text
7 / 7 workers ready
```

No generation run should be considered canonical if worker readiness is incomplete.

## Canonical prompt

```text
The capital of France is
```

## Canonical workload

```text
prefill positions: 8
generated tokens: 8
```

## Expected structural result

```text
all required NPU requests complete successfully

248,320 / 248,320 logits are produced at every generation step

no stale RPC response is accepted

no replayed request result is accepted

the generation loop reaches completion
```

## Initial reference token IDs

The successful 16 September 2026 run produced:

```text
[26705, 92795, 97950, 99952, 72787, 126104, 126104, 126104]
```

Until T6-b closes the CPU/NPU numerical discrepancy, these IDs should be treated as a reference witness for this specific arithmetic/runtime configuration rather than a universal semantic golden.

---

# Model-Driven Compiler — Implementation Status

The long-term architecture remains:

```text
model
 ↓
inspection
 ↓
internal representation
 ↓
kernel selection
 ↓
code generation
 ↓
packing
 ↓
validation
 ↓
execution
 ↓
profiling
 ↓
autotuning
```

The current implementation status should be distinguished from the long-term architecture.

## IMPLEMENTED / DEMONSTRATED

```text
real GGUF model tensors

real tokenizer

Q4_0 packing path

direct XDNA2 NPU execution

persistent worker execution

signed Q4_0 GEMV

real-state transformer execution

full vocabulary logits

autoregressive generation loop

complete E2E execution

transport integrity validation

differential kernel/block validation
```

## PARTIAL

```text
automatic kernel specialization

validation automation

persistent executable/pack caching

heterogeneous CPU/NPU placement

complete operation-level profiling

model-wide automatic placement
```

## PLANNED

```text
full cost-model-driven scheduling

automatic fusion search

automatic tile exploration

automatic DMA exploration

full model-wide autotuning

automatic correctness/performance search
```

---

# Current Roadmap

The immediate validation and optimization sequence is:

```text
T6
│
├── COMPLETE — first E2E execution
│
▼
T6-R
│
├── repeated fresh-run reproducibility
│
▼
T6-b
│
├── BF16-aligned CPU golden
│
├── complete 248,320-logit differential analysis
│
▼
T6-L
│
├── exhaustive 32-layer control
│
├── KV/state evolution validation
│
▼
T6-64+
│
├── extended autoregressive generation
│
▼
PROFILE-418
│
├── complete wall-time decomposition
│
▼
G10
│
├── Q6_K CPU
│
├── Q4_0 CPU
│
├── Q4_0 NPU
│
▼
G9
│
├── batched GEMV correctness
│
├── batched GEMV performance
│
▼
optimized E2E
```

---

# Historical Status Notice

Results from earlier campaign dates remain useful for traceability but no longer represent the current project state.

Sections describing the state on:

```text
04 September 2026
08 September 2026
11 September 2026
```

should be considered historical baselines.

In particular, previous statements such as:

```text
Qwen3.5-9B E2E semantic generation | NOT CLOSED
```

are superseded by the 16 September 2026 T6 result.

Historical sections should be marked:

```text
HISTORICAL — SUPERSEDED BY 16 SEP 2026 STATUS
```

They should not be deleted when they contain useful evidence about failed approaches, invalidated performance claims, numerical failures, or the progression of the reverse-engineering and compiler work.

---

# Current Project State

As of 16 September 2026:

```text
Transport                     PASS
Signed Q4_0 GEMV              PASS / ~95% validation coverage
Differential block validation PASS
Cross-layer validation        PASS
Real-state NPU execution      PASS
32-layer model chain          PASS
Real tokenizer                PASS
Full 248,320 logits           PASS
Resident 7-worker pool        PASS
8-position prefill            PASS
8-token generation            PASS
Complete E2E execution        PASS

Strict CPU/NPU equivalence    OPEN — T6-b
Exhaustive 32-layer sweep     OPEN
Long generation               OPEN
LM-head NPU optimization      OPEN — G10
Batched GEMV optimization     OPEN — G9
```

The project has therefore moved from:

```text
isolated NPU kernel correctness
```

through:

```text
real transformer-state execution
```

to:

```text
complete GGUF → tokenizer → prefill → transformer → logits
→ autoregressive state update → generation execution
```

on the XDNA2 path.

The remaining work is now concentrated on:

```text
numerical equivalence
reproducibility
long-context robustness
complete performance attribution
LM-head acceleration
batched GEMV
model-wide optimization
```












NPU MINIMAL — 9B on NPU without FLM (status 04/09/2026)
Self-contained and reproducible dossier: everything required to replay the only 2 NPU paths that produce CLEAN text on this machine. No absolute paths required — binary + DLLs + kernels + models are included in this folder (models are hardlinks → do not delete them here if the source must remain).

Contents
Element	Role
bin/	llama-cli.exe (repo_0808, commit 18b583a) + 6 DLLs (ggml-xdna.dll = the per-op NPU runtime)
kernels/	complete kernel cache (45 .xclbin+.insts pairs + flowkv subdirectories) — INT4 GEMV, INT4 SwiGLU, FlowKV
models/	Llama-3.2-1B-Instruct-Q4_0.gguf (773 MB) + Qwen3.5-9B-q40-lmhead-f16.gguf (5.3 GB) — hardlinks
run_1B_npu_propre.sh	1B complete per-op NPU path — clean text ~5.2 t/s
run_9B_npu_conservateur.sh	9B conservative NPU path (INT4 GEMV+SwiGLU on NPU, attention on CPU) — clean text ~1.8 t/s
INFOS_MANQUANTES.md	THE list of precise information still missing to finish the 9B on NPU
How to launch (from this folder)
# 1B — complete per-op path on NPU (attention included), clean text
bash run_1B_npu_propre.sh

# 9B — GEMV + SwiGLU on NPU, CPU attention (only clean 9B path)
bash run_9B_npu_conservateur.sh
Expected results:

1B: “The capital of France is Paris.” — Prompt ~150 t/s, Generation ~5.2 t/s
9B: “Thinking Process: ... Capital City: Paris ...” — Prompt 8.9 t/s, Generation 1.8 t/s
Why this folder exists
Everything else (f3best fused 33.9 t/s mechanical, OGA, FLM, multi-layer runlist batching) is either numerically broken, non-reproducible, or invalidated by measurement. These 2 scripts are the ONLY states that generate clean text on NPU, verified from the archive on 04/09/2026.

What is still missing to go further (summary)
Read INFOS_MANQUANTES.md — in one sentence:

Map of the 8 linear-attention layers of the hybrid 9B (position + SSM geometry) → unlocks dispatch of the 24 full layers through FlowKV (→ ~3–5 t/s expected)
Correction of softmax×V in the fused kernel (source .s + op-by-op BF16 dump) → unlocks the 33.9 t/s mechanical path (→ ~30 t/s)
µs-level decomposition of graph_compute per op (tooling, not research)
Technical notes
The binary loads kernels from GGML_XDNA_CACHE_DIR (here kernels/). A missing kernel → JIT compilation (slow, ~minutes) or CPU fallback depending on the case.
GGML_XDNA_NUM_COLS=8: the 8-column kernels are in the cache.
The 9B requires -c 512 and ~20 GB of free RAM (default context = ~30 GB).
Models are HARDLINKED to their source: runtimes_permanents/... and E:\Qwen3.5-9B-q40-lmhead-f16.gguf. Deleting a hardlink does not delete the data unless it is the last hardlink.
Binary identical to the archives (md5 91f3ead2...) — it is the same llama-cli.exe as runtimes_permanents/perop_repo0808 and /e/tmp/repo_0808/build/bin/Release.
Evidence / history
Session 08/08: ggml-xdna backend validated, 42.8 t/s f3best = false positive (min_prefix_match=1)
Sessions 03–04/09: 9B hybrid identified (GateDeltaNet 24+8), conservative mode = clean text 1.8 t/s; kernel×host isolation matrix; verdict #258 H2 (softmax×V); runlist batching invalidated by measurement (calls /2 → same time)
Detailed report: E:\trixdna_test\docs\COMMENT_LES_AUTRES_DEPASSENT_EN_TOKEN_04_09.md
Correction of softmax × V
If the 24/8 inversion has already been corrected in your local code, the main problem becomes numerical validation of the fused attention kernel. The public README still displays the old formulation, so this correction will also need to be pushed.

1. Correct softmax × V
For the 8 full-attention layers of Qwen3.5-9B, the exact contract is:

16 Q heads;
4 K/V heads;
head dimension: 256;
GQA grouping: kv_head = q_head / 4;
scale: 1 / sqrt(256) = 1/16;
RoPE applied only to the first 64 dimensions;
softmax performed over the temporal dimension;
output gate applied after attention and before o_proj.
The official configuration confirms these dimensions: Qwen3.5-9B config.json.

The CPU reference to reproduce exactly is:

for (int hq = 0; hq < 16; ++hq) {
    const int hkv = hq / 4;

    float max_score = -INFINITY;

    for (int t = 0; t < seq_len; ++t) {
        float dot = 0.0f;

        for (int d = 0; d < 256; ++d) {
            dot += float(q[hq][d]) * float(k[t][hkv][d]);
        }

        score[t] = dot * (1.0f / 16.0f);
        max_score = std::max(max_score, score[t]);
    }

    float denominator = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        probability[t] = expf(score[t] - max_score);
        denominator += probability[t];
    }

    const float inverse_sum = 1.0f / denominator;

    for (int d = 0; d < 256; ++d) {
        float accumulator = 0.0f;

        for (int t = 0; t < seq_len; ++t) {
            accumulator += probability[t]
                         * inverse_sum
                         * float(v[t][hkv][d]);
        }

        output[hq][d] = accumulator;
    }
}
Most probable cause
The kernel names in the repository are already consistent with H16_KV4_d256. The problem is therefore probably in one of these four locations:

1. Wrong GQA mapping
Using hkv = hq % 4 is wrong. It must be:

hkv = hq / 4

Q 0–3    → KV 0
Q 4–7    → KV 1
Q 8–11   → KV 2
Q 12–15  → KV 3
2. Wrong V memory order
The producer may write [token][kv_head][dim] while the kernel reads [kv_head][token][dim]. Such an error produces exactly the symptom “reasonable values but incoherent text”.

3. Incorrect softmax reduction
The maximum and sum must cover all valid tokens for one Q head — never the head dimensions, the 4 KV heads, or only a local block of 32 values.

4. BF16 conversion too early
The following should remain in float or a wide accumulator:

Q·K dot product;
maximum;
exponential sum;
probability × V accumulation.
BF16 conversion should occur only at the block output. The official MLIR-AIE examples provide BF16 softmax and vector operations that can be used as a reference.

2. Run a test that localizes the fault in one execution
Compare CPU and NPU step by step, on a single full-attention layer and a single decode token.

Use seq_len = 4, then 16, 64 and 256. For every step, export:

Q after norm/RoPE
K after norm/RoPE
V
scores QK before scale
scores after scale
max(scores)
exp(scores - max)
sum of exp
normalized probabilities
softmax × V result
result after gate
result after o_proj
For each tensor, calculate:

max_abs_error
mean_abs_error
cosine_similarity
number of NaN/Inf
index of the first divergence
Initial reasonable criteria:

Q/K converted to BF16: cosine > 0.999;
scores: cosine > 0.995;
probabilities: sum between 0.995 and 1.005;
attention output: cosine > 0.99;
no NaN or Inf.
Do not test only the prompt “capital of France”. Correct text can hide errors. Use deterministic synthetic inputs:

Q and K zero: uniform softmax;
one highly dominant score: output ≈ one V vector;
V identical for all tokens: output must be exactly V;
one non-zero component in each V: immediately reveals a transposition;
negative Q/K values: detects signed/conversion errors.
The “V identical” test is the fastest: if softmax×V != V, the problem is necessarily in normalization, strides, DMA, or accumulation.

3. Separate the fused kernel before repairing it
The many rawscore, smax, scopy, sdirect, vexp and fix kernels show that isolation has already been attempted. It should be formalized into four reference kernels:

A. QK only

B. QK + softmax

C. softmax × V with probabilities supplied by the CPU

D. QK + softmax × V

Interpretation:

A wrong → Q/K, RoPE, scale, GQA or stride;
A correct, B wrong → max/exp/sum reduction;
B correct, C wrong → V layout or accumulator;
A/B/C correct, D wrong → synchronization, buffer reuse or DMA;
D correct but model wrong → gate, o_proj, KV cache or GGML integration.
C must be correct before re-enabling the fused kernel. This is the shortest path to the actual cause.

4. Verify the KV cache
Only the 8 full-attention layers use a conventional KV cache.

For each layer, verify:

K cache: [token][4][256]
V cache: [token][4][256]
Or document precisely any other layout.

Mandatory tests:

after token 0, read K/V back from the NPU and compare with CPU;
after token 1, verify that token 0 has not been overwritten;
verify the offset in bytes, not only in elements;
verify the alignment required by DMA;
verify that seq_len means the number of valid tokens and not buffer capacity;
apply the causal mask before the softmax maximum.
Do not use FlowKV for the 24 Gated DeltaNet layers: they require recurrent DeltaNet state and a causal convolution state, not a conventional K/V cache.

5. Properly fix Gated DeltaNet
Each linear-attention layer must preserve at minimum:

causal convolution state of size 4;
recurrent DeltaNet state per head;
decay/gate parameters;
exact operation and normalization ordering.
The safe strategy is:

Keep Gated DeltaNet on CPU.
Accelerate only its INT4 projections on NPU.
Compare the complete output of each layer against CPU.
Then port the causal convolution.
Finally port the DeltaNet recurrence.
Only fuse after token-by-token validation.
The state must be checked after multiple tokens, not only the first one: an incorrect recurrent update may be exact at token 0 and then progressively diverge.

6. Avoid false batching gains
The README indicates that halving the number of calls did not reduce the time. This probably means:

weights are still reread;
the same DMA transfers are executed;
the runtime waits for each command;
or kernels are still serialized.
Measure separately:

host preparation
host → NPU copy
loading/reconfiguration
AIE execution
NPU → host copy
waiting/synchronization
The fix is not simply to group several commands. Fusion must actually eliminate:

intermediate transfers;
weight rereads;
overlay changes;
host waits between operations.
The best objective is a resident block:

RMSNorm
→ QKV projections
→ attention or DeltaNet
→ output projection
→ residual
→ RMSNorm
→ FFN/SwiGLU
→ residual
with activations kept on the NPU between stages.

7. Do not take 30 tokens/s as an established target
A weight file of approximately 5.3 GB read once per token represents:

5.3 × 30 ≈ 159 GB/s
And this does not include activations, caches, rereads, or transfers.

On some XDNA2 machines, 30 tokens/s may be close to or beyond the actually exploitable NPU bandwidth.

Before announcing this figure, measure:

bytes transferred per token;
number of reads of each matrix;
effective NPU bandwidth;
reconfiguration time;
total time per layer.
A more defensible initial objective would be:

numerically correct text;
100% of layers validated;
fewer dispatches;
reproducible improvement over 1.8 tokens/s;
measured power consumption.
8. What is actually missing to apply the fix
The public repository does not contain:

the source of ggml-xdna.dll;
the .s/C++ source of the fused kernel;
the host code that prepares the buffers;
the .xclbin/.insts generator;
CPU/NPU traces;
the source commit corresponding to binary 18b583a.
Without these elements, the correction method can be determined, but the actual bug cannot be modified. The executables alone cannot establish whether the fault is in the kernel, DMA, or GGML backend.

The concrete priority is therefore:

Publish or recover the exact source of ggml-xdna.dll and f3best.
Add the isolated softmax×V test with identical V.
Verify hkv = hq / 4.
Lock down the layout [token][kv_head][dim].
Accumulate softmax and P×V in FP32.
Test C alone, then re-enable fusion.
Add per-operation traces.
Optimize dispatches only after numerical equality.
The most likely correction is a combination of “GQA mapping + V stride/layout + block-wise softmax reduction”. That is where I would immediately focus the investigation.

XDNA2 Experimental LLM Runtime
Experimental work on accelerating LLM inference on AMD XDNA2 NPUs through a custom ggml/llama.cpp backend and dedicated AIE kernels.

The repository is intended to provide a reproducible environment for testing and characterizing XDNA2 inference, including decode performance, kernel execution, memory behavior, and heterogeneous CPU/NPU execution.

Project scope
The current implementation includes:

Experimental ggml-xdna backend
XDNA2 NPU execution path
INT4 GEMV kernels
INT4 SwiGLU kernels
FlowKV / attention experiments
Compiled .xclbin and .insts kernel artifacts
llama.cpp integration
Reproducibility scripts
Reference Qwen3.5 implementation
Performance and validation experiments
The compiled kernel artifacts included in the repository correspond to the configurations used for the documented experiments.

Upstream projects and technical references
This work builds upon and references several open-source projects.

llama.cpp / ggml
Repository:

https://github.com/ggml-org/llama.cpp

llama.cpp provides the inference engine and ggml execution framework used as the host architecture for the experimental XDNA2 backend.

License: MIT

Copyright and licensing remain with the respective upstream authors.

AMD IRON
Repository:

https://github.com/amd/IRON

IRON provides open-source infrastructure and examples for programming AMD Ryzen AI NPUs.

The project includes NPU-oriented kernels and examples covering operations relevant to LLM inference, including matrix operations, attention-related workloads and other AIE compute primitives.

License: Apache License 2.0.

MLIR-AIE
Repository:

https://github.com/Xilinx/mlir-aie

MLIR-AIE provides the compiler infrastructure and programming model used for targeting AMD/Xilinx AI Engine architectures.

It includes tooling for generating executable NPU artifacts and instruction streams used by AIE applications.

License: Apache License 2.0 with LLVM exceptions where applicable.

AMD XDNA
Repository:

https://github.com/amd/xdna-driver

The AMD XDNA open-source project provides architecture and runtime references for XDNA-based accelerators and is an important technical reference for understanding the execution environment targeted by this project.

Refer to the upstream repository for the licenses applicable to individual components.

Hugging Face Transformers / Qwen3.5
Repository:

https://github.com/huggingface/transformers

Qwen:

https://github.com/QwenLM

The Qwen3.5 reference implementation is used for architectural and numerical comparison with the accelerated implementation.

The corresponding Transformers implementation is distributed under the Apache License 2.0.

Copyright remains with the Qwen Team, Hugging Face, and the respective upstream contributors.

Architecture
The experimental execution stack can be summarized as:

Qwen3.5
   |
   v
llama.cpp / ggml
   |
   v
ggml-xdna
   |
   +-----------------------+
   |                       |
   v                       v
Host execution        XDNA2 execution
                           |
                  +--------+--------+
                  |        |        |
                  v        v        v
               GEMV     SwiGLU    FlowKV
                  \        |        /
                   \       |       /
                    v      v      v
                  AIE kernels
                       |
                       v
                .xclbin / .insts
Kernel artifacts
The repository contains compiled kernel configurations used by the experimental runtime.

These include kernels targeting operations such as:

INT4 GEMV
INT4 SwiGLU
FlowKV
attention-related execution paths
.xclbin and .insts files are executable artifacts consumed by the XDNA2 execution path.

Different kernel variants correspond to different tensor dimensions and execution configurations required by the model.

Keeping the tested artifacts together with the runtime allows benchmark configurations to remain reproducible.

Reproducibility
The repository is designed around reproducible measurements rather than isolated peak-performance results.

When comparing configurations, relevant parameters should therefore be recorded, including:

model
quantization
context size
prompt length
generated token count
kernel configuration
CPU/NPU placement
runtime version
thermal conditions
prefill throughput
decode throughput
Throughput is expressed in tokens per second (tokens/s).

Prefill and decode measurements should be considered separately because they exercise substantially different computational and memory-access patterns.

Experimental status
This project is research and experimental software.

The XDNA2 backend and kernels are under active development. Performance characteristics, kernel selection and supported model configurations may therefore change between revisions.

Results should be associated with the exact repository revision and test configuration used to produce them.

Licensing and attribution
This repository combines original experimental work with interfaces, references and components from open-source projects.

Major upstream projects include:

Project	Upstream	License
llama.cpp / ggml	https://github.com/ggml-org/llama.cpp	MIT
AMD IRON	https://github.com/amd/IRON	Apache-2.0
MLIR-AIE	https://github.com/Xilinx/mlir-aie	Apache-2.0 / LLVM exceptions where applicable
AMD XDNA	https://github.com/amd/xdna-driver	See upstream component licenses
Hugging Face Transformers	https://github.com/huggingface/transformers	Apache-2.0
Qwen	https://github.com/QwenLM	See individual model/repository license
Each upstream component remains subject to its respective copyright and license terms.

Original project-specific code, scripts, experimental integration work and documentation should be considered separately from third-party components and retain the licensing specified by this repository.

References
ggml-org — llama.cpp https://github.com/ggml-org/llama.cpp
AMD — IRON https://github.com/amd/IRON
AMD/Xilinx — MLIR-AIE https://github.com/Xilinx/mlir-aie
AMD — XDNA https://github.com/amd/xdna-driver
Hugging Face — Transformers https://github.com/huggingface/transformers
Qwen Team — Qwen https://github.com/QwenLM update # XDNA2 NPU — Qwen3.5-9B — Reproducible Validation
Current status — 08/09/2026
This repository contains a frozen, reproducible validation chain for the AMD XDNA2 NPU on Windows/Strix Point, with a focus on the f3best full-K kernel used by the Qwen3.5-9B path.

The current state must be separated into two distinct claims:

The f3best RR full-K NPU dispatch is validated.
End-to-end semantic Qwen3.5-9B generation at the previously observed ~34 tok/s is NOT validated.
These are different validation levels and must not be conflated.

Validation status
Area	Status	Meaning
Frozen f3best ctrlcode	PASS	Byte-identical to the frozen reference artifact
Frozen f3best xclbin	PASS	Reproducible geometry / ctrlcode contract
XDNA2 dispatch	PASS	Kernel reaches state=4 deterministically
BO ABI	PASS	5 data BOs + instruction BO, exact argument contract
BO dependency	PASS	bo1..bo4 demonstrably affect the result
All-AA deterministic replay	PASS	Inter-process deterministic output
Steady-state NPU execution	PASS	~4.89 ms NPU wait
Host submit overhead	LOW	~54.5 µs
Forced 4 MiB readback	ARTIFICIAL	Diagnostic only, not representative of runtime
Full numerical f3best semantics	NOT YET CLOSED	Determinism is not numerical correctness
Qwen3.5-9B E2E semantic generation	NOT CLOSED	Still the active integration target
~34 tok/s E2E claim	NOT VALIDATED	Previous mechanical benchmark was not a semantic throughput proof
1. Frozen f3best witness
The frozen artifact is:

f3best — 9B — s128 — a2_kv4 — RR — full-K

The ctrlcode is 3172 bytes / 793 words.

The frozen ctrlcode is byte-for-byte identical to the corresponding reference artifact used in the previous functional runtime.

The frozen xclbin and ctrlcode are recorded together with SHA-256 provenance in the reproducibility package.

The purpose of this artifact is to establish a stable NPU execution target before attempting further end-to-end integration.

2. Important correction to the previous witness
The first A2 replay used:

bo0 = 0xAA
bo1..bo4 not explicitly initialized by the replay tool
Therefore the original output hash was process-dependent.

That result must not be used as the byte-level reproducibility contract.

The corrected reproducibility contract initializes all five data BOs to 0xAA.

The official deterministic witness is therefore:

5 data BOs
4 MiB each
all initialized to 0xAA
instruction BO containing the frozen ctrlcode
arg0 = 3
arg1 = instruction BO
arg2 = 3172
arg3..arg7 = data BOs
Expected state:

state = 4
Expected deterministic output:

SHA256:
df78300f8b6364169706285ee1cc9aa0f6da921f722cc8d97a8ef5d6546d54f4
The corresponding diagnostic output contains the expected BF16 pattern over the useful output region, with the remaining bytes retaining the initialized BO pattern.

3. BO dependency was independently demonstrated
A zero-control experiment was performed with:

bo0 = 0xAA
bo1..bo4 = 0x00
The kernel still reached:

state = 4
but produced an all-zero useful output.

This proves that the kernel does not operate solely on bo0.

Therefore the previous process-dependent A2 witness was explained by the fact that the replay tool did not initialize all input BOs.

The corrected B1 protocol removes this ambiguity.

4. B1 temporal decomposition
B1 uses persistent BOs and separates the host-side components of one dispatch.

Protocol:

3 warmup runs
10 measured runs
5 × 4 MiB data BOs
persistent allocation
single H2D initialization
NPU execution
D2H synchronization
forced 4 MiB diagnostic readback
Measured steady-state values:

Component	Time
Host submit()	~54.5 µs
NPU wait	~4,889.8 µs
D2H synchronization	~58.4 µs
Forced 4 MiB CPU read	~1,222.5 µs
Complete diagnostic cycle	~6,225.2 µs
The critical number is:

NPU wait ≈ 4.89 ms
This matches the independent xclbin_replay measurement of approximately:

4.905 ms
The difference is approximately 0.3%.

Therefore the ~4.9 ms execution time is a real steady-state NPU/AIE execution cost.

It is not explained by:

XRT submit() overhead
host-side synchronization
the diagnostic readback
BO allocation churn
5. Important interpretation of the 4.89 ms
The 4.89 ms number must be described carefully.

It is the time spent waiting for completion of the AIE/XDNA2 execution.

It does not mean that 4.89 ms is pure matrix multiplication time.

The f3best emitter contains AIE-side:

DMA descriptors
DMA starts
lock operations
buffer cycling
synchronization
compute operations
Therefore the 4.89 ms includes the internal execution/dataflow schedule of the kernel.

The next optimization target is consequently the AIE execution/dataflow itself, not host submit() overhead.

6. Diagnostic readback caveat
The B1 protocol intentionally reads the complete 4 MiB output BO.

This is a diagnostic operation and must not be interpreted as the normal runtime cost.

Only approximately 72 KiB of the BO are useful for the current diagnostic.

The measured:

~1.22 ms
therefore represents an intentionally forced 4 MiB CPU-mapped read.

It is not the expected production readback cost.

The useful-output read is expected to be on the order of tens of microseconds rather than milliseconds.

7. What is proven
The current evidence proves:

frozen ctrlcode
      ↓
frozen xclbin
      ↓
XDNA2 registration
      ↓
correct kernel ABI
      ↓
5 data BO dependency
      ↓
state=4
      ↓
deterministic output
      ↓
stable ~4.89 ms NPU execution
This is a valid NPU execution witness.

It establishes that the f3best RR full-K artifact is executing on the XDNA2 NPU with a reproducible ABI and deterministic diagnostic behavior.

8. What is NOT yet proven
The following claims remain open:

Numerical correctness
Deterministic output does not imply that the output is numerically correct for the intended Qwen3.5-9B computation.

The current witness uses synthetic initialized BO contents.

Numerical validation must compare the kernel against an independent CPU/reference implementation using controlled tensors and known expected results.

Full f3best semantic correctness
The complete mathematical semantics of the fused kernel are not yet closed.

End-to-end Qwen3.5-9B generation
A successful kernel dispatch does not imply that the complete Qwen3.5-9B model generates correct text.

The remaining chain includes:

model weights
→ quantization/dequantization
→ projections
→ normalization
→ attention / GDN
→ KV state
→ residuals
→ LM head
→ sampling
→ token generation
Each stage must be validated independently before claiming end-to-end performance.

9. Correction regarding the previous ~34 tok/s result
A previous f3best benchmark produced approximately:

~33.9 tok/s
That result must be classified as:

MECHANICAL / NON-SEMANTIC THROUGHPUT
It is not a validated Qwen3.5-9B semantic generation rate.

Earlier measurements were also affected by an overly permissive prefix-matching condition (min_prefix_match=1), which could produce a false-positive generation result.

Therefore:

~33.9 tok/s ≠ validated Qwen3.5-9B performance
The number must not be used as the current performance claim.

However, this does not invalidate the f3best kernel itself.

The correct distinction is:

f3best NPU dispatch:
VALIDATED as deterministic execution

f3best numerical semantics:
NOT YET CLOSED

Qwen3.5-9B E2E:
NOT YET CLOSED

~34 tok/s semantic throughput:
NOT VALIDATED
10. Current performance interpretation
The single f3best dispatch currently costs approximately:

4.89 ms
at the NPU execution level.

A purely arithmetic conversion would give:

1000 / 4.89 ≈ 204 dispatches/s
but this must NOT be interpreted as tokens/s.

One Qwen token requires multiple layer operations and multiple dispatches.

Therefore the correct next step is to establish:

number of f3best dispatches / token
×
4.89 ms / dispatch
and then add:

GDN layers
attention layers
KV handling
host orchestration
other NPU kernels
CPU work
synchronization
Only that complete composition can produce a defensible tokens/s estimate.

11. Current bottleneck hypothesis
The current B1 evidence changes the optimization priority.

The immediate problem is no longer:

"XRT submit is too slow"
or:

"the 4 MiB readback is dominating the kernel"
for the single-dispatch protocol.

Instead, the primary target is:

AIE-side execution/dataflow
including:

internal DMA
lock synchronization
buffer scheduling
tile utilization
compute occupancy
memory movement inside the AIE graph
possible stalls between producer/consumer stages
The .bin transaction stream and the emitter source should therefore be correlated with the measured 4.89 ms timeline.

12. Recommended validation sequence
The remaining work should proceed in this order:

A — Frozen artifact
    CLOSED

B1 — Temporal decomposition
     CLOSED

B2 — Internal transaction/dataflow decomposition
     NEXT

B3 — Numerical kernel validation
     NEXT

B4 — Real Qwen tensor integration
     NEXT

E2E — Full Qwen3.5-9B semantic generation
      FINAL GATE
The E2E benchmark should only be published after the semantic output has been independently validated.

13. Reproducibility contract
A reproduction is valid only if all of the following are satisfied:

same ctrlcode SHA
same xclbin SHA
same instruction length
same kernel ABI
same BO sizes
same BO initialization
state = 4
expected output SHA
stable NPU timing
The all-0xAA five-BO protocol is the current byte-level diagnostic witness.

The older witness using uninitialized bo1..bo4 is retained only as historical evidence and must not be presented as the canonical reproducibility result.

14. Repository structure
reproductible/
├── MANIFEST.md
├── SHA256SUMS.txt
│
├── temoin/
│   └── frozen f3best RR full-K artifact
│
├── donnees/
│   └── B1 measurements / raw outputs
│
├── rapports/
│   ├── ANGLE_A_GEL_T17RR0906_08_09_2026.md
│   └── B1_RAPPORT_DECOMPOSITION_08_09_2026.md
│
├── scripts/
│   ├── b1_decompose.py
│   ├── b1_decompose_v2.py
│   ├── b1_diag_allaa.py
│   ├── b1_diag_zerobo.py
│   ├── decode_t17_order.py
│   └── decode_t17_txn.py
│
└── sources/
    ├── compile_f3best_9b_s128_rr.py
    └── f3best_emit_nokv.py
15. Bottom line
The current state is:

The f3best RR full-K kernel is a real, deterministic XDNA2 execution artifact with a measured steady-state NPU execution time of approximately 4.89 ms per dispatch.

The previous ~34 tok/s result is not a valid semantic Qwen3.5-9B throughput claim.

The next bottleneck investigation should focus on the AIE-side dataflow and internal DMA/lock schedule, followed by numerical validation and only then full end-to-end generation.

NA2 / ggml-xdna — Technical Status Update (11 Sep 2026)

Current technical status as of 11 Sep 2026. This section supersedes the historical README status that described the project as of 4 Sep 2026.

Current Status

This repository has moved beyond the demonstration of clean NPU execution paths. The current work is focused on bit-exact differential diagnosis of an end-to-end numerical corruption observed on the XDNA2 NPU path.

The investigation is split into two distinct tracks:

B4→B9: hardware/data-path attribution — CLOSED, no fix required

STAGE: numerical corruption localization — IN PROGRESS

B4→B9 — Hardware/Data-Path Attribution: CLOSED

Question

Does the DMA fail to read Q, or is there a +0x240000 offset / lost Q fill region?

Final result

No corrective change is required.

The investigation established the following:

No +0x240000 offset was demonstrated.

No lost Q fills were demonstrated.

Memtile BDs form a generic ring and do not encode a Q/O/G/U/D-specific distinction.

Shim BDs operate on the full frame.

The runtime passes the full BO; there is no section-level patch that removes Q.

The earlier B4/B6/B6d influence maps are retired as causal evidence because they were obtained under a degenerate KV regime.

B6F repeated the visibility experiment with valid randomized bf16 KV data and confirmed that the Q prefix is active.

B6F replay matched T0 and all 10 dispatches completed successfully.

The B4→B9 track is therefore closed without modifying the runtime.

Important correction to the earlier interpretation

The +0x240000 hypothesis came from influence maps obtained under a degenerate regime where constant K caused uniform softmax behavior and made attention insensitive to Q. Those maps were not sufficient to establish a missing-Q or DMA-offset bug.

The corrected B6F experiment removed that ambiguity by using randomized bf16 KV data and directly restored Q sensitivity.

STAGE — Numerical Corruption Diagnosis: IN PROGRESS

Objective

Identify the first buffer or tensor that diverges from the correct CPU reference on the NPU path, and determine whether the divergence is caused by computation, packing, buffer state, planning, or synchronization.

Established results

STAGE 4 — GEMV INT4 K4096→N8192

The GEMV INT4 K4096→N8192 path was independently cleared:

5/5 probes passed.

Maximum observed difference was 2 bf16 ULP.

The result was consistent across the tested Q/K/V/gate probes.

This path is therefore not currently considered the source of the observed corruption.

STAGE 5 — First-layer corruption

The first fused layer is already corrupted on affected execution paths.

The GEMV K4096→N4096 path using the real SSM weights was also cleared by the corresponding differential checks.

STAGE 5C — SwiGLU isolation

The control matrix established that SwiGLU alone does not explain the corruption.

The current evidence points instead toward an interaction involving the fused data path, buffer state, packing, or execution plan.

STAGE 5G-B — First divergent node

Nodes 16 and 17 of block 31 are clean:

SHA-256 matches the reference.

Binary comparison reports no differences.

The first currently localized divergence is:

Node 44 — CONT gate_reshaped-31

Observed signature:

sentinel / poison pattern: 0x7FC1

the buffer is only partially written

at M=2, 3824 / 8192 values become NaN with the same 0x7FC1 pattern

the same behavior is not reproduced at the same location for the M=4 and M=11 regimes

This is currently the strongest diagnostic signal in the project.

Current Working Interpretation

The evidence no longer points to a missing-Q or simple attention-layout problem.

The strongest current hypothesis is:

A partial buffer write occurs under specific M-dependent execution conditions.

This is not yet sufficient to assign the fault to a specific layer of the stack. The remaining possibilities include:

host-side packing of the fused weight buffer;

buffer reuse or lifetime;

GEMV output / input hand-off into the following operation;

execution-plan partitioning;

incomplete or conditional writes;

synchronization / completion visibility.

No single one of these has yet been proven as the root cause.

Next Experiments

The current investigation proceeds in this order:

STAGE 5C-A.2 — Bit-exact fused-weight comparison
Compare the C++ w_fused_bo contents directly against the Python reference blob.

The comparison must be byte-for-byte and must record:

SHA-256;

raw size;

exact offsets;

representative byte samples;

the region corresponding to the affected gate_reshaped-31 data.

This separates:

C++ packing is wrong

from

the design/runtime expects a different layout.

STAGE 5C-B — c_bo census
Use the existing XDNA_NAN_SWEEP instrumentation to identify which GEMV buffer regions become invalid and whether the same M-dependent write boundary is reproduced.

STAGE 5G-B(a) — Single-plan test
Run:

XDNA_LAYER_PLAN=1

on node 44 to test the hypothesis that the corruption is caused by an execution-plan hole or partitioning decision.

STAGE 5D — GEMV implementation matrix
Repeat the A..F control matrix with:

XDNA_DISABLE_GEMV_INT4_V2=1

This determines whether the V2 GEMV implementation participates in the observed corruption.

STAGE 6
Proceed only after the first-divergent-buffer investigation has been completed and the responsible mechanism has been narrowed sufficiently.

Diagnostic Rules

The following conclusions are now considered closed unless new experimental evidence directly contradicts them:

no demonstrated +0x240000 Q offset;

no demonstrated lost Q fills;

no current evidence for a BD/SHIM geometry correction;

no current evidence that RMSNorm arithmetic alone is the root cause;

no current evidence that GEMV INT4 K4096→N8192 arithmetic is the root cause.

The project should therefore remain focused on the first divergent buffer and the M-dependent partial-write signature.

Repository Integrity

This update documents experimental results, diagnostic conclusions, and the remaining test sequence.

It does not represent a corrective modification to ggml-xdna.cpp.

No runtime fix is claimed at this stage.

MODEL-DRIVEN XDNA2 CODEGEN
The project evolves from a fixed kernel/runtime architecture toward a model-specialized compiler that generates the XDNA2 execution program directly from the target model.

The model is treated as compiler input.

                     MODEL
                       │
                       ▼
             ┌─────────────────────┐
             │   MODEL ANALYZER     │
             │                     │
             │ • architecture     │
             │ • tensors/shapes    │
             │ • quantization      │
             │ • attention/KV      │
             │ • memory footprint  │
             │ • dependencies      │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │       MODEL IR       │
             └──────────┬──────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    COMPUTE IR      MEMORY IR      SCHEDULE IR
         │              │              │
         └──────────────┼──────────────┘
                        ▼
             ┌─────────────────────┐
             │    XDNA2 CODEGEN     │
             │                     │
             │ • GEMV/GEMM         │
             │ • FFN/SwiGLU        │
             │ • attention         │
             │ • DMA               │
             │ • memory layout     │
             │ • tile placement    │
             │ • synchronization   │
             │ • fusion            │
             │ • precision         │
             └──────────┬──────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ XDNA2 EXECUTABLE     │
             │                     │
             │ xclbin / PDI        │
             │ instruction stream  │
             │ BO/DMA plan         │
             │ kernel metadata     │
             └──────────┬──────────┘
                        │
                        ▼
                   XDNA2 NPU
                        │
                        ▼
                 VALIDATED OUTPUT
CORE PRINCIPLE
There is no fixed universal kernel catalog.

For each model/configuration, the compiler generates a specialized XDNA2 program optimized for:

model architecture
tensor shapes
quantization
memory layout
NPU topology
tile allocation
DMA traffic
synchronization
operator fusion
CPU/NPU placement
context configuration
The generated program is compiled once and cached.

Decode then reuses the specialized executable instead of recompiling for every token.

SPECIALIZATION KEY
A generated executable is uniquely associated with:

MODEL_HASH
ARCHITECTURE
QUANTIZATION
TENSOR_SHAPES
CONTEXT_CONFIGURATION
NPU_TOPOLOGY
MEMORY_BUDGET
COMPILER_VERSION
Example:

model + Q4/Q8 + sequence shape + context
    │
    ▼
specialized XDNA2 program
    │
    ▼
compile
    │
    ▼
cache
    │
    ▼
repeated inference
COMPILER OPTIMIZATION SPACE
The compiler jointly optimizes:

COMPUTE
  ├─ GEMV
  ├─ GEMM
  ├─ FFN
  ├─ SwiGLU
  ├─ attention
  └─ normalization

MEMORY
  ├─ tensor packing
  ├─ BO layout
  ├─ L1/L2 placement
  ├─ DMA descriptors
  ├─ read/write regions
  └─ cache reuse

EXECUTION
  ├─ tile assignment
  ├─ column utilization
  ├─ synchronization
  ├─ instruction ordering
  ├─ fusion
  └─ CPU/NPU partitioning

PRECISION
  ├─ INT4
  ├─ INT8
  ├─ BF16/FP16
  └─ mixed precision
COST MODEL
Optimization is not based only on theoretical TOPS.

The compiler evaluates:

total_cost =
    compute
  + DMA
  + memory movement
  + synchronization
  + submission
  + mapping
  + host overhead
  + readback
This allows the generated program to optimize the complete CPU → BO → DMA → NPU → DMA → CPU execution path.

MODEL-SPECIFIC MEMORY GENERATION
Memory layout becomes part of compilation rather than a fixed runtime assumption.

Example model-specific weight packing:

[ Q | O | GATE | UP | DOWN ]

256 + 256 + 768 + 768 + 768 tiles
= 2816 tiles
Observed memory regions and read/write sets become explicit compiler invariants.

Example:

0x240000 = 9 × 256 KiB

affected region:
[0x240000, 0x480000)
The compiler therefore generates both:

COMPUTE PLAN
MEMORY / DMA PLAN
as one coupled program.

AUTOTUNING
The generator can emit several legal variants:

Variant A
  tile layout A
  DMA schedule A
  column allocation A

Variant B
  tile layout B
  DMA schedule B
  column allocation B

Variant C
  fused operators
  different memory placement
  different synchronization
Each variant is:

1. compiled
2. validated against golden output
3. checked for corruption / NaN
4. benchmarked
5. scored by the cost model
The best valid variant is cached.

COMPILATION PIPELINE
GGUF / ONNX / model
         │
         ▼
   Model Analyzer
         │
         ▼
      Model IR
         │
         ├── Compute IR
         ├── Memory IR
         └── Schedule IR
         │
         ▼
    XDNA2 lowering
         │
         ▼
  AIE / DMA generation
         │
         ▼
   xclbin / PDI / TXN
         │
         ▼
   correctness tests
         │
         ▼
    autotuning
         │
         ▼
   executable cache
         │
         ▼
   inference runtime
RUNTIME
The runtime becomes deliberately thin.

Its responsibilities are primarily:

load cached executable
allocate / map BOs
bind model buffers
update runtime arguments
submit execution
maintain KV state
synchronize
return generated tokens
The expensive model-specific decisions are made by the compiler, not by a large collection of hand-written runtime special cases.

VALIDATION MODEL
Every generated program must pass:

golden tensor comparison
deterministic prefix test
NaN / corruption detection
BO bounds validation
DMA read/write validation
instruction validation
output regression tests
Existing B1–B9 investigations become regression tests rather than permanent runtime logic.

TARGET
The long-term objective is:

MODEL
  ↓
AUTOMATIC ANALYSIS
  ↓
AUTOMATIC XDNA2 PROGRAM GENERATION
  ↓
AUTOMATIC VALIDATION
  ↓
AUTOMATIC AUTOTUNING
  ↓
CACHED SPECIALIZED EXECUTABLE
  ↓
E2E LLM INFERENCE
The resulting system is therefore not simply an XDNA2 kernel library.

It is a model-driven XDNA2 compiler/runtime in which the model itself determines the generated compute graph, memory layout, DMA schedule, tile allocation, synchronization and execution strategy.
