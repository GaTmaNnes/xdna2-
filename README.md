

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
