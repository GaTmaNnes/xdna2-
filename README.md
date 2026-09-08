# NPU MINIMAL — 9B on NPU without FLM (status 04/09/2026)

**Self-contained and reproducible dossier**: everything required to replay the only 2 NPU paths that produce CLEAN text on this machine. No absolute paths required — binary + DLLs + kernels + models are included in this folder (models are hardlinks → do not delete them here if the source must remain).

## Contents

| Element                      | Role                                                                                                     |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| `bin/`                       | llama-cli.exe (repo_0808, commit 18b583a) + 6 DLLs (ggml-xdna.dll = the per-op NPU runtime)              |
| `kernels/`                   | complete kernel cache (45 .xclbin+.insts pairs + flowkv subdirectories) — INT4 GEMV, INT4 SwiGLU, FlowKV |
| `models/`                    | Llama-3.2-1B-Instruct-Q4_0.gguf (773 MB) + Qwen3.5-9B-q40-lmhead-f16.gguf (5.3 GB) — hardlinks           |
| `run_1B_npu_propre.sh`       | 1B complete per-op NPU path — clean text ~5.2 t/s                                                        |
| `run_9B_npu_conservateur.sh` | 9B conservative NPU path (INT4 GEMV+SwiGLU on NPU, attention on CPU) — clean text ~1.8 t/s               |
| `INFOS_MANQUANTES.md`        | **THE list of precise information still missing** to finish the 9B on NPU                                |

## How to launch (from this folder)

```bash
# 1B — complete per-op path on NPU (attention included), clean text
bash run_1B_npu_propre.sh

# 9B — GEMV + SwiGLU on NPU, CPU attention (only clean 9B path)
bash run_9B_npu_conservateur.sh
```

Expected results:

* 1B: “The capital of France is Paris.” — Prompt ~150 t/s, Generation ~5.2 t/s
* 9B: “Thinking Process: ... Capital City: Paris ...” — Prompt 8.9 t/s, Generation 1.8 t/s

## Why this folder exists

Everything else (f3best fused 33.9 t/s mechanical, OGA, FLM, multi-layer runlist batching) is either numerically broken, non-reproducible, or invalidated by measurement. These 2 scripts are the ONLY states that generate clean text on NPU, verified from the archive on 04/09/2026.

## What is still missing to go further (summary)

Read `INFOS_MANQUANTES.md` — in one sentence:

1. **Map of the 8 linear-attention layers** of the hybrid 9B (position + SSM geometry) → unlocks dispatch of the 24 full layers through FlowKV (→ ~3–5 t/s expected)
2. **Correction of softmax×V in the fused kernel** (source .s + op-by-op BF16 dump) → unlocks the 33.9 t/s mechanical path (→ ~30 t/s)
3. µs-level decomposition of `graph_compute` per op (tooling, not research)

## Technical notes

* The binary loads kernels from `GGML_XDNA_CACHE_DIR` (here `kernels/`). A missing kernel → JIT compilation (slow, ~minutes) or CPU fallback depending on the case.
* `GGML_XDNA_NUM_COLS=8`: the 8-column kernels are in the cache.
* The 9B requires `-c 512` and ~20 GB of free RAM (default context = ~30 GB).
* Models are HARDLINKED to their source: `runtimes_permanents/...` and `E:\Qwen3.5-9B-q40-lmhead-f16.gguf`. Deleting a hardlink does not delete the data unless it is the last hardlink.
* Binary identical to the archives (md5 91f3ead2...) — it is the same `llama-cli.exe` as `runtimes_permanents/perop_repo0808` and `/e/tmp/repo_0808/build/bin/Release`.

## Evidence / history

* Session 08/08: `ggml-xdna` backend validated, 42.8 t/s f3best = false positive (`min_prefix_match=1`)
* Sessions 03–04/09: 9B hybrid identified (GateDeltaNet 24+8), conservative mode = clean text 1.8 t/s; kernel×host isolation matrix; verdict #258 H2 (softmax×V); runlist batching invalidated by measurement (calls /2 → same time)
* Detailed report: `E:\trixdna_test\docs\COMMENT_LES_AUTRES_DEPASSENT_EN_TOKEN_04_09.md`

---

# Correction of softmax × V

If the 24/8 inversion has already been corrected in your local code, the main problem becomes numerical validation of the fused attention kernel. The public README still displays the old formulation, so this correction will also need to be pushed.

## 1. Correct softmax × V

For the 8 full-attention layers of Qwen3.5-9B, the exact contract is:

* 16 Q heads;
* 4 K/V heads;
* head dimension: 256;
* GQA grouping: `kv_head = q_head / 4`;
* scale: `1 / sqrt(256) = 1/16`;
* RoPE applied only to the first 64 dimensions;
* softmax performed over the temporal dimension;
* output gate applied after attention and before `o_proj`.

The official configuration confirms these dimensions: Qwen3.5-9B `config.json`.

The CPU reference to reproduce exactly is:

```cpp
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
```

## Most probable cause

The kernel names in the repository are already consistent with H16_KV4_d256. The problem is therefore probably in one of these four locations:

### 1. Wrong GQA mapping

Using `hkv = hq % 4` is wrong. It must be:

`hkv = hq / 4`

```text
Q 0–3    → KV 0
Q 4–7    → KV 1
Q 8–11   → KV 2
Q 12–15  → KV 3
```

### 2. Wrong V memory order

The producer may write `[token][kv_head][dim]` while the kernel reads `[kv_head][token][dim]`. Such an error produces exactly the symptom “reasonable values but incoherent text”.

### 3. Incorrect softmax reduction

The maximum and sum must cover all valid tokens for one Q head — never the head dimensions, the 4 KV heads, or only a local block of 32 values.

### 4. BF16 conversion too early

The following should remain in float or a wide accumulator:

* Q·K dot product;
* maximum;
* exponential sum;
* probability × V accumulation.

BF16 conversion should occur only at the block output. The official MLIR-AIE examples provide BF16 softmax and vector operations that can be used as a reference.

---

# 2. Run a test that localizes the fault in one execution

Compare CPU and NPU step by step, on a single full-attention layer and a single decode token.

Use `seq_len = 4`, then 16, 64 and 256. For every step, export:

* Q after norm/RoPE
* K after norm/RoPE
* V
* scores QK before scale
* scores after scale
* max(scores)
* exp(scores - max)
* sum of exp
* normalized probabilities
* softmax × V result
* result after gate
* result after o_proj

For each tensor, calculate:

* `max_abs_error`
* `mean_abs_error`
* `cosine_similarity`
* number of NaN/Inf
* index of the first divergence

Initial reasonable criteria:

* Q/K converted to BF16: cosine > 0.999;
* scores: cosine > 0.995;
* probabilities: sum between 0.995 and 1.005;
* attention output: cosine > 0.99;
* no NaN or Inf.

Do not test only the prompt “capital of France”. Correct text can hide errors. Use deterministic synthetic inputs:

* Q and K zero: uniform softmax;
* one highly dominant score: output ≈ one V vector;
* V identical for all tokens: output must be exactly V;
* one non-zero component in each V: immediately reveals a transposition;
* negative Q/K values: detects signed/conversion errors.

The “V identical” test is the fastest: if softmax×V != V, the problem is necessarily in normalization, strides, DMA, or accumulation.

---

# 3. Separate the fused kernel before repairing it

The many `rawscore`, `smax`, `scopy`, `sdirect`, `vexp` and `fix` kernels show that isolation has already been attempted. It should be formalized into four reference kernels:

**A. QK only**

**B. QK + softmax**

**C. softmax × V with probabilities supplied by the CPU**

**D. QK + softmax × V**

Interpretation:

* A wrong → Q/K, RoPE, scale, GQA or stride;
* A correct, B wrong → max/exp/sum reduction;
* B correct, C wrong → V layout or accumulator;
* A/B/C correct, D wrong → synchronization, buffer reuse or DMA;
* D correct but model wrong → gate, o_proj, KV cache or GGML integration.

C must be correct before re-enabling the fused kernel. This is the shortest path to the actual cause.

---

# 4. Verify the KV cache

Only the 8 full-attention layers use a conventional KV cache.

For each layer, verify:

```text
K cache: [token][4][256]
V cache: [token][4][256]
```

Or document precisely any other layout.

Mandatory tests:

* after token 0, read K/V back from the NPU and compare with CPU;
* after token 1, verify that token 0 has not been overwritten;
* verify the offset in bytes, not only in elements;
* verify the alignment required by DMA;
* verify that `seq_len` means the number of valid tokens and not buffer capacity;
* apply the causal mask before the softmax maximum.

Do not use FlowKV for the 24 Gated DeltaNet layers: they require recurrent DeltaNet state and a causal convolution state, not a conventional K/V cache.

---

# 5. Properly fix Gated DeltaNet

Each linear-attention layer must preserve at minimum:

* causal convolution state of size 4;
* recurrent DeltaNet state per head;
* decay/gate parameters;
* exact operation and normalization ordering.

The safe strategy is:

1. Keep Gated DeltaNet on CPU.
2. Accelerate only its INT4 projections on NPU.
3. Compare the complete output of each layer against CPU.
4. Then port the causal convolution.
5. Finally port the DeltaNet recurrence.
6. Only fuse after token-by-token validation.

The state must be checked after multiple tokens, not only the first one: an incorrect recurrent update may be exact at token 0 and then progressively diverge.

---

# 6. Avoid false batching gains

The README indicates that halving the number of calls did not reduce the time. This probably means:

* weights are still reread;
* the same DMA transfers are executed;
* the runtime waits for each command;
* or kernels are still serialized.

Measure separately:

```text
host preparation
host → NPU copy
loading/reconfiguration
AIE execution
NPU → host copy
waiting/synchronization
```

The fix is not simply to group several commands. Fusion must actually eliminate:

* intermediate transfers;
* weight rereads;
* overlay changes;
* host waits between operations.

The best objective is a resident block:

```text
RMSNorm
→ QKV projections
→ attention or DeltaNet
→ output projection
→ residual
→ RMSNorm
→ FFN/SwiGLU
→ residual
```

with activations kept on the NPU between stages.

---

# 7. Do not take 30 tokens/s as an established target

A weight file of approximately 5.3 GB read once per token represents:

```text
5.3 × 30 ≈ 159 GB/s
```

And this does not include activations, caches, rereads, or transfers.

On some XDNA2 machines, 30 tokens/s may be close to or beyond the actually exploitable NPU bandwidth.

Before announcing this figure, measure:

* bytes transferred per token;
* number of reads of each matrix;
* effective NPU bandwidth;
* reconfiguration time;
* total time per layer.

A more defensible initial objective would be:

* numerically correct text;
* 100% of layers validated;
* fewer dispatches;
* reproducible improvement over 1.8 tokens/s;
* measured power consumption.

---

# 8. What is actually missing to apply the fix

The public repository does not contain:

* the source of `ggml-xdna.dll`;
* the `.s`/C++ source of the fused kernel;
* the host code that prepares the buffers;
* the `.xclbin`/`.insts` generator;
* CPU/NPU traces;
* the source commit corresponding to binary `18b583a`.

Without these elements, the correction method can be determined, but the actual bug cannot be modified. The executables alone cannot establish whether the fault is in the kernel, DMA, or GGML backend.

The concrete priority is therefore:

1. Publish or recover the exact source of `ggml-xdna.dll` and `f3best`.
2. Add the isolated softmax×V test with identical V.
3. Verify `hkv = hq / 4`.
4. Lock down the layout `[token][kv_head][dim]`.
5. Accumulate softmax and P×V in FP32.
6. Test C alone, then re-enable fusion.
7. Add per-operation traces.
8. Optimize dispatches only after numerical equality.

The most likely correction is a combination of **“GQA mapping + V stride/layout + block-wise softmax reduction”**. That is where I would immediately focus the investigation.

---

# XDNA2 Experimental LLM Runtime

Experimental work on accelerating LLM inference on AMD XDNA2 NPUs through a custom ggml/llama.cpp backend and dedicated AIE kernels.

The repository is intended to provide a reproducible environment for testing and characterizing XDNA2 inference, including decode performance, kernel execution, memory behavior, and heterogeneous CPU/NPU execution.

## Project scope

The current implementation includes:

* Experimental ggml-xdna backend
* XDNA2 NPU execution path
* INT4 GEMV kernels
* INT4 SwiGLU kernels
* FlowKV / attention experiments
* Compiled .xclbin and .insts kernel artifacts
* llama.cpp integration
* Reproducibility scripts
* Reference Qwen3.5 implementation
* Performance and validation experiments

The compiled kernel artifacts included in the repository correspond to the configurations used for the documented experiments.

## Upstream projects and technical references

This work builds upon and references several open-source projects.

### llama.cpp / ggml

Repository:

https://github.com/ggml-org/llama.cpp

llama.cpp provides the inference engine and ggml execution framework used as the host architecture for the experimental XDNA2 backend.

License: MIT

Copyright and licensing remain with the respective upstream authors.

### AMD IRON

Repository:

https://github.com/amd/IRON

IRON provides open-source infrastructure and examples for programming AMD Ryzen AI NPUs.

The project includes NPU-oriented kernels and examples covering operations relevant to LLM inference, including matrix operations, attention-related workloads and other AIE compute primitives.

License: Apache License 2.0.

### MLIR-AIE

Repository:

https://github.com/Xilinx/mlir-aie

MLIR-AIE provides the compiler infrastructure and programming model used for targeting AMD/Xilinx AI Engine architectures.

It includes tooling for generating executable NPU artifacts and instruction streams used by AIE applications.

License: Apache License 2.0 with LLVM exceptions where applicable.

### AMD XDNA

Repository:

https://github.com/amd/xdna-driver

The AMD XDNA open-source project provides architecture and runtime references for XDNA-based accelerators and is an important technical reference for understanding the execution environment targeted by this project.

Refer to the upstream repository for the licenses applicable to individual components.

### Hugging Face Transformers / Qwen3.5

Repository:

https://github.com/huggingface/transformers

Qwen:

https://github.com/QwenLM

The Qwen3.5 reference implementation is used for architectural and numerical comparison with the accelerated implementation.

The corresponding Transformers implementation is distributed under the Apache License 2.0.

Copyright remains with the Qwen Team, Hugging Face, and the respective upstream contributors.

## Architecture

The experimental execution stack can be summarized as:

```text
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
```

## Kernel artifacts

The repository contains compiled kernel configurations used by the experimental runtime.

These include kernels targeting operations such as:

* INT4 GEMV
* INT4 SwiGLU
* FlowKV
* attention-related execution paths

`.xclbin` and `.insts` files are executable artifacts consumed by the XDNA2 execution path.

Different kernel variants correspond to different tensor dimensions and execution configurations required by the model.

Keeping the tested artifacts together with the runtime allows benchmark configurations to remain reproducible.

## Reproducibility

The repository is designed around reproducible measurements rather than isolated peak-performance results.

When comparing configurations, relevant parameters should therefore be recorded, including:

* model
* quantization
* context size
* prompt length
* generated token count
* kernel configuration
* CPU/NPU placement
* runtime version
* thermal conditions
* prefill throughput
* decode throughput

Throughput is expressed in tokens per second (tokens/s).

Prefill and decode measurements should be considered separately because they exercise substantially different computational and memory-access patterns.

## Experimental status

This project is research and experimental software.

The XDNA2 backend and kernels are under active development. Performance characteristics, kernel selection and supported model configurations may therefore change between revisions.

Results should be associated with the exact repository revision and test configuration used to produce them.

## Licensing and attribution

This repository combines original experimental work with interfaces, references and components from open-source projects.

Major upstream projects include:

| Project                   | Upstream                                    | License                                       |
| ------------------------- | ------------------------------------------- | --------------------------------------------- |
| llama.cpp / ggml          | https://github.com/ggml-org/llama.cpp       | MIT                                           |
| AMD IRON                  | https://github.com/amd/IRON                 | Apache-2.0                                    |
| MLIR-AIE                  | https://github.com/Xilinx/mlir-aie          | Apache-2.0 / LLVM exceptions where applicable |
| AMD XDNA                  | https://github.com/amd/xdna-driver          | See upstream component licenses               |
| Hugging Face Transformers | https://github.com/huggingface/transformers | Apache-2.0                                    |
| Qwen                      | https://github.com/QwenLM                   | See individual model/repository license       |

Each upstream component remains subject to its respective copyright and license terms.

Original project-specific code, scripts, experimental integration work and documentation should be considered separately from third-party components and retain the licensing specified by this repository.

## References

1. ggml-org — llama.cpp
   https://github.com/ggml-org/llama.cpp
2. AMD — IRON
   https://github.com/amd/IRON
3. AMD/Xilinx — MLIR-AIE
   https://github.com/Xilinx/mlir-aie
4. AMD — XDNA
   https://github.com/amd/xdna-driver
5. Hugging Face — Transformers
   https://github.com/huggingface/transformers
6. Qwen Team — Qwen
   https://github.com/QwenLM
