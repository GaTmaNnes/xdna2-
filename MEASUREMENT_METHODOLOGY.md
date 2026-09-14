# Measurement Methodology & Evidence Classification

**Purpose:** Formalize how measurements are conducted, labeled, and interpreted in this repository.  
**Rationale:** Following ROCm/FastFlowLM #636 methodology, distinguish between proven facts, valid comparisons, and open hypotheses.

---

## Evidence Labels (Applied Throughout Repo)

All claims in README.md, MANIFEST.md, and reports are tagged with one of these labels:

### [CONFIRMED]

**Meaning:** Published documentation or direct review of authoritative source explicitly states the fact.

**Example:**
> "llama.cpp provides the inference engine." [CONFIRMED]
> — Verified by direct inspection of github.com/ggml-org/llama.cpp

**Threshold:**
- Source must be public and verifiable
- No interpretation or extrapolation needed
- Direct quote or clear logical consequence of source

---

### [CONFIRMED PRINCIPLE]

**Meaning:** A general engineering principle supported by multiple independent authoritative sources.

**Example:**
> "Operator fusion reduces intermediate memory traffic." [CONFIRMED PRINCIPLE]
> — Standard compiler/architecture optimization (MLIR-AIE, TileFuse, AMD Agent)

**Threshold:**
- Principle, not specific fact
- Multiple independent sources or textbooks
- Applies broadly; not limited to one project

---

### [MY MEASUREMENT]

**Meaning:** Observation from own external instrumentation. Reproducible by anyone who runs the same setup.

**Methodology:**
See § Hardware, § Software, § Instrumentation below for full setup and tooling.

**Example:**
> "~4.89 ms NPU wait per dispatch." [MY MEASUREMENT]
> — B1 protocol with pyxrt harness on Strix Point / XDNA2

**Threshold:**
- Own instrumentation or profiling
- Setup and methodology fully disclosed
- Result is reproducible (anyone can run the same test)
- NOT FLM-internal or black-box telemetry

**Important caveat:**
This is **external measurement**, not access to internal runtime state. If a claim sounds like "I have inside information," it's mislabeled.

---

### [VALID IN CONTEXT]

**Meaning:** Upstream publication provides a number, but only within that publication's specific experimental setup.

**Example:**
> "STEEL achieves 22.8× improvement over layer-by-layer execution." [VALID IN CONTEXT]
> — True in STEEL's paper on STEEL's test setup, not proven for FastFlowLM

**Threshold:**
- Source is reputable (peer-reviewed or AMD official)
- Number is reproducible within that source's context
- **NOT generalized** to this project without direct experiment
- Used as reference point for comparison, not as predicted outcome

---

### [PLAUSIBLE]

**Meaning:** Technically coherent interpretation consistent with observations, but lacking direct validation.

**Example:**
> "Kernel numerical correctness might account for the text divergence." [PLAUSIBLE]
> — Consistent with observed symptoms, but not proven without controlled test

**Threshold:**
- Hypothesis is technically sound
- Consistent with available evidence
- Does not contradict known facts
- Requires experiment to validate

---

### [HYPOTHESIS]

**Meaning:** Proposed explanation for observed symptom, requiring controlled experiment.

**Example:**
> "AIE-side DMA stalls may account for the 4.89 ms execution time." [HYPOTHESIS]
> — Reasonable explanation, but needs per-phase timeline to confirm

**Threshold:**
- Testable (not vague)
- Proposed experiment is clear
- Does not assume result

---

### [NOT PROVEN]

**Meaning:** Could not be independently established despite effort.

**Example:**
> "Exact cause of 2362 ms per-request overhead." [NOT PROVEN]
> — Overhead is measured [MY MEASUREMENT], but its source is unidentified

**Threshold:**
- Claim is significant enough to measure
- Measurement was attempted
- Result is indeterminate with available tools
- Not an opinion; an open technical question

---

### [EXTRAPOLATION]

**Meaning:** Used as an external reference point, not as a direct claim about this repo.

**Example:**
> "Striking the Balance reports 38.05 TOPS INT8 on XDNA2." [EXTRAPOLATION]
> — Real fact about that paper; used here to set context, not to predict FastFlowLM performance

**Threshold:**
- Fact is true about the cited source
- Used for comparison or architectural context
- **Explicitly not claimed as outcome** for this project

---

## Hardware & Software Configuration

All [MY MEASUREMENT] entries apply to:

### Hardware

| Component | Specification |
|-----------|---|
| SoC | AMD Ryzen AI 9 365 (Strix Point) |
| NPU Device ID | 0x17F0 |
| RAM | 32 GB DDR5 |
| Storage | NVMe SSD |

### Software

| Component | Version |
|---|---|
| OS | Windows 11 Pro (Build 26200) |
| FLM | v0.9.45 (if applicable) |
| XRT | As bundled with FLM or NPU driver |
| NPU Driver | IpuMcdmDriver v32.0.203.329+ |
| Python | 3.13.14 (for harness scripts) |

**Variability note:** Results will differ with different hardware, driver versions, FLM versions, and thermal conditions.

---

## Instrumentation Tools

### XRT Call Tracing

**Method:** Custom DLL wrapper using Microsoft Detours to intercept `xrt_coreutil.dll` exports.

**Captured:** Entry/exit timestamps via `QueryPerformanceCounter` for:
- `xrtBOAlloc`, `xrtBOMap`, `xrtBOSync`
- `xrtRunExec`, `xrtRunWait`
- Associated functions

**Output:** `xrt_trace.csv` with call count, duration, and argument sizes.

### IOCTL Capture

**Method:** ETW (Event Tracing for Windows) provider for NPU driver, captured via `xperf` / `wpr`.

**Captured:** IOCTL command codes, return values, latencies per transaction.

**Output:** `.etl` files (~50 MB per session), decoded to CSV.

### DMA Timing

**Method:** Derived from XRT wrapper's `xrtBOSync` pairs, cross-referenced with ETW DMA completion.

**Output:** Per-transfer latency, throughput, overlap with compute.

### Power Measurement

**Method:** `hwmonitor` polling at 100 ms intervals via APU power management interface.

**Output:** Power consumption timeline.

### TTFT Decomposition

**Method:** Custom timer in HTTP request path via local proxy (wall-clock `POST` → first token output).

**Output:** Total time to first token, correlated with phase breakdown.

---

## Example: B1 Temporal Decomposition Report

### Protocol

```
3 warmup runs (not measured)
10 measured runs
5 × 4 MiB data BOs (persistent allocation)
Single H2D initialization
NPU execution
D2H synchronization
Forced 4 MiB diagnostic readback
```

### Measured Results

| Phase | Mean | Std Dev | % Cycle |
|-------|------|---------|---------|
| XRT `submit()` | 54.5 µs | 8.2 µs | 0.9% |
| NPU wait | 4,889.8 µs | 23.9 µs | **78.5%** |
| D2H sync | 58.4 µs | 12.1 µs | 0.9% |
| Readout (4 MiB) | 1,222.5 µs | 45.3 µs | 19.6% |
| **Total cycle** | **6,225.2 µs** | **62.8 µs** | **100%** |

### Interpretation

**Claim:** NPU execution dominates the critical path.  
**Evidence:** 78.5% of cycle time is NPU wait, not host overhead.  
**Implication:** Host `submit()` overhead (0.9%) is not the primary optimization target.  
**Next priority:** AIE-side execution/dataflow internal to the 4.89 ms window.

### Caveat

The 1.2 ms diagnostic readback is artificial (forcing full 4 MiB CPU read). In normal operation, only ~72 KB is needed (~20 µs expected, not 1.2 ms).

---

## Validation Levels

### Level 1: Deterministic Execution

✅ **Proven**
- Kernel loads and dispatches without crashing
- Output hashes are reproducible across runs
- State machine reaches expected terminal state

**Example:** F3best kernel reaches state=4 deterministically (README §7).

---

### Level 2: Numerical Correctness

❌ **Not Yet Proven**
- Requires controlled BF16 tensor comparison against CPU reference
- Synthesis test cases (e.g., Q=K→uniform softmax, all-identical V→output=V)

**Next step:** Decomposed kernel test suite (README §2).

---

### Level 3: End-to-End Model Validation

❌ **Not Yet Proven**
- Full Qwen3.5-9B generates coherent text across multiple test cases
- Output matches expected model behavior
- Requires multi-layer integration

**Next step:** Complete pipeline validation (README § 10).

---

### Level 4: Throughput Claim

❌ **Not Yet Proven**
- Measured token/s on complete model
- Reproducible across runs and workloads
- Accounts for all overhead (host, synchronization, memory)

**Current status:** Mechanical 33.9 dispatches/s ≠ 33.9 tokens/s.

---

## Reporting Template

When presenting a measurement in issues or documentation, use this template:

```markdown
**Claim:** [Statement of what was measured]

**Status:** [CONFIRMED | MY MEASUREMENT | VALID IN CONTEXT | PLAUSIBLE | HYPOTHESIS | NOT PROVEN]

**Evidence:**
- [Source 1]
- [Source 2]
- [Methodology if [MY MEASUREMENT]]

**Refinement/Caveats:**
- [Limitations or alternative interpretations]

**Next validation:**
- [Proposed experiment or improvement]
```

---

## Common Mistakes to Avoid

### ❌ Mislabeling Black-Box Output

**Wrong:**
> "The kernel produces correct output." [MY MEASUREMENT]

**Right:**
> "The kernel produces deterministic output matching the frozen witness hash." [MY MEASUREMENT]
> "Whether that output is numerically correct for Qwen3.5-9B requires controlled comparison." [NOT PROVEN]

---

### ❌ Extrapolating [VALID IN CONTEXT] to this Project

**Wrong:**
> "STEEL achieves 22.8×, so FastFlowLM will achieve 22.8×." [VALID IN CONTEXT — MISAPPLIED]

**Right:**
> "STEEL achieves 22.8× improvement on its own test setup, which demonstrates that similar optimization strategies can be effective." [VALID IN CONTEXT]
> "FastFlowLM gains from similar approaches are unknown and require direct experiment." [HYPOTHESIS]

---

### ❌ Unqualified Performance Claims

**Wrong:**
> "33.9 tokens/s achieved"

**Right:**
> "~33.9 mechanical dispatches/s measured on f3best kernel in isolation." [MY MEASUREMENT]
> "This is NOT a validated end-to-end Qwen3.5-9B throughput claim." [NOT PROVEN end-to-end]

---

### ❌ Mixing Measurement Contexts

**Wrong:**
> "The overhead is X because paper Y says Z." 

**Right:**
> "Paper Y reports Z in its context; in our setup, we observe X." [VALID IN CONTEXT] vs. [MY MEASUREMENT]
> "Possible explanations for the difference:" [HYPOTHESIS]

---

## Updating This Document

As new measurements are conducted:

1. **Record the full setup** (hardware, software, tool versions)
2. **Apply appropriate label** from § Evidence Labels
3. **Document the methodology** (tool + protocol)
4. **State caveats** (what this does and doesn't prove)
5. **Propose next validation** (what would strengthen or refute this)

---

**Version:** 1.0  
**Last updated:** September 2026  
**Authors:** GaTmaNnes + community review (see COMPLIANCE.md)
