# Compliance & Attribution Statement

**Document:** COMPLIANCE.md  
**Date:** September 2026  
**Repo:** GaTmaNnes/xdna2-  
**Purpose:** Address licensing, provenance, and attribution requirements per external review

---

## 1. License Status

This repository is published under the **MIT License** (see `LICENSE` file).

### Scope of MIT license (this repo)

Original work covered by MIT:
- Host-side integration code
- Measurement harnesses and scripts
- Reproducibility documentation
- Experimental methodology and reports
- This compliance statement

### Third-party components (not covered by repo's MIT)

| Component | License | Status |
|-----------|---------|--------|
| llama.cpp, ggml, llama.dll, ggml*.dll | MIT | Retain original MIT notice |
| IRON, MLIR-AIE | Apache 2.0 | Subject to Apache 2.0 terms |
| AMD XDNA driver (Windows) | Proprietary | AMD terms apply |
| Qwen3.5-9B | See model license | Model owner's terms |

See `LICENSE` for full third-party attribution.

---

## 2. Redistributed Binary Components

### llama.cpp Binaries (llama-cli.exe, *.dll)

**Status:** Redistributed without MIT notice in binary form.  
**Issue:** MIT license requires inclusion of copyright and license text in distributions.  
**Fix:** Added comprehensive third-party notice in `LICENSE` file.

**MIT Notice (required by llama.cpp/ggml):**
> Copyright (c) 2023–2026 Georgi Gerganov and llama.cpp contributors.
> Licensed under the MIT License (see LICENSE for full text).

**Upstream source:**  
Commit 18b583a in the lineage of ggml-xdna backend (PrismML-Eng/llama.cpp fork → bong-water-water-bong/llama.cpp).

---

## 3. NPU Kernel Artifacts (.xclbin, .insts)

### Compilation Origin

| Artifact | Tool | License | Compiled by |
|----------|------|---------|-------------|
| `.xclbin` | MLIR-AIE (aiecc) | Apache 2.0 | User (local machine) |
| `.insts` / `.bin` | AMD IRON emitter | Apache 2.0 | User (local machine) |

### Redistribution Terms

Kernel artifacts were compiled locally using AMD's open-source MLIR-AIE / IRON toolchain.

**Compliance question (for AMD):**  
Whether public redistribution of such artifacts requires:
- Source kernel code disclosure
- Compilation record or reproducibility attestation
- XRT/driver version compatibility notice
- Any additional license/notice

**Current approach:**
- All source emitters are provided (`sources/f3best_emit_nokv.py`)
- Compilation script is provided (`sources/compile_f3best_9b_s128_rr.py`)
- Commit provenance is documented (`reproductible/MANIFEST.md`)
- SHA256 hashes enable verification (`reproductible/SHA256SUMS.txt`)

---

## 4. Provenance and Reproducibility

### Source Attribution Chain

Every kernel artifact is traceable to its source:

```
sources/f3best_emit_nokv.py
  ↓ (copied from)
E:\tmp\iron_am100 (commit 42178241db02648c3d54438c5f58b4b265809662, 13/08/2026)
  ↓ (emits)
temoin/decode_layer_f3best_9b_s128_rr.mlir
  ↓ (compiled via aiecc)
reproductible/MANIFEST.md § 2
  ↓ (frozen witness)
temoin/decode_layer_f3best_9b_s128_rr.xclbin + .bin (SHA256 verified)
```

### Measurement Methodology

All claimed performance numbers in README.md and reports are labeled by evidence type:

| Label | Meaning |
|-------|---------|
| [CONFIRMED] | Published documentation or direct review of source |
| [MY MEASUREMENT] | Own external instrumentation; reproducible by others with same hardware/software config |
| [VALID IN CONTEXT] | Upstream publication on same hardware; not a prediction for this project |
| [PLAUSIBLE] | Technically coherent but not independently validated |
| [HYPOTHESIS] | Proposed explanation requiring further experiment |

**Critical distinction:**  
- `~33.9 tok/s` result is labeled **[MECHANICAL / NON-SEMANTIC]** (not validated end-to-end text generation)
- End-to-end Qwen3.5-9B semantics remain **[NOT VALIDATED]** per README section 10

---

## 5. Validation Status

### What is Proven

- ✅ Frozen kernel executes deterministically on XDNA2
- ✅ NPU dispatch reaches state=4 with reproducible ~4.89 ms latency
- ✅ Output hashes match across runs (B1 witness protocol)
- ✅ Kernel ABI and BO dependencies validated

### What Remains Open

- ❌ Kernel numerical correctness (requires BF16 reference comparison)
- ❌ End-to-end Qwen3.5-9B semantic text generation
- ❌ Multi-layer pipeline validation
- ❌ Full-model throughput claim (pending integration work)

**Implication:**  
Claims in README do not exceed these validation boundaries. Any claim of achieved throughput is explicitly qualified.

---

## 6. Attribution to Technical Reviewers

### 1bit-MONSTER Cross-Validation

The work described in this repository was reviewed and cross-validated by the 1bit-MONSTER team:

**Reference:**  
- 1bit-MONSTER/1bit-MONSTER : `research/ws13-gatman45-xdna-collab/`
- Collaboration record: `research/ws13-gatman45-xdna-collab/1bit-monster-collab-record.md`
- Discord thread: 1545394554956423269 (1bit-MONSTER guild)

**Contribution:**  
Independent technical review, verification of kernel execution pathway, and feedback on experimental design.

**Courtesy attribution:** While not required by MIT license, credit should be given where external technical review significantly influenced the work.

---

## 7. Compliance Checklist

- ✅ **License declaration:** MIT (LICENSE file)
- ✅ **Third-party notice:** llama.cpp/ggml MIT notice included
- ✅ **Source provenance:** Exact commits documented in MANIFEST.md
- ✅ **Reproducibility:** Frozen artifacts + rebuild scripts + SHA256 verification
- ✅ **Measurement labels:** All claims tagged by evidence type
- ✅ **Validation scope:** Distinctions between proven/open clearly marked
- ✅ **Attribution:** 1bit-MONSTER review documented
- ⚠️ **Kernel redistribution terms:** Awaiting AMD clarification (see Section 3)

---

## 8. Recommended Next Steps

1. **AMD licensing review:** Confirm whether `.xclbin`/`.insts` redistribution requires additional notice or terms
2. **ggml-xdna source publication:** Host-side backend code should be published with appropriate license
3. **End-to-end validation:** Complete semantic tests before claiming model throughput figures
4. **Community contribution guidelines:** Document how external collaborators should be credited

---

## Contact & Questions

For questions on licensing, provenance, or attribution:

- **Repository:** GaTmaNnes/xdna2- (GitHub issues)
- **Upstream:** llama.cpp, IRON, MLIR-AIE projects (see LICENSE)
- **AMD:** XDNA/IRON licensing terms (github.com/amd/IRON, github.com/amd/xdna-driver)

---

**Last reviewed:** September 8–9, 2026  
**Status:** Provisional — pending AMD clarification on kernel artifact redistribution terms
