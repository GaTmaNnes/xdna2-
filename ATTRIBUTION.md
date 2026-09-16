# Attribution and Acknowledgments

## Project Information

**Project**: XDNA2 Experimental LLM Runtime  
**Author**: GaTmaNnes  
**Repository**: https://github.com/GaTmaNnes/xdna2-  
**Date**: 16 September 2026  
**License**: See LICENSE file (non-commercial free / commercial restricted)

---

## Core Contributors & Original Work

### GaTmaNnes
- **Role**: Author, experimental implementation, XDNA2 validation, end-to-end execution
- **Contributions**:
  - Direct GGUF → XDNA2 NPU execution path
  - T6 end-to-end run validation (8 prefill + 8 autoregressive generation)
  - Transport integrity validation (6/6 SHA)
  - Differential validation (blk.7, blk.15, blk.23, blk.31)
  - Resident 7-worker RPC architecture
  - Model-driven compiler conceptualization
  - Performance decomposition and measurement methodology
- **Contact**: [Your contact information]

---

## Upstream Projects & Libraries

### llama.cpp / ggml
- **Repository**: https://github.com/ggml-org/llama.cpp
- **License**: MIT
- **Copyright**: ggml-org contributors
- **Role**: Core inference engine, tensor computation framework, GGML backend
- **Attribution**: Required in source and binary distributions

### AMD IRON (XDNA Infrastructure)
- **Repository**: https://github.com/amd/IRON
- **License**: Apache License 2.0
- **Copyright**: AMD
- **Role**: 
  - AIE kernel infrastructure and programming model
  - Matrix operation examples (GEMV, GEMM, SwiGLU)
  - FlowKV attention reference
  - Foundational XDNA2 examples
- **Attribution**: Required

### MLIR-AIE (Compiler Infrastructure)
- **Repository**: https://github.com/Xilinx/mlir-aie
- **License**: Apache License 2.0 with LLVM exceptions where applicable
- **Copyright**: Xilinx, AMD, LLVM Project
- **Role**:
  - Compiler infrastructure for AIE code generation
  - XCLBIN artifact generation
  - Instruction stream generation
  - Tile and DMA scheduling
- **Attribution**: Required

### AMD XDNA Driver
- **Repository**: https://github.com/amd/xdna-driver
- **License**: See individual component licenses in upstream repository
- **Copyright**: AMD
- **Role**:
  - XDNA2 NPU runtime architecture reference
  - Command submission, mailbox communication, ERT scheduling
  - Linux driver implementation (referenced for architecture understanding)
- **Attribution**: Required where components are derived or extended

### Hugging Face Transformers
- **Repository**: https://github.com/huggingface/transformers
- **License**: Apache License 2.0
- **Copyright**: Hugging Face Team
- **Role**:
  - Model architecture reference (Qwen3.5)
  - Transformer building blocks
  - Tokenization and inference baseline
- **Attribution**: Required for model-specific code

### Qwen / QwenLM
- **Repository**: https://github.com/QwenLM
- **License**: Apache License 2.0 (or per-model; verify independently)
- **Copyright**: Qwen Team
- **Role**:
  - Qwen3.5-9B model implementation
  - Reference quantization and inference
  - Test and validation model
- **Attribution**: Required; model weights may have separate terms

---

## Research Papers and Measurement References

### Foundational XDNA2 Performance

| Paper | Authors | Date | Reference | Key Finding |
|-------|---------|------|-----------|-------------|
| **Striking the Balance** | — | 2024 | arXiv:2512.13282 | 38.05 TOPS INT8 / 14.71 TOPS BF16 on XDNA2 |
| **TileFuse** | — | 2026 | arXiv:2606.11357 | +121.6% GEMM / +281% GEMV; 2.0× prefill latency, 64%+ energy reduction |
| **AMD Agent** | Li, Wang, Zhang, Bayliss (Cornell/AMD) | 2026 | arXiv:2606.07586 | 2.2× prefill / 4.0× decode on XDNA2; 15→3 dispatches/layer via kernel fusion; persistent XRT context; zero-copy BO recycling |
| **STEEL** | — | 2026 | arXiv:2607.09385 | 22.8× improvement over layer-by-layer; 9.6× latency vs. XDNA1 baseline |
| **MLIR-AIR** | — | 2025 | arXiv:2510.14871 | Spatial mapping, memory locality, scheduling, fusion as first-order factors |
| **Zen-Attention** | — | 2025 | arXiv:2508.17593 | Attention folding; up to 4× attention-block latency / 32% end-to-end |

### Independent Performance Characterization

| Source | Author | Date | Reference | Finding |
|--------|--------|------|-----------|---------|
| **XDNA2 Peak Performance** | Estévez | May 2026 | https://destevez.net/2026/05/getting-peak-tops-on-a-ryzen-ai-7-350-npu/ | 56 TOPS INT8 peak with custom mlir-aie kernels |
| **OllamaAMDNPU** | BrandedTamarasu | 2026 | https://github.com/BrandedTamarasu-glitch/OllamaAMDNPU | Dispatch batching alone (TILE_M 2048→14336) without kernel fusion: zero throughput gain; compute-bound at 0.6% utilization |

### FastFlowLM / XDNA2 Performance Analysis

| Issue | Repository | Author | Date | Title | Key Content |
|-------|-----------|--------|------|-------|-------------|
| **#636** | ROCm/FastFlowLM | [External contributor] | July 23, 2026 | "FastFlowLM / XDNA2 — Performance Observations & Validation Roadmap" | Comprehensive analysis of host/runtime overhead, dispatch granularity, context initialization, persistent runtime state, kernel fusion, DMA overlap. Instrumental in validating AMD Agent results and quantifying optimization targets. Critical reference for Priority 2/P3 (persistent context, kernel fusion). |

---

## Validation and Reproducibility Contributors

### Experimental Methodology
- **Instrumentation**: XRT call tracing, IOCTL capture (ETW), DMA timing, power measurement, TTFT decomposition
- **Configuration**: AMD Ryzen AI 9 365 (Strix Point), Windows 11, FLM v0.9.45
- **Trace Data**: 205K lines XRT trace, 50 MB ETW logs per session
- **Reproducibility**: Methodology documented; rebuild instructions available upon request

### Quality Assurance
- Differential validation protocols (bit-exact, hash-based, statistical)
- Cross-layer validation (blk.7, blk.15, blk.23, blk.31)
- Transport integrity checks (SHA256)
- State evolution tracking
- Evidence levels classification (PROVEN, MEASURED, SUPPORTED HYPOTHESIS, TARGET, OPEN)

---

## Secondary Sources and References

### Documentation and Standards
- AMD XDNA Architecture Reference
- XCLBIN/PDI/ERT Specification
- XRT (Xilinx Runtime) API Reference
- Windows ETW (Event Tracing for Windows) Documentation

### Open-Source Communities
- ggml-org (tensor computation, backend integration)
- AMD Open Source (IRON, XDNA driver, AIE infrastructure)
- LLVM Project (compiler infrastructure)
- Hugging Face (model implementations, tokenization)

---

## Citation and Referencing

### For Academic / Research Use

**BibTeX Format (Repository)**:
```bibtex
@misc{GaTmaNnes2026xdna2,
  author = {GaTmaNnes},
  title = {XDNA2 Experimental LLM Runtime: End-to-End Autoregressive Inference on AMD XDNA2 NPU},
  year = {2026},
  month = {September},
  url = {https://github.com/GaTmaNnes/xdna2-},
  note = {Commit: c31b0432199a6bfe4e7027e373ce1c58c8457a06}
}
```

**If citing specific results**:
- Include test configuration (hardware, software versions, driver version)
- Reference the exact commit hash
- Cite this ATTRIBUTION.md file
- Acknowledge all upstream projects
- Reference FastFlowLM #636 for related performance analysis

### For Blog Posts / Articles

```markdown
XDNA2 Experimental LLM Runtime by GaTmaNnes
https://github.com/GaTmaNnes/xdna2-
(See LICENSE for restrictions; non-commercial use free, commercial use requires agreement)

Built on: llama.cpp, AMD IRON, MLIR-AIE, Qwen, Hugging Face Transformers
```

---

## Known Contributors to Related Work

### AMD / Xilinx Teams
- IRON library developers
- MLIR-AIE compiler infrastructure
- AMD Agent implementation team (Li, Wang, Zhang, Bayliss)
- XDNA Driver and runtime teams

### Academic Research
- TileFuse authors
- STEEL authors
- MLIR-AIR authors
- Zen-Attention authors

### Open Source Community
- ggml-org maintainers
- Hugging Face Transformers team
- Qwen model development team
- OllamaAMDNPU contributors (BrandedTamarasu)
- Estévez (XDNA2 performance characterization)

---

## Missing or Incomplete Information

The following aspects require independent verification or additional attribution:

1. **Qwen3.5-9B Model License**: Verify per-model terms at https://github.com/QwenLM
2. **AMD Component Licenses**: Confirm each upstream license version in use
3. **Kernel Source Attribution**: Original XCLBIN/instruction source (not included in this repository)
4. **ggml-xdna Backend**: Source code location and version
5. **Test Results**: Measurements are configuration-specific; results will vary with different hardware/driver/FLM versions

---

## License Compliance Checklist

- [ ] Apache-2.0 components: License text and copyright notice included
- [ ] MIT components: Attribution in documentation
- [ ] Third-party model weights: Obtained with proper license verification
- [ ] Commercial use: Explicit Commercial License Agreement obtained (if applicable)
- [ ] Research publications: All sources cited, upstream attribution acknowledged
- [ ] Distribution: LICENSE and ATTRIBUTION.md included in all copies

---

## Contact for Attribution Issues or Corrections

If you believe any attribution is incorrect, incomplete, or missing:
- Create an issue on GitHub: https://github.com/GaTmaNnes/xdna2-/issues
- Or contact: [Your contact information]

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-09-16 | 1.0 | Initial comprehensive attribution and acknowledgments |

---

**Last Updated**: 16 September 2026
