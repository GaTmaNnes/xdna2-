# Upstream Licenses and Third-Party Components

## Overview

This project integrates or references components from several open-source projects and research implementations.

**Critical**: Each upstream component remains subject to its original license. **Compliance is mandatory.**

For your own use of this project:
1. **Non-commercial use**: Verify you comply with all upstream licenses
2. **Commercial use**: Obtain explicit Commercial License Agreement AND verify upstream license compatibility with your commercial application

---

## Components — Detailed License Information

### 1. llama.cpp / ggml

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/ggml-org/llama.cpp |
| **License** | MIT |
| **Copyright** | ggml-org contributors |
| **Used For** | Core inference engine, tensor computation framework, GGML backend |
| **Integration** | Backend for this project; linked/called at runtime |

**License Text (MIT)**:
```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

**Compliance Requirements**:
- ✅ Include MIT license notice in binary/source distributions
- ✅ Retain copyright notice
- ✅ No additional restrictions

**Commercial Implication**: MIT places no restrictions on commercial use.

---

### 2. AMD IRON (XDNA Infrastructure)

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/amd/IRON |
| **License** | Apache License 2.0 |
| **Copyright** | AMD |
| **Used For** | AIE kernel infrastructure, XDNA2 examples, matrix operations (GEMV/GEMM/SwiGLU), FlowKV reference |

**License Text (Apache 2.0 Summary)**:
```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**Compliance Requirements**:
- ✅ Include Apache 2.0 license text
- ✅ Retain copyright and license notices
- ✅ Include NOTICE file (if provided upstream)
- ✅ Document modifications to original code
- ✅ Provide source code access (if source was provided)
- ✅ State significant changes

**Commercial Implication**: Apache 2.0 permits commercial use, but requires compliance with all terms (including patent grants and source availability for derivative works).

---

### 3. MLIR-AIE (Compiler Infrastructure)

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/Xilinx/mlir-aie |
| **License** | Apache License 2.0 with LLVM exceptions where applicable |
| **Copyright** | Xilinx, AMD, LLVM Project |
| **Used For** | Compiler infrastructure, XCLBIN generation, instruction stream generation, tile/DMA scheduling |

**License Text (Apache 2.0 + LLVM)**:
```
This project is licensed under the Apache License v2.0.

Where LLVM-derived code is present, it is subject to the Apache 2.0 license
and the LLVM License (Apache 2.0 with LLVM exceptions).

For details, see: https://github.com/Xilinx/mlir-aie/blob/main/LICENSE
```

**Compliance Requirements**:
- ✅ Include Apache 2.0 license text
- ✅ Identify code derived from LLVM Project
- ✅ Include LLVM license notices
- ✅ Document modifications

**Commercial Implication**: Same as Apache 2.0 (permits commercial use with compliance).

---

### 4. AMD XDNA Driver

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/amd/xdna-driver |
| **License** | See individual component licenses (mixed: GPL-2.0, Apache-2.0, MIT) |
| **Copyright** | AMD |
| **Used For** | XDNA2 runtime architecture reference, command submission, ERT scheduling documentation |

**License Note**: The XDNA driver contains multiple components under different licenses (firmware, drivers, user-space utilities). **Consult the upstream repository's LICENSE file for details.**

**Compliance Requirements**:
- ✅ For **reference/documentation only**: No special license compliance needed
- ⚠️ If you **derive or port** code: Check which component's license applies
- ⚠️ If GPL-2.0 components are used: Derived work must be GPL-2.0 compatible

**Commercial Implication**: Depends on which driver components are used. GPL-2.0 components require source disclosure for commercial products.

**Recommendation**: Use this repository only for architecture reference; avoid directly porting driver code without explicit license verification.

---

### 5. Hugging Face Transformers

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/huggingface/transformers |
| **License** | Apache License 2.0 |
| **Copyright** | Hugging Face Team |
| **Used For** | Model architecture reference (Qwen3.5), transformer building blocks, tokenization, inference baseline |

**License Text**: Apache 2.0 (see https://github.com/huggingface/transformers/blob/main/LICENSE)

**Compliance Requirements**:
- ✅ Include Apache 2.0 license notice
- ✅ Retain copyright notices
- ✅ Include NOTICE file

**Commercial Implication**: Apache 2.0 (permits commercial use with compliance).

---

### 6. Qwen / QwenLM

| Property | Value |
|----------|-------|
| **Repository** | https://github.com/QwenLM |
| **License** | Apache License 2.0 (or per-model; verify independently) |
| **Copyright** | Qwen Team |
| **Used For** | Qwen3.5-9B model, reference quantization, test/validation baseline |

**License Note**: Model weights and code may have separate license terms. **Verify at**: https://huggingface.co/Qwen/Qwen3.5-9B

**Compliance Requirements**:
- ✅ Verify model-specific license at Hugging Face model card
- ✅ Comply with model weight restrictions (if any)
- ✅ Acknowledge model source in documentation
- ⚠️ Some model variants may restrict commercial use (check model card)

**Commercial Implication**: Generally permissive, but **verify per-model**. Some Qwen variants restrict commercial use without explicit licensing.

**⚠️ Critical for Commercial Use**: Before commercializing any product using Qwen3.5 weights:
1. Check the model card: https://huggingface.co/Qwen/Qwen3.5-9B
2. Contact Qwen Team if commercial use is unclear
3. Verify compatibility with your jurisdiction's AI/export regulations

---

## Composite License Matrix

| Component | License | Commercial Use | Patent Grant | Source Disclosure | Can Modify |
|-----------|---------|-----------------|---------------|-------------------|-----------|
| llama.cpp | MIT | ✅ Permitted | No explicit | Not required | ✅ Yes |
| AMD IRON | Apache-2.0 | ✅ Permitted | ✅ Yes | For derivatives | ✅ Yes |
| MLIR-AIE | Apache-2.0 + LLVM | ✅ Permitted | ✅ Yes | For derivatives | ✅ Yes |
| XDNA Driver | Mixed (see upstream) | ⚠️ Verify | ⚠️ Depends | ⚠️ Depends | ⚠️ Depends |
| Transformers | Apache-2.0 | ✅ Permitted | ✅ Yes | For derivatives | ✅ Yes |
| Qwen | Apache-2.0 (typically) | ⚠️ Verify per-model | ✅ Yes | For derivatives | ✅ Yes |

---

## Practical Compliance Checklist

### For Non-Commercial Use (Personal/Research/Educational)

- [ ] Retain all upstream copyright and license notices
- [ ] Include LICENSE file from this repository
- [ ] Include ATTRIBUTION.md with all upstream acknowledgments
- [ ] If publishing research: cite upstream sources
- [ ] For Qwen model: acknowledge Qwen Team and comply with model card terms

### For Commercial Use (SaaS / Product / Enterprise)

- [ ] Obtain explicit **Commercial License Agreement** from GaTmaNnes (see LICENSE)
- [ ] For each upstream component:
  - [ ] **Apache-2.0 (IRON, MLIR-AIE, Transformers, Qwen)**: Verify source availability terms; if you modify them, you may need to provide source to customers
  - [ ] **MIT (llama.cpp)**: No additional requirements beyond copyright notice
  - [ ] **XDNA Driver (mixed)**: Verify which components you use and their individual terms; GPL-2.0 components require special attention
- [ ] **Qwen model weights**: Contact Qwen Team if you plan to monetize models based on their weights
- [ ] **Patent considerations**: Apache-2.0 includes express patent grant (your use case is covered), but verify your business model complies
- [ ] Include all licenses in your distribution or product documentation
- [ ] Disclose source modifications for Apache-2.0 derivatives (if required by your commercial model)

### For Research Publications

- [ ] Cite upstream papers (TileFuse, AMD Agent, STEEL, MLIR-AIR, Zen-Attention — see ATTRIBUTION.md)
- [ ] Cite repository with commit hash: `https://github.com/GaTmaNnes/xdna2- (commit: <hash>)`
- [ ] Acknowledge all third-party components
- [ ] Disclose measurement configuration (hardware, software versions)
- [ ] Reproduce methodology to enable peer verification

---

## Conflict Resolution

If upstream licenses conflict:

1. **Apache-2.0 vs. MIT**: Apache-2.0 is compatible with MIT. Use Apache-2.0 terms as the binding requirement.
2. **Apache-2.0 vs. GPL-2.0**: Incompatible. Avoid incorporating GPL-2.0 components unless necessary (e.g., XDNA driver). If used, derivative work must be GPL-2.0 compatible.
3. **Custom License (this project) vs. Upstream**: Upstream licenses always apply to their own code. The custom license applies only to GaTmaNnes' original work. When combining, all licenses bind simultaneously.

**Safe Approach**: If using components with conflicting licenses, consult with legal/licensing counsel before commercial distribution.

---

## Verification and Updates

**Last Verified**: 16 September 2026

**If upstream licenses change**, this document may become outdated. **Always check the upstream repository for the current license.**

```bash
# Quick check: Visit these URLs for the latest license terms
- https://github.com/ggml-org/llama.cpp/blob/master/LICENSE
- https://github.com/amd/IRON/blob/main/LICENSE
- https://github.com/Xilinx/mlir-aie/blob/main/LICENSE
- https://github.com/amd/xdna-driver/blob/main/LICENSE
- https://github.com/huggingface/transformers/blob/main/LICENSE
- https://huggingface.co/Qwen/Qwen3.5-9B (model card)
```

---

## Questions or Corrections

- **License clarification**: Consult the upstream repository's LICENSE file directly
- **This document is wrong/outdated**: File an issue at https://github.com/GaTmaNnes/xdna2-/issues
- **Commercial licensing**: Contact the copyright holder (GaTmaNnes) — see LICENSE

---

**Version**: 1.0  
**Date**: 16 September 2026
