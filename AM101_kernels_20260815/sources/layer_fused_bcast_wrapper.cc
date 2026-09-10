// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Hand-asm broadcast GEMV wrapper for f3best chunk-based decode layer.
//
// Replaces the C++ template implementations (_gate_up_bcast_chunk / _down_bcast_chunk)
// by delegating per-chunk GEMV to the proven hand-asm kernel
// `layer_fused_gemv_handasm_tile_bf16` (bcast_gemv.s, KC=256 variant).
//
// The hand-asm kernel expects 3 args (p0=weights, p1=activation, p2=output) and
// produces 32 bf16 partial results. This wrapper handles cross-chunk accumulation
// in float32 for numerical stability, matching the existing C++ template behavior.
//
// Integration (changes needed in layer_fused_f3best.cc):
//   - Remove `static` from: lf_bc_partial, lf_down_bc_partial, lf_left_buf, lf_right_buf
//     (so the extern declarations below resolve at link time).
//   - The _gate_up_bcast_chunk and _down_bcast_chunk template instantiations in
//     layer_fused_gate_up_bcast_bf16 / layer_fused_down_bcast_bf16 are replaced
//     by the functions in this file (same signatures — MLIR sees no difference).
//
// Hand-asm build (bcast_gemv.s → KC=256):
//   - Compile the hand-asm .s with clang --target=aie2p-none-unknown-elf -c
//   - Link the resulting .o with this wrapper .o and the rest of layer_fused_relay.o
//   - KC=256 constants in the .s: see the constants analysis section below.

#define NOCPP
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Statics shared with layer_fused_f3best.cc ──────────────────────────────
// These are defined (non-static) in layer_fused_f3best.cc after removing the
// `static` qualifier. The linker resolves them within the same tile .o.
extern "C" {
extern float     lf_bc_partial[32];
extern float     lf_down_bc_partial[32];
extern float     lf_qkv_bc_partial[32];
extern bfloat16  lf_left_buf[];
extern bfloat16  lf_right_buf[];
}

// lf_silu_buf is lf_left_buf (same memory, re-used after gate).
#define lf_silu_buf lf_left_buf

// ── Hand-asm kernel declaration ────────────────────────────────────────────
// Signature: (weights: packed-int4 + scales, activation: KC bf16, output: 32 bf16)
// KC = 256 (EMBED_DIM / NCHUNK for gate/up; same for down via INTER_DIM_PER_COL/4)
extern "C" void layer_fused_gemv_bcast_kc256_bf16(
    const uint8_t *__restrict w,
    const bfloat16 *__restrict x,
    bfloat16 *__restrict out);

// ════════════════════════════════════════════════════════════════════════════
// Gate/Up broadcast wrapper  (replaces _gate_up_bcast_chunk<32, G, NCHUNK>)
// ════════════════════════════════════════════════════════════════════════════
//
// j encodes (block, chunk):  block = j / NCHUNK,  chunk = j % NCHUNK.
// MLIR calls this once per chunk; cross-chunk accumulation uses lf_bc_partial[].
// On the last chunk, flushes to lf_left_buf (phase=0, gate) or lf_right_buf (phase=1, up).
//
// Weight layout (per chunk, 4608 bytes):
//   [0 .. KC*N/2-1]          packed int4 weights, column-major
//   [KC*N/2 .. end]          bf16 scales, KC/G groups of N=32 scales each
// Activation: b points to the FULL 2048-element activation; chunk offset applied here.

extern "C"
void layer_fused_gate_up_bcast_bf16(
        uint32_t j, uint32_t /*unused*/,
        const uint8_t *a, const bfloat16 *b, int phase)
{
#ifdef STUB_FFN
    (void)j; (void)a; (void)b; (void)phase;
    return;
#else
    constexpr uint32_t N      = 32;
    constexpr uint32_t NCHUNK = 8;   // EMBED_DIM / KC  =  2048 / 256
    constexpr uint32_t KC     = 256;  // columns per chunk

    const uint32_t chunk = j % NCHUNK;
    const uint32_t block = j / NCHUNK;

    // The hand-asm sets crrnd internally, but set it here too so the wrapper's
    // own float32 accumulation (aie::add, aie::store_v) uses consistent rounding.
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Temp buffer for hand-asm output (32 bf16 values → 64 bytes, L1-aligned).
    alignas(64) bfloat16 partial_bf16[N];

    // ── Per-chunk GEMV via hand-asm ──
    // a: chunk-positioned weight buffer (MLIR orchestrator positions a correctly).
    // b + chunk*KC: this chunk's 256-element activation slice.
    layer_fused_gemv_bcast_kc256_bf16(a, b + chunk * KC, partial_bf16);

    // ── Convert bf16 partial → float32 for cross-chunk accumulation ──
    aie::accum<accfloat, N> tmp;
    tmp.from_vector(aie::load_v<N>(partial_bf16));          // bf16 → accfloat
    aie::vector<float, N> cur = tmp.template to_vector<float>();  // accfloat → float32

    // Accumulate with previous chunks (first chunk: no prior partial).
    if (chunk != 0)
        cur = aie::add(cur, aie::load_v<N>(lf_bc_partial));

    // Last chunk: flush to destination. Else: store float32 partial.
    if (chunk == NCHUNK - 1) {
        bfloat16 *dest = (phase == 0) ? lf_left_buf : lf_right_buf;
        aie::accum<accfloat, N> facc;
        facc.from_vector(cur);                               // float32 → accfloat
        aie::store_v(dest + block * N, facc.template to_vector<bfloat16>());
    } else {
        aie::store_v(lf_bc_partial, cur);
    }
#endif  // STUB_FFN
}


// ════════════════════════════════════════════════════════════════════════════
// Down broadcast wrapper  (replaces _down_bcast_chunk<32, G, NCHUNK>)
// ════════════════════════════════════════════════════════════════════════════
//
// Activation comes from the static lf_silu_buf (written by silu*mul upstream),
// NOT from a function argument. The hand-asm treats it like any other activation.
// Output goes to c_out (the output buffer passed by MLIR).

extern "C"
void layer_fused_down_bcast_bf16(
        uint32_t j, uint32_t /*unused*/,
        const uint8_t *a, bfloat16 *c_out)
{
#ifdef STUB_FFN
    (void)j; (void)a;
    for (int i = 0; i < 32; i++) c_out[i] = (bfloat16)0;
    return;
#else
    constexpr uint32_t N      = 32;
    constexpr uint32_t NCHUNK = 4;    // INTER_DIM_PER_COL / KC  =  1024 / 256
    constexpr uint32_t KC     = 256;   // columns per chunk

    const uint32_t chunk = j % NCHUNK;
    const uint32_t block = j / NCHUNK;

    alignas(64) bfloat16 partial_bf16[N];

    // Activation from static silu buffer (NOT from function arg).
    layer_fused_gemv_bcast_kc256_bf16(a, lf_silu_buf + chunk * KC, partial_bf16);

    aie::accum<accfloat, N> tmp;
    tmp.from_vector(aie::load_v<N>(partial_bf16));
    aie::vector<float, N> cur = tmp.template to_vector<float>();

    if (chunk != 0)
        cur = aie::add(cur, aie::load_v<N>(lf_down_bc_partial));

    if (chunk == NCHUNK - 1) {
        aie::accum<accfloat, N> facc;
        facc.from_vector(cur);
        aie::store_v(c_out + block * N, facc.template to_vector<bfloat16>());
    } else {
        aie::store_v(lf_down_bc_partial, cur);
    }
#endif  // STUB_FFN
}


// ════════════════════════════════════════════════════════════════════════════
// QKV broadcast wrapper  (replaces _qkv_bcast_chunk<32, G, NCHUNK>)
// ════════════════════════════════════════════════════════════════════════════
//
// Same K-streaming pattern as gate/up, but output goes to c_out (passed by MLIR),
// not to static buffers. Called by both layer_fused_qkv_bcast_bf16 (Q-proj)
// and layer_fused_oproj_bcast_bf16 (O-proj) — same logic, different output buffer.

extern "C"
void layer_fused_qkv_bcast_bf16(
        uint32_t j, uint32_t /*unused*/,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c_out)
{
    constexpr uint32_t N      = 32;
    constexpr uint32_t NCHUNK = 8;   // EMBED_DIM / KC  =  2048 / 256
    constexpr uint32_t KC     = 256;  // columns per chunk

    const uint32_t chunk = j % NCHUNK;
    const uint32_t block = j / NCHUNK;

    alignas(64) bfloat16 partial_bf16[N];
    layer_fused_gemv_bcast_kc256_bf16(a, b + chunk * KC, partial_bf16);

    aie::accum<accfloat, N> tmp;
    tmp.from_vector(aie::load_v<N>(partial_bf16));
    aie::vector<float, N> cur = tmp.template to_vector<float>();

    if (chunk != 0)
        cur = aie::add(cur, aie::load_v<N>(lf_qkv_bc_partial));

    if (chunk == NCHUNK - 1) {
        aie::accum<accfloat, N> facc;
        facc.from_vector(cur);
        aie::store_v(c_out + block * N, facc.template to_vector<bfloat16>());
    } else {
        aie::store_v(lf_qkv_bc_partial, cur);
    }
}

// O-proj bcast wrapper — identical interface to Q (same chunk logic).
extern "C"
void layer_fused_oproj_bcast_bf16(
        uint32_t j, uint32_t /*unused*/,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c_out)
{
    // N=32, NCHUNK=8, KC=256 — same as Q
    constexpr uint32_t N      = 32;
    constexpr uint32_t NCHUNK = 8;
    constexpr uint32_t KC     = 256;

    const uint32_t chunk = j % NCHUNK;
    const uint32_t block = j / NCHUNK;

    alignas(64) bfloat16 partial_bf16[N];
    layer_fused_gemv_bcast_kc256_bf16(a, b + chunk * KC, partial_bf16);

    aie::accum<accfloat, N> tmp;
    tmp.from_vector(aie::load_v<N>(partial_bf16));
    aie::vector<float, N> cur = tmp.template to_vector<float>();

    if (chunk != 0)
        cur = aie::add(cur, aie::load_v<N>(lf_qkv_bc_partial));  // reuse Q's partial buffer for O

    if (chunk == NCHUNK - 1) {
        aie::accum<accfloat, N> facc;
        facc.from_vector(cur);
        aie::store_v(c_out + block * N, facc.template to_vector<bfloat16>());
    } else {
        aie::store_v(lf_qkv_bc_partial, cur);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// HAND-ASM CONSTANTS ANALYSIS  (bcast_gemv.s → KC=256)
// ════════════════════════════════════════════════════════════════════════════
//
// The proven hand-asm kernel (bcast_gemv.s) currently processes K=2048 columns.
// To adapt it for chunked KC=256, exactly 2 constants must change:
//
//   LINE  CURRENT          NEW (KC=256)    RATIONALE
//   ──────────────────────────────────────────────────────────────────────
//   9     movxm r1, #32768  movxm r1, #4096  K*N/2 = total packed weight bytes
//                                            KC=256:  256*32/2 = 4096
//                                            (was:   2048*32/2 = 32768)
//
//   10    movxm dj0, #32768 movxm dj0, #4096 Same constant — offset from
//                                            weight base to scales section.
//                                            Used in epilogue: vldb x0, [p3, dj0]
//                                            loads the tail scale group.
//
//   CONSTANTS THAT STAY UNCHANGED:
//
//   Line 8:  r0  = 19201    — dead store; overwritten at line 15 (mova r0, #0).
//   Line 19: r2  = 16       — inner loop count: G/2 = 32/2 = 16. Unchanged
//                              because GROUP_SIZE and N don't change.
//   Line 20: r3  = 60       — accumulator register config (sub.f shift value).
//   Line 20: r4  = 828      — accumulator register config (vmac.f post-shift).
//   Line 21: r5  = 512      — outer loop step: 32 columns × 16 bytes/row.
//                              Unchanged because each outer iteration still
//                              processes 32 columns.
//   Line 21: r6  = 1        — left-shift for scale-group byte offset.
//
//   EFFECT ON LOOP COUNTS (with KC=256):
//   - Outer iterations: r1 / r5 = 4096 / 512 = 8  (was 64 for KC=2048).
//   - Inner iterations: r2 - 2 = 14  (unchanged).
//   - Total columns: 8 × 32 = 256  (matches KC).
//   - Total weight buffer size: 4096 (weights) + 8×32×2 (scales) = 4608 bytes.
//
//   VERIFIED: The scale-group loop (kg=0..KC/G-1 = 0..7) and the activation
//   pointer (r7 increments by 32 per outer iter, covering 0..224 for 256 values)
//   are both correct for KC=256 without any additional changes.
