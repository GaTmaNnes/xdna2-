#!/usr/bin/env python3
"""Parametric raw-AIE emitter for F3-best decode layer — KV ABI variant.

Inherits from f3best_emit_nokv.OFold8F3BestEmitter and adds K/V-GEMV phases
on center tiles (Wk, Wv in weight stream; K/V scalar copy into P tail).
"""
import os as _os

from f3best_emit_nokv import OFold8F3BestEmitter as _BaseEmitter, CENTER_COLS, SCORE_COLS, VALUE_COLS
from f3best_emit_nokv import _LockAlloc, _DmaAlloc


class OFold8F3BestEmitter(_BaseEmitter):
    """KV ABI: adds K/V GEMV on center tiles, K/V drain in output P tail."""

    def __init__(self, NH=8, E=2048, H=8192, G=32, M=4, HD=64, AG=4, SEQ=256, POS=5):
        super().__init__(NH=NH, E=E, H=H, G=G, M=M, HD=HD, AG=AG, SEQ=SEQ, POS=POS)

        # KV-specific dimensions (2 KV heads per center tile, each head_dim outputs)
        self.KV_M = 128                           # K/V outputs per head (2 KV heads × 64)
        self.KV_T = self.KV_M // self.M            # K/V weight tiles per tile (32)
        self.KV_SZ = self.KV_M + 2                 # K/V buffer size (130 = 128 + 2 scratch)
        self.PH_STRIDE = E + 2 * self.KV_M         # per-head stride in output P (2304 for 1B)

        # Override WT_TILES: add 2*KV_T for Wk and Wv
        self.WT_TILES = (self.GEMV_T + 2 * self.KV_T + self.OPROJ_T
                         + 2 * self.GU_T + self.DN_T)
        self.WT_BYTES = self.WT_TILES * self.PACKED

        # Override output elements: NH*(E + 2*KV_M) + E
        self.OUT_ELEMS = self.NH * (E + 2 * self.KV_M) + E

        # Check L1 budget with extra K/V buffers
        self._check_kv_l1()

    def _check_kv_l1(self):
        """Ensure K+V buffers fit within L1."""
        kv_extra = 2 * self.KV_SZ * 2              # K + V buffers (bf16 bytes)
        if not self._tb_env_forced_off and self._triple_l1 + kv_extra > 60 * 1024:
            self.TRIPLE_B = False
        if self._single_l1 + kv_extra > 64 * 1024:
            raise ValueError(f'KV ABI: single-B L1 ({self._single_l1 + kv_extra}B) exceeds 64KB')

    # ═══════════════════════════════════════════════════════════════════════
    #  Override center tile methods — add K/V phases + scalar copy to P tail
    # ═══════════════════════════════════════════════════════════════════════

    def _center_triple_b(self, h, p):
        E = self.E; PK = self.PACKED; XB = self.XB; QSZ = self.Q_SZ
        OSZ = self.O_SZ; PSZ = self.P_SZ; HP = self.HPER; GT = self.GU_T
        PT = self.PER_TILE; UNI = self.UNI_SZ; KSZ = self.KV_SZ; KT = self.KV_T
        PH = self.PH_STRIDE; KV2 = self.KV_M // 2

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core')
        D = _DmaAlloc('core')
        _a0 = L.pair('A0'); _b0 = L.pair('B0'); _q = L.pair('Q')
        _o = L.pair('O');   _p = L.pair('P');   _a1 = L.pair('A1')
        _b1 = L.pair('B1'); _b2 = L.pair('B2')
        _s2mm_w  = D.s2mm('weight')
        _s2mm_b  = D.s2mm('bcast')
        _mm2s_p  = D.mm2s('drain')
        _mm2s_qo = D.mm2s('qo')

        _ld = lambda name, ids: _LockAlloc.mlir_decl(f't{h}', f'{p}', name, *ids)
        _lock_decls = '\n'.join([
            _ld('A0', _a0), _ld('B0', _b0), _ld('Q', _q), _ld('O', _o),
            _ld('P', _p), _ld('A1', _a1), _ld('B1', _b1), _ld('B2', _b2),
        ])

        return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<{PK}xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<{PK}xi8>
    %{p}_B0 = aie.buffer(%t{h}) {{sym_name = "{p}_B0"}} : memref<{XB}xbf16>
    %{p}_B1 = aie.buffer(%t{h}) {{sym_name = "{p}_B1"}} : memref<{XB}xbf16>
    %{p}_B2 = aie.buffer(%t{h}) {{sym_name = "{p}_B2"}} : memref<{XB}xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<{QSZ}xbf16>
    %{p}_K = aie.buffer(%t{h}) {{sym_name = "{p}_K"}} : memref<{KSZ}xbf16>
    %{p}_V = aie.buffer(%t{h}) {{sym_name = "{p}_V"}} : memref<{KSZ}xbf16>
    %{p}_O = aie.buffer(%t{h}) {{sym_name = "{p}_O"}} : memref<{OSZ}xbf16>
    %{p}_P = aie.buffer(%t{h}) {{sym_name = "{p}_P"}} : memref<{PSZ}xbf16>
    {_lock_decls}
    %{p}_gate = aie.buffer(%t{h}) {{sym_name = "{p}_gate"}} : memref<{HP}xbf16>
    %{p}_up   = aie.buffer(%t{h}) {{sym_name = "{p}_up"}}   : memref<{HP}xbf16>
    %{p}_silu = aie.buffer(%t{h}) {{sym_name = "{p}_silu"}} : memref<{HP}xbf16>
    %{p}_uni_partial = aie.buffer(%t{h}) {{sym_name = "{p}_uni_partial"}} : memref<{UNI}xi8>
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant {self.GEMV_T} : index
      %cKV = arith.constant {KT} : index
      %cF = arith.constant {GT} : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant {PT} : i32
      %kr = arith.constant {KV2} : i32
      %hc = arith.constant {HP} : i32
      %cE = arith.constant {E} : index
      %cEk = arith.constant {E + self.KV_M} : index
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope + K-GEMV + K-RoPE + V-GEMV (B0 = x_bundle) ----
        aie.use_lock(%{p}_B0c, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %{p}_A0, %{p}_B0, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %{p}_A1, %{p}_B0, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B0, %{p}_Q, %qr) : (memref<{QSZ}xbf16>, memref<{XB}xbf16>, memref<{QSZ}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Qc, Release, 1)
        // phase 1b: K-GEMV + K-RoPE
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B0, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B0, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_kv_bundled(%{p}_K, %{p}_B0, %{p}_K, %kr) : (memref<{KSZ}xbf16>, memref<{XB}xbf16>, memref<{KSZ}xbf16>, i32) -> ()
        // phase 1c: V-GEMV
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B0, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B0, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_B0p, Release, 1)
        // ---- phase 2: O-proj (B1 = attn_out) ----
        aie.use_lock(%{p}_B1c, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %{p}_A0, %{p}_B1, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %{p}_A1, %{p}_B1, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_B1p, Release, 1)
        aie.use_lock(%{p}_Oc, Release, 1)
        // ---- phase 3: FFN (B2 = ffn_in) ----
        aie.use_lock(%{p}_B2c, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A0, %{p}_B2, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B2, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A1, %{p}_B2, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B2, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_B2p, Release, 1)
        func.call @layer_fused_silu_mul_explicit_bf16(%{p}_gate, %{p}_up, %{p}_silu, %hc) : (memref<{HP}xbf16>, memref<{HP}xbf16>, memref<{HP}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %{p}_A0, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<{PK}xi8>, memref<{HP}xbf16>, memref<{UNI}xi8>, i32, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %{p}_A1, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<{PK}xi8>, memref<{HP}xbf16>, memref<{UNI}xi8>, i32, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        // Copy K_buf -> P[E : E+KV_M]
        scf.for %i = %c0 to %cE step %c1 {{
          %val = memref.load %{p}_K[%i] : memref<{KSZ}xbf16>
          %dst = arith.addi %i, %cE : index
          memref.store %val, %{p}_P[%dst] : memref<{PSZ}xbf16>
        }}
        // Copy V_buf -> P[E+KV_M : E+2*KV_M]
        scf.for %i = %c0 to %cE step %c1 {{
          %val = memref.load %{p}_V[%i] : memref<{KSZ}xbf16>
          %dst = arith.addi %i, %cEk : index
          memref.store %val, %{p}_P[%dst] : memref<{PSZ}xbf16>
        }}
        aie.use_lock(%{p}_Pc, Release, 1)
      }}
      aie.end
    }}
    %mem_t{h} = aie.mem(%t{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_w}, ^a0{h}, ^bs{h})
    ^a0{h}:
      aie.use_lock(%{p}_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A0 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A0c, Release, 1)
      aie.next_bd ^a1{h}
    ^a1{h}:
      aie.use_lock(%{p}_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A1 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A1c, Release, 1)
      aie.next_bd ^a0{h}
    ^bs{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_b}, ^b0{h}, ^m0{h})
    ^b0{h}:
      aie.use_lock(%{p}_B0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B0 : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_B0c, Release, 1)
      aie.next_bd ^b1{h}
    ^b1{h}:
      aie.use_lock(%{p}_B1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B1 : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_B1c, Release, 1)
      aie.next_bd ^b2{h}
    ^b2{h}:
      aie.use_lock(%{p}_B2p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B2 : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_B2c, Release, 1)
      aie.next_bd ^b0{h}
    ^m0{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_p}, ^pf{h}, ^qo{h})
    ^pf{h}:
      aie.use_lock(%{p}_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_P : memref<{PSZ}xbf16>, 0, {PH})
      aie.use_lock(%{p}_Pp, Release, 1)
      aie.next_bd ^pf{h}
    ^qo{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_qo}, ^qi{h}, ^e{h})
    ^qi{h}:
      aie.use_lock(%{p}_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Q : memref<{QSZ}xbf16>, 0, {QSZ})
      aie.use_lock(%{p}_Qp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_O : memref<{OSZ}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^qi{h}
    ^e{h}:
      aie.end
    }}"""

    def _center_single_b(self, h, p):
        E = self.E; PK = self.PACKED; XB = self.XB; QSZ = self.Q_SZ
        OSZ = self.O_SZ; PSZ = self.P_SZ; HP = self.HPER; GT = self.GU_T
        PT = self.PER_TILE; UNI = self.UNI_SZ; KSZ = self.KV_SZ; KT = self.KV_T
        PH = self.PH_STRIDE; KV2 = self.KV_M // 2

        # Auto-allocate locks and DMA channels (#154)
        L = _LockAlloc('core')
        D = _DmaAlloc('core')
        _a0 = L.pair('A0'); _b = L.pair('B'); _q = L.pair('Q')
        _o = L.pair('O');   _p = L.pair('P'); _a1 = L.pair('A1')
        _s2mm_w  = D.s2mm('weight')
        _s2mm_b  = D.s2mm('bcast')
        _mm2s_p  = D.mm2s('drain')
        _mm2s_qo = D.mm2s('qo')

        _ld = lambda name, ids: _LockAlloc.mlir_decl(f't{h}', f'{p}', name, *ids)
        _lock_decls = '\n'.join([
            _ld('A0', _a0), _ld('B', _b), _ld('Q', _q),
            _ld('O', _o), _ld('P', _p), _ld('A1', _a1),
        ])

        return f"""
    %{p}_A0 = aie.buffer(%t{h}) {{sym_name = "{p}_A0"}} : memref<{PK}xi8>
    %{p}_A1 = aie.buffer(%t{h}) {{sym_name = "{p}_A1"}} : memref<{PK}xi8>
    %{p}_B = aie.buffer(%t{h}) {{sym_name = "{p}_B"}} : memref<{XB}xbf16>
    %{p}_Q = aie.buffer(%t{h}) {{sym_name = "{p}_Q"}} : memref<{QSZ}xbf16>
    %{p}_K = aie.buffer(%t{h}) {{sym_name = "{p}_K"}} : memref<{KSZ}xbf16>
    %{p}_V = aie.buffer(%t{h}) {{sym_name = "{p}_V"}} : memref<{KSZ}xbf16>
    %{p}_O = aie.buffer(%t{h}) {{sym_name = "{p}_O"}} : memref<{OSZ}xbf16>
    %{p}_P = aie.buffer(%t{h}) {{sym_name = "{p}_P"}} : memref<{PSZ}xbf16>
    {_lock_decls}
    %{p}_gate = aie.buffer(%t{h}) {{sym_name = "{p}_gate"}} : memref<{HP}xbf16>
    %{p}_up   = aie.buffer(%t{h}) {{sym_name = "{p}_up"}}   : memref<{HP}xbf16>
    %{p}_silu = aie.buffer(%t{h}) {{sym_name = "{p}_silu"}} : memref<{HP}xbf16>
    %{p}_uni_partial = aie.buffer(%t{h}) {{sym_name = "{p}_uni_partial"}} : memref<{UNI}xi8>
    %core{h} = aie.core(%t{h}) {{
      %c0 = arith.constant 0 : index
      %cN = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant {self.GEMV_T} : index
      %cKV = arith.constant {KT} : index
      %cF = arith.constant {GT} : index
      %c4 = arith.constant 4 : i32
      %c8 = arith.constant 8 : i32
      %qr = arith.constant {PT} : i32
      %kr = arith.constant {KV2} : i32
      %hc = arith.constant {HP} : i32
      %cE = arith.constant {E} : index
      %cEk = arith.constant {E + self.KV_M} : index
      scf.for %tok = %c0 to %cN step %c1 {{
        func.call @_ha_noop() : () -> ()
        // ---- phase 1: Q-GEMV + rope + K-GEMV + K-RoPE + V-GEMV (B = x_bundle) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Qp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_q(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_Q) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{QSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_bundled(%{p}_Q, %{p}_B, %{p}_Q, %qr) : (memref<{QSZ}xbf16>, memref<{XB}xbf16>, memref<{QSZ}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Qc, Release, 1)
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_K) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        func.call @rope_kv_bundled(%{p}_K, %{p}_B, %{p}_K, %kr) : (memref<{KSZ}xbf16>, memref<{XB}xbf16>, memref<{KSZ}xbf16>, i32) -> ()
        scf.for %j = %c0 to %cKV step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_kv(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_V) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        // ---- phase 2: O-proj (B = attn_out) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        aie.use_lock(%{p}_Op, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %c64 step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_o(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_O) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{OSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        aie.use_lock(%{p}_Oc, Release, 1)
        // ---- phase 3: FFN (B = ffn_in) ----
        aie.use_lock(%{p}_Bc, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A0, %{p}_B, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_gate) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji0, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_g(%ji1, %{p}_A1, %{p}_B, %{p}_uni_partial, %c8, %{p}_up) : (i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        aie.use_lock(%{p}_Bp, Release, 1)
        func.call @layer_fused_silu_mul_explicit_bf16(%{p}_gate, %{p}_up, %{p}_silu, %hc) : (memref<{HP}xbf16>, memref<{HP}xbf16>, memref<{HP}xbf16>, i32) -> ()
        aie.use_lock(%{p}_Pp, AcquireGreaterEqual, 1)
        scf.for %j = %c0 to %cF step %c2 {{
          aie.use_lock(%{p}_A0c, AcquireGreaterEqual, 1)
          %ji0 = arith.index_cast %j : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji0, %{p}_A0, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<{PK}xi8>, memref<{HP}xbf16>, memref<{UNI}xi8>, i32, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A0p, Release, 1)
          aie.use_lock(%{p}_A1c, AcquireGreaterEqual, 1)
          %j1 = arith.addi %j, %c1 : index
          %ji1 = arith.index_cast %j1 : index to i32
          func.call @generic_bcast_gemv_bf16_d(%ji1, %{p}_A1, %{p}_silu, %{p}_uni_partial, %c4, %{p}_P) : (i32, memref<{PK}xi8>, memref<{HP}xbf16>, memref<{UNI}xi8>, i32, memref<{PSZ}xbf16>) -> ()
          aie.use_lock(%{p}_A1p, Release, 1)
        }}
        scf.for %i = %c0 to %cE step %c1 {{
          %val = memref.load %{p}_K[%i] : memref<{KSZ}xbf16>
          %dst = arith.addi %i, %cE : index
          memref.store %val, %{p}_P[%dst] : memref<{PSZ}xbf16>
        }}
        scf.for %i = %c0 to %cE step %c1 {{
          %val = memref.load %{p}_V[%i] : memref<{KSZ}xbf16>
          %dst = arith.addi %i, %cEk : index
          memref.store %val, %{p}_P[%dst] : memref<{PSZ}xbf16>
        }}
        aie.use_lock(%{p}_Pc, Release, 1)
      }}
      aie.end
    }}
    %mem{h} = aie.mem(%t{h}) {{
      %s0 = aie.dma_start(S2MM, {_s2mm_w}, ^a0{h}, ^bs{h})
    ^a0{h}:
      aie.use_lock(%{p}_A0p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A0 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A0c, Release, 1)
      aie.next_bd ^a1{h}
    ^a1{h}:
      aie.use_lock(%{p}_A1p, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_A1 : memref<{PK}xi8>, 0, {PK})
      aie.use_lock(%{p}_A1c, Release, 1)
      aie.next_bd ^a0{h}
    ^bs{h}:
      %s1 = aie.dma_start(S2MM, {_s2mm_b}, ^b{h}, ^m0{h})
    ^b{h}:
      aie.use_lock(%{p}_Bp, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_B : memref<{XB}xbf16>, 0, {XB})
      aie.use_lock(%{p}_Bc, Release, 1)
      aie.next_bd ^b{h}
    ^m0{h}:
      %m0 = aie.dma_start(MM2S, {_mm2s_p}, ^pf{h}, ^m1{h})
    ^pf{h}:
      aie.use_lock(%{p}_Pc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_P : memref<{PSZ}xbf16>, 0, {PH})
      aie.use_lock(%{p}_Pp, Release, 1)
      aie.next_bd ^pf{h}
    ^m1{h}:
      %m1 = aie.dma_start(MM2S, {_mm2s_qo}, ^qo{h}, ^e{h})
    ^qo{h}:
      aie.use_lock(%{p}_Qc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_Q : memref<{QSZ}xbf16>, 0, {QSZ})
      aie.use_lock(%{p}_Qp, Release, 1)
      aie.next_bd ^oh{h}
    ^oh{h}:
      aie.use_lock(%{p}_Oc, AcquireGreaterEqual, 1)
      aie.dma_bd(%{p}_O : memref<{OSZ}xbf16>, 0, {PT})
      aie.use_lock(%{p}_Op, Release, 1)
      aie.next_bd ^qo{h}
    ^e{h}:
      aie.end
    }}"""

    def emit_mlir(self):
        """Like base but adds rope_kv_bundled + generic_bcast_gemv_bf16_kv funcs,
        and KV-specific runtime sequence (P drain uses PH_STRIDE, output BO layout)."""
        NH = self.NH; E = self.E; WT_BYTES = self.WT_BYTES
        KVN = self.KVN; XB = self.XB; HALF_KV = self.HALF_KV
        PK = self.PACKED; QSZ = self.Q_SZ; OSZ = self.O_SZ; PSZ = self.P_SZ
        HP = self.HPER; UNI = self.UNI_SZ; KSZ = self.KV_SZ; PT = self.PER_TILE
        KCH = self.KCH; ITC = self.ITC; PH = self.PH_STRIDE

        # tile declarations (same as base)
        tile_decls = "".join(f"    %t{h} = aie.tile({CENTER_COLS[h][0]}, {CENTER_COLS[h][1]})\n" for h in range(NH))
        tile_decls += "".join(f"    %sc{h} = aie.tile({SCORE_COLS[h][0]}, {SCORE_COLS[h][1]})\n" for h in range(NH))
        tile_decls += "".join(f"    %va{h} = aie.tile({VALUE_COLS[h][0]}, {VALUE_COLS[h][1]})\n" for h in range(NH))
        tile_decls += "    %jA = aie.tile(2, 1)\n    %jB = aie.tile(3, 1)\n"
        tile_decls += "    %Klo = aie.tile(0, 1)\n    %Khi = aie.tile(1, 1)\n    %Vlo = aie.tile(6, 1)\n    %Vhi = aie.tile(7, 1)\n"
        tile_decls += "    %oJ = aie.tile(4, 1)\n    %oK = aie.tile(5, 1)\n"
        if self.RL_FIX:
            tile_decls += "    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
        else:
            tile_decls += "    %rl = aie.tile(2, 4)\n    %op = aie.tile(3, 4)\n    %nm = aie.tile(4, 4)\n    %mx = aie.tile(5, 4)\n"
        shim_decls = "".join(f"    %sh{h} = aie.tile({h}, 0)\n" for h in range(NH))

        # KV kernel declarations (adds rope_kv_bundled + generic_bcast_gemv_bf16_kv vs no-KV)
        _cmlfix = getattr(self, 'HANDASM_CMLFIX', False)
        _ha_link = 'layer_fused_bcast_kc256_fix.o' if _cmlfix else ('layer_fused_bcast_kc256_rr.o' if getattr(self, 'HANDASM_RR', False) else 'layer_fused_bcast_kc256.o')
        funcs = (
            f'    func.func private @_ha_noop() -> () attributes {{link_with = "{_ha_link}"}}\n'
            f'    func.func private @fused_dequant_matvec_v2_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{QSZ}xbf16>) attributes {{link_with = "fused_dequant_gemv_v2_signed_2048k_g32.o"}}\n'
            f'    func.func private @rope_bundled(memref<{QSZ}xbf16>, memref<{XB}xbf16>, memref<{QSZ}xbf16>, i32) attributes {{link_with = "rope_il.o"}}\n'
            f'    func.func private @rope_kv_bundled(memref<{KSZ}xbf16>, memref<{XB}xbf16>, memref<{KSZ}xbf16>, i32) attributes {{link_with = "rope_il.o"}}\n'
            f'    func.func private @layer_fused_gate_up_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, i32) attributes {{link_with = "layer_fused_relay.o"}}\n'
            f'    func.func private @generic_bcast_gemv_bf16_q(i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{QSZ}xbf16>) attributes {{link_with = "layer_fused_unified_bcast.o"}}\n'
            f'    func.func private @generic_bcast_gemv_bf16_o(i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{OSZ}xbf16>) attributes {{link_with = "layer_fused_unified_bcast.o"}}\n'
            f'    func.func private @generic_bcast_gemv_bf16_g(i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{HP}xbf16>) attributes {{link_with = "layer_fused_unified_bcast.o"}}\n'
            f'    func.func private @generic_bcast_gemv_bf16_kv(i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{UNI}xi8>, i32, memref<{KSZ}xbf16>) attributes {{link_with = "layer_fused_unified_bcast.o"}}\n'
            f'    func.func private @generic_bcast_gemv_bf16_d(i32, memref<{PK}xi8>, memref<{HP}xbf16>, memref<{UNI}xi8>, i32, memref<{PSZ}xbf16>) attributes {{link_with = "layer_fused_unified_bcast.o"}}\n'
            f'    func.func private @layer_fused_silu_mul_explicit_bf16(memref<{HP}xbf16>, memref<{HP}xbf16>, memref<{HP}xbf16>, i32) attributes {{link_with = "layer_fused_relay.o"}}\n'
            f'    func.func private @layer_fused_down_v2_x4_bf16(i32, i32, i32, memref<{PK}xi8>, memref<{PSZ}xbf16>) attributes {{link_with = "layer_fused_relay.o"}}\n'
            f'    func.func private @attn_copy_bf16(memref<{XB}xbf16>, memref<{XB}xbf16>, i32) attributes {{link_with = "attn_concat.o"}}\n'
            f'    func.func private @oproj_matvec_v2_bf16(i32, i32, memref<{PK}xi8>, memref<{XB}xbf16>, memref<{OSZ}xbf16>) attributes {{link_with = "fused_dequant_gemv_v2_oproj_signed_2048k_g32.o"}}\n'
            f'    func.func private @layer_fused_add_bf16(memref<{E}xbf16>, memref<{E}xbf16>, memref<{XB}xbf16>, i32) attributes {{link_with = "layer_fused_relay.o"}}\n'
            f'    func.func private @layer_fused_rms_norm2_bf16(memref<{XB}xbf16>, memref<{E}xbf16>, memref<{XB}xbf16>, i32) attributes {{link_with = "layer_fused_relay.o"}}\n'
            f'    func.func private @flowkv_score_init_bf16(i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_score_rope_q_bf16(memref<{QSZ}xbf16>, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_score_chunk_bf16(memref<{QSZ}xbf16>, memref<{KCH}xbf16>, memref<{ITC}xbf16>, i32, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_value_init_bf16(i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_value_accum_bf16(memref<{ITC}xbf16>, memref<{KCH}xbf16>, i32, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n'
            f'    func.func private @flowkv_value_normalize_bf16(memref<{PT}xbf16>, i32, i32) attributes {{link_with = "{self.FLOWKV_LIB}"}}\n')

        # flows (same as base)
        flows = []
        flows.append("    aie.flow(%sh0, DMA : 1, %mx, DMA : 0)   // x_bundle -> mux S2MM0")
        if self.RL_FIX:
            flows.append("    aie.packet_flow(0) { aie.packet_source<%jA, DMA : 0> aie.packet_dest<%mx, DMA : 1> }")
            flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }")
            flows.append("    aie.packet_flow(2) { aie.packet_source<%jB, DMA : 0> aie.packet_dest<%mx, DMA : 1> }")
        else:
            flows.append("    aie.packet_flow(0) { aie.packet_source<%rl, DMA : 0> aie.packet_dest<%mx, DMA : 1> }")
            flows.append("    aie.packet_flow(1) { aie.packet_source<%nm, DMA : 0> aie.packet_dest<%mx, DMA : 1> }")
        EDGE_RELAY = {0: 'Klo', 1: 'Khi', 6: 'Vlo', 7: 'Vhi'} if self.DECOUPLE else {}
        for h in range(NH):
            oj = 'oJ' if h < 4 else 'oK'; osl = h if h < 4 else h - 4
            rc = h % 4 if h % 4 in self.RELAY_COLS else None
            edge_mt = EDGE_RELAY.get(h)
            if edge_mt is not None:
                flows.append(f"    aie.flow(%sh{h}, DMA : 0, %{edge_mt}, DMA : 4)")
                flows.append(f"    aie.flow(%{edge_mt}, DMA : 4, %t{h}, DMA : 0)")
            elif rc is not None:
                mt_name = ['jA', 'jB', 'oJ', 'oK'][rc]
                s2mm_ch = 4 if h < 4 else 5; mm2s_ch = 1 if h < 4 else 2
                flows.append(f"    aie.flow(%sh{h}, DMA : 0, %{mt_name}, DMA : {s2mm_ch})")
                flows.append(f"    aie.flow(%{mt_name}, DMA : {mm2s_ch}, %t{h}, DMA : 0)")
            else:
                flows.append(f"    aie.flow(%sh{h}, DMA : 0, %t{h}, DMA : 0)")
            flows.append(f"    aie.flow(%mx, DMA : 0, %t{h}, DMA : 1)")
            flows.append(f"    aie.flow(%t{h}, DMA : 0, %sh{h}, DMA : 0)")
            flows.append(f"    aie.flow(%t{h}, DMA : 1, %sc{h}, DMA : 0)")
            flows.append(f"    aie.flow(%sc{h}, DMA : 0, %va{h}, DMA : 0)")
            flows.append(f"    aie.flow(%sc{h}, DMA : 1, %{oj}, DMA : {osl})")
        flows.append("    aie.flow(%sh3, DMA : 1, %Klo, DMA : 0)")
        flows.append("    aie.flow(%sh4, DMA : 1, %Khi, DMA : 0)")
        flows.append("    aie.flow(%sh5, DMA : 1, %Vlo, DMA : 0)")
        flows.append("    aie.flow(%sh6, DMA : 1, %Vhi, DMA : 0)")
        for k in range(4):
            flows.append(f"    aie.flow(%Klo, DMA : {k}, %sc{k}, DMA : 1)")
            flows.append(f"    aie.flow(%Khi, DMA : {k}, %sc{k+4}, DMA : 1)")
            flows.append(f"    aie.flow(%Vlo, DMA : {k}, %va{k}, DMA : 1)")
            flows.append(f"    aie.flow(%Vhi, DMA : {k}, %va{k+4}, DMA : 1)")
        for k in range(4):
            flows.append(f"    aie.flow(%va{k}, DMA : 0, %jA, DMA : {k})")
            flows.append(f"    aie.flow(%va{k+4}, DMA : 0, %jB, DMA : {k})")
        if not self.RL_FIX:
            flows.append("    aie.flow(%jA, DMA : 0, %rl, DMA : 0)")
            flows.append("    aie.flow(%jB, DMA : 0, %rl, DMA : 1)")
        flows.append("    aie.flow(%oJ, DMA : 0, %op, DMA : 0)")
        flows.append("    aie.flow(%oK, DMA : 0, %op, DMA : 1)")
        flows.append("    aie.flow(%op, DMA : 0, %nm, DMA : 0)")
        flows.append("    aie.flow(%sh2, DMA : 1, %nm, DMA : 1)")
        flows.append("    aie.flow(%nm, DMA : 1, %sh4, DMA : 1)")
        flows_txt = "\n".join(flows) + "\n"

        # shim allocations
        shim_allocs = "    aie.shim_dma_allocation @X_alloc(%sh0, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @R_alloc(%sh2, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Klo_alloc(%sh3, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Khi_alloc(%sh4, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Vlo_alloc(%sh5, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @Vhi_alloc(%sh6, MM2S, 1)\n"
        shim_allocs += "    aie.shim_dma_allocation @S_alloc(%sh4, S2MM, 1)\n"
        shim_allocs += "".join(
            f'    aie.shim_dma_allocation @A{h}(%sh{h}, MM2S, 0)\n'
            f'    aie.shim_dma_allocation @P{h}(%sh{h}, S2MM, 0)\n' for h in range(NH))

        WT_TY = f"{NH*WT_BYTES}xi8"
        P_TY = f"{self.OUT_ELEMS}xbf16"
        KV_TY = f"{2*NH*KVN}xbf16"

        # runtime sequence (KV: P drain uses PH_STRIDE, output strides)
        rt = []
        rt.append(f"""      %tx = aiex.dma_configure_task_for @X_alloc {{
        aie.dma_bd(%arg1 : memref<{self.XR_ELEMS}xbf16>, 0, {XB}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {XB}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tx)
      %tr = aiex.dma_configure_task_for @R_alloc {{
        aie.dma_bd(%arg1 : memref<{self.XR_ELEMS}xbf16>, {XB}, {2*E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {2*E}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tr)
      %tklo = aiex.dma_configure_task_for @Klo_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, 0, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tklo)
      %tkhi = aiex.dma_configure_task_for @Khi_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, {HALF_KV}, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tkhi)
      %tvlo = aiex.dma_configure_task_for @Vlo_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, {2*HALF_KV}, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tvlo)
      %tvhi = aiex.dma_configure_task_for @Vhi_alloc {{
        aie.dma_bd(%arg4 : memref<{KV_TY}>, {3*HALF_KV}, {HALF_KV}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {HALF_KV}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%tvhi)""")
        for h in range(NH):
            rt.append(f"""      %ta{h} = aiex.dma_configure_task_for @A{h} {{
        aie.dma_bd(%arg2 : memref<{WT_TY}>, {h*WT_BYTES}, {WT_BYTES}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {WT_BYTES}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%ta{h})""")
            # KV: P drain uses PH_STRIDE (E + K/V tail), output offset uses E (packed stride in BO)
            rt.append(f"""      %tp{h} = aiex.dma_configure_task_for @P{h} {{
        aie.dma_bd(%arg0 : memref<{P_TY}>, {h*E}, {PH}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {PH}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%tp{h})""")
        # S_alloc: residual output at NH*E
        rt.append(f"""      %ts = aiex.dma_configure_task_for @S_alloc {{
        aie.dma_bd(%arg0 : memref<{P_TY}>, {NH*E}, {E}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {E}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%ts)""")
        rt.append("".join(f"      aiex.dma_await_task(%tp{h})\n" for h in range(NH)).rstrip()
                  + "\n      aiex.dma_await_task(%ts)")
        rt_body = "\n".join(rt) + "\n"
        rt_args = (f"%arg0: memref<{P_TY}>, %arg1: memref<{self.XR_ELEMS}xbf16>, %arg2: memref<{WT_TY}>, "
                   f"%arg3: memref<{self.WO_BYTES}xi8>, %arg4: memref<{KV_TY}>")

        # assemble
        centers = "".join(self._center(h) for h in range(NH))
        scores = "".join(self._score(h) for h in range(NH))
        values = "".join(self._value(h) for h in range(NH))
        if self.RL_FIX:
            joins = (self._join_memtile('jA', 'jA', with_weight_relay=(0 in self.RELAY_COLS), packet_id=0)
                   + self._join_memtile('jB', 'jB', with_weight_relay=(1 in self.RELAY_COLS), packet_id=2)
                   + self._join_memtile('oJ', 'oJ', with_weight_relay=(2 in self.RELAY_COLS))
                   + self._join_memtile('oK', 'oK', with_weight_relay=(3 in self.RELAY_COLS)))
        else:
            joins = "".join(self._join_memtile(n, n, with_weight_relay=(c in self.RELAY_COLS))
                            for c, n in enumerate(['jA', 'jB', 'oJ', 'oK']))
        if self.DECOUPLE:
            ksplit = (self._split_memtile('Klo', 'Klo', with_weight_relay=True)
                    + self._split_memtile('Khi', 'Khi', with_weight_relay=True)
                    + self._split_memtile('Vlo', 'Vlo', with_weight_relay=True)
                    + self._split_memtile('Vhi', 'Vhi', with_weight_relay=True))
        else:
            ksplit = (self._split_memtile('Klo', 'Klo') + self._split_memtile('Khi', 'Khi')
                    + self._split_memtile('Vlo', 'Vlo') + self._split_memtile('Vhi', 'Vhi'))
        _relay_block = "" if self.RL_FIX else self._relay_block()

        MLIRTXT = (f"module {{\n  aie.device(npu2) {{\n{tile_decls}{shim_decls}\n{funcs}\n"
                   f"{flows_txt}\n{centers}\n{scores}\n{values}\n{joins}\n{ksplit}\n{_relay_block}\n{self._op_block()}\n{self._nm_block()}\n{self._mux_block()}\n"
                   f"{shim_allocs}\n    aie.runtime_sequence({rt_args}) {{\n{rt_body}    }}\n  }}\n}}\n")
        return MLIRTXT


def emit_mlir(**kwargs):
    return OFold8F3BestEmitter(**kwargs).emit_mlir()
