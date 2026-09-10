# SPDX-License-Identifier: Apache-2.0
"""decode_layer_f3best operator — full fused decode layer in ONE dispatch.

8 phase-blind center tiles (cols 2-5 x rows 2-3) time-mux Q-GEMV+RoPE -> O-proj ->
FFN on one weight stream A=[Wq|Wo|gate|up|down]; 8 score + 8 value edge tiles run
spatial-8 attention; one continuous BD-chain. The MLIR is authored as raw aie-dialect
(below IRON ObjectFifo) by design.my_decode_layer_f3best -> f3best_emit; aiecc is the
same backend the IRON operators use.

Kernel objects are the same five as decode_attn_oproj (signed int4 GEMV + its O-proj
rename + interleaved RoPE + flowkv + attn concat) PLUS the fused relay kernel built
from layer_fused_f3best.cc. Shape is fixed to Llama-3.2-1B (the emitter validates);
the BO layout matches the standalone replay (bo0 out, bo1 X|resid, bo2 weights A,
bo3 unused Wo placeholder, bo4 KV cache)."""
import numpy as np
from ml_dtypes import bfloat16
from pathlib import Path

from iron.common import (
    AIEOperatorBase,
    XclbinArtifact,
    InstsBinArtifact,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
)


class AIEDecodeLayerF3Best(AIEOperatorBase):
    # f3best center layout is fixed: 8 center tiles, npu2.
    NH = 8
    WO_BYTES = 2359296          # unused arg3 placeholder (Wo lives inside A)

    def __init__(self, embed_dim=2048, hidden_dim=8192, K_gemv=2048, head_dim=64,
                 group_size=32, attn_group=4, num_kv_heads=8, m_input=4, seq_len=256,
                 num_q_heads=32, with_npu_kv=False, context=None):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.K = K_gemv
        self.head_dim = head_dim
        self.group_size = group_size
        self.attn_group = attn_group
        self.num_kv_heads = num_kv_heads
        self.m_input = m_input
        self.seq_len = seq_len
        self.num_q_heads = num_q_heads
        self.with_npu_kv = with_npu_kv
        self.WT_TILES = 960 if with_npu_kv else 896
        self.xclbin_artifact = None
        self.insts_artifact = None
        AIEOperatorBase.__init__(self, context=context)

    def get_artifacts(self, prefix="decode_layer_f3best_"):
        operator_dir = Path(__file__).parent
        E, H, g = self.embed_dim, self.hidden_dim, self.group_size
        import os as _os
        _ffn_div = _os.environ.get('F3BEST_FFN_DIV', '1')
        _div_suffix = f'_d{_ffn_div}' if _ffn_div != '1' else ''
        _decouple = '_decouple' if _os.environ.get('F3BEST_MT_DECOUPLE', '').strip() != '' else ''
        _triple_b = '_tb' if _os.environ.get('F3BEST_TRIPLE_B', '1').strip() != '' else ''
        _handasm_rr = '_rr' if _os.environ.get('F3BEST_HANDASM_RR', '').strip() != '' else ''  # hand-asm bcast path (kc256_rr.o when RR, else kc256.o)
        _cmlfix = '_cmlfix' if _os.environ.get('F3BEST_HANDASM_CMLFIX', '').strip() != '' else ''  # AM-101 copie _fix : cml1/cmh1 seedés
        _qdump = '_qdump7' if _os.environ.get('F3BEST_QDUMP', '').strip() != '' else ''  # #188 Ш21 Direct-NPU Q dump (v6 = fixed hand-asm kc256.s + output-width store)
        _kv_abi = '_kv' if self.with_npu_kv else '_nokv'
        base = (f"{prefix}{E}x{H}_d{self.head_dim}_g{g}_s{self.seq_len}"
                f"_a{self.attn_group}_kv{self.num_kv_heads}_mc_preq_vexp_vreg_dq8_qp_mxp_ub_amac{_kv_abi}{_div_suffix}{_decouple}{_triple_b}{_handasm_rr}{_cmlfix}{_qdump}")   # _rr = hand-asm RR variant (#135, F3BEST_HANDASM_RR), _qdump = #188 Ш21 direct-NPU Q dump (routes Q into the O_h relay), _mc = multi-chunk attn fix (#70/#74), _preq/_vexp = flowkv score density cuts, _vreg = register-resident value accumulator, _dq8 = single int4->int8 unpack, _qp = 4 groups/iteration in two chains, _mxp = mx packet demux fix (#142), _ub = unified bcast GEMV (#157 L1), _amac = native bf16 MAC in value tile (#176)

        mlir_artifact = PythonGeneratedMLIRArtifact.new(
            f"{base}.mlir",
            import_path=operator_dir / "design.py",
            callback_fn="my_decode_layer_f3best",
            callback_args=[
                self.context.device_manager.device_type,
                E, H, g, self.head_dim, self.num_kv_heads,
                self.attn_group, self.seq_len, self.with_npu_kv,
            ],
        )

        k2p = self.context.base_dir / "aie_kernels" / "aie2p"
        gen = self.context.base_dir / "aie_kernels" / "generic"

        # SIGNED int4 GEMV (Q-proj): on-chip (nib-8)*scale, host packs (nib-8)&0xF.
        gemv_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_signed_{self.K}k_g{g}.o",
            depends=[SourceArtifact.new(k2p / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={g}", "-DWEIGHT_SIGNED"],
        )
        # Same GEMV body renamed for the O-proj phase (distinct link symbol).
        oproj_obj = KernelObjectArtifact.new(
            f"fused_dequant_gemv_v2_oproj_signed_{self.K}k_g{g}.o",
            depends=[SourceArtifact.new(k2p / "fused_dequant_gemv_v2.cc")],
            extra_flags=[f"-DDIM_K={self.K}", f"-DGROUP_SIZE={g}", "-DWEIGHT_SIGNED",
                         "-Dfused_dequant_matvec_v2_bf16=oproj_matvec_v2_bf16"],
        )
        rope_obj = KernelObjectArtifact.new(
            "rope_il.o",
            depends=[SourceArtifact.new(gen / "rope.cc")],
            extra_flags=["-DINTERLEAVED", f"-DLUT_OFF={self.K}",
                         f"-DSEQ_META={self.head_dim}"],
        )
        flowkv_obj = KernelObjectArtifact.new(
            f"flowkv_{self.head_dim}d_h{self.attn_group}_c{self.seq_len}.o",
            depends=[SourceArtifact.new(k2p / "flowkv.cc")],
            extra_flags=[f"-DHEAD_DIM={self.head_dim}",
                         f"-DMAX_Q_HEADS={self.attn_group}",
                         f"-DMAX_CHUNK={self.seq_len}",
                         "-DFLOWKV_PRESCALE_Q=1",
                         "-DFLOWKV_VEC_EXP=1",
                         "-DFLOWKV_VALUE_AMAC=1"],
        )
        concat_obj = KernelObjectArtifact.new(
            "attn_concat.o",
            depends=[SourceArtifact.new(k2p / "attn_concat.cc")],
        )
        # Fused relay: the phase-blind center body (Q-GEMV+RoPE / O-proj / FFN) +
        # rms hub + mux, built for the 8-center-column f3best layout.
        # Guard C++ bcast functions so the hand-asm .o provides them instead (#131).
        relay_obj = KernelObjectArtifact.new(
            "layer_fused_relay.o",
            depends=[SourceArtifact.new(k2p / "layer_fused_f3best.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={H}", f"-DGROUP_SIZE={g}",
                f"-DHEAD_DIM={self.head_dim}", f"-DNUM_HEADS={self.num_q_heads}",
                f"-DNUM_KV_HEADS={self.num_kv_heads}", f"-DMAX_SEQ_LEN=2048",
                "-DNUM_AIE_COLUMNS=8", "-DM_OUTPUT_MAX=1024",
                "-DUSE_HAND_ASM_BCAST",
            ],
        )

        # #131: C++ wrapper that calls hand-asm kernel for unified broadcast GEMV.
        # Level 1: single generic_bcast_gemv_bf16 replaces 4 per-phase wrappers.
        unified_bcast_obj = KernelObjectArtifact.new(
            "layer_fused_unified_bcast.o",
            depends=[SourceArtifact.new(k2p / "layer_fused_unified_bcast.cc")],
            extra_flags=[
                f"-DEMBED_DIM={E}", f"-DHIDDEN_DIM={H}", f"-DGROUP_SIZE={g}",
                f"-DHEAD_DIM={self.head_dim}", f"-DNUM_HEADS={self.num_q_heads}",
                f"-DNUM_KV_HEADS={self.num_kv_heads}", f"-DMAX_SEQ_LEN=2048",
                "-DNUM_AIE_COLUMNS=8", "-DM_OUTPUT_MAX=1024",
                # #188 Ш22.7: cross-chunk accumulation restored (the DI diagnostic
                # bypass is gone — the wrapper accumulates float32 across chunks).
            ],
        )
        # #131: KC=256 hand-asm kernel (bcast_gemv adapted for chunk mode)
        bcast_kc256_obj = KernelObjectArtifact.new(
            "layer_fused_bcast_kc256.o",
            depends=[SourceArtifact.new(k2p / "layer_fused_bcast_kc256.s")],
            extra_flags=[],
        )
        # AM-101 (copie _fix) : même kernel avec cml1/cmh1 seedés depuis cml0=0
        # (l'original les lit sans seed → état résiduel → NaN head4). Activé par
        # F3BEST_HANDASM_CMLFIX=1. L'original n'est jamais modifié.
        bcast_kc256_cmlfix_obj = KernelObjectArtifact.new(
            "layer_fused_bcast_kc256_fix.o",
            depends=[SourceArtifact.new(k2p / "layer_fused_bcast_kc256_fix.s")],
            extra_flags=[],
        )
        # #135: RR kernel (same cml-init fix applied; scales/blow-up still suspect)
        bcast_kc256_rr_obj = KernelObjectArtifact.new(
            "layer_fused_bcast_kc256_rr.o",
            depends=[SourceArtifact.new(k2p / "layer_fused_bcast_kc256_rr.s")],
            extra_flags=[],
        )
        _use_rr = _os.environ.get('F3BEST_HANDASM_RR', '').strip() != ''
        _use_cmlfix = _os.environ.get('F3BEST_HANDASM_CMLFIX', '').strip() != ''
        if _use_cmlfix:
            _bcast_obj = bcast_kc256_cmlfix_obj
        else:
            _bcast_obj = bcast_kc256_rr_obj if _use_rr else bcast_kc256_obj

        xclbin_artifact = XclbinArtifact.new(
            f"{base}.xclbin",
            depends=[mlir_artifact, gemv_obj, oproj_obj, rope_obj, flowkv_obj,
                     concat_obj, relay_obj, unified_bcast_obj, _bcast_obj],
        )
        insts_artifact = InstsBinArtifact.new(f"{base}.bin", depends=[mlir_artifact])
        return xclbin_artifact, insts_artifact

    def set_up_artifacts(self):
        self.xclbin_artifact, self.insts_artifact = self.get_artifacts()
        self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def set_up_runtime(self):
        NH, E, g, m = self.NH, self.embed_dim, self.group_size, self.m_input
        PACKED = m * E // 2 + m * (E // g) * 2
        WT_BYTES = self.WT_TILES * PACKED
        KVN = self.seq_len * self.head_dim

        # no-KV ABI: [NH*P | s], 18432 bf16. KV ABI: [NH*(P|K|V) | s],
        # with KV_M=128 bf16 per head per projection, 20480 bf16 total.
        # bo1 [X(XB) | residual(E) | ffn_norm gain(E)]; bo2 weights A;
        # bo3 unused Wo placeholder; bo4 KV cache (K then V, NH heads).
        KV_M = 128 if self.with_npu_kv else 0
        self.add_buffer("output", NH * (E + 2*KV_M) + E, dtype=bfloat16)
        self.add_buffer("XR", (E + 256 + 16) + E + E, dtype=bfloat16)
        self.add_buffer("A", NH * WT_BYTES, dtype=np.uint8)
        self.add_buffer("Wo", self.WO_BYTES, dtype=np.uint8)
        self.add_buffer("KV", 2 * NH * KVN, dtype=bfloat16)
        self.add_kernel("decode_layer_f3best", self.xclbin_artifact,
                        self.xclbin_artifact.kernel_name, self.insts_artifact)
        self.add_to_runlist("decode_layer_f3best", "output", "XR", "A", "Wo", "KV")
