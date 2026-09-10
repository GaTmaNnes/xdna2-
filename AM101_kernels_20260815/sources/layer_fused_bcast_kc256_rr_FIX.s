// ppbase-density hand-asm GEMV kernel adapted for KC=256
// Inner loop: 4 bundles per 2 vmac (2.0 b/vmac) — FFLM density
// Bank map: dm0=bias, dm1=acc_even(g0), dm2=acc_odd(g1), dm3=vups_raw, dm4=vsub_adjusted
// vconv reads from dm4 (PREV-iter vsub.f), not dm3 → correct dequant pipeline
//
// Adapted from dev_notes/track_a_build/handasm_probe/ppbase.s
// Changes: K total 32768→4096, function name → bcast_kc256_bf16, +_ha_noop

		.file	"bcast_gemv_rr.cpp"
		.section	.text.layer_fused_gemv_bcast_kc256_bf16,"ax",@progbits
		.globl	layer_fused_gemv_bcast_kc256_bf16 // -- Begin function layer_fused_gemv_bcast_kc256_bf16
		.p2align	4
		.type	layer_fused_gemv_bcast_kc256_bf16,@function
	layer_fused_gemv_bcast_kc256_bf16:     // @layer_fused_gemv_bcast_kc256_bf16
	// %bb.0:                               // %entry
		movxm	r0, #19201
		movxm	r5, #4096                   // KC=256: 256*32/2 = 4096 weight bytes
		nopx	;		mov	crrnd, #12
		mov	crunpacksize, #0
		mov	crupsmode, #0
		mova	r1, #0;		vbcst.16	 x0, r0
		vbcst.32	 x2, r1
		mov	s0, r1
		vmov	bmll1, x2
		mova	r2, #16;		vconv.fp32.bf16	cml0, x0
		mova	r4, #512;		movx	r3, #828;		vmov	bmlh1, x2
		mova	r7, #0;		movx	r6, #1;		vmov	cmh0, cml0
		mova	r0, #60;		movx	r16, #0;		vmov	cml2, cml0 // FIX AM-101: seed acc dm1/dm2 from cml0=0 (cml1 = résidu non-init)
	.LBB0_1:                                // %for.body
	                                        // =>This Loop Header: Depth=1
	                                        //     Child Loop BB0_2 Depth 2
		nopa	;		nopb	;		lshl	 r17, r7, r6;		nopm
		mov	dj0, r17
		vldb	 x0, [p1, dj0];		mov	r17, r16
		movs	dj0, r17;		add	r17, r17, #32
		vldb.unpack	 x1, unpacksign1, [p0, dj0]
		nop
		nop
		movs	dj0, r17;		add	r17, r17, #32
		vldb.unpack	 x1, unpacksign1, [p0, dj0]
		nop
		nop
		movs	dj0, r17;		add	r17, r17, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r1
		vldb.unpack	 x1, unpacksign1, [p0, dj0]
		vsub.f	dm4, dm3, dm0, r0
		movxm	ls, #.LBB0_2
		mova	r18, #0;		nopb	;		movs	dj0, r17;		add	r17, r17, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r1
		vldb.unpack	 x1, unpacksign1, [p0, dj0];		add	r19, r18, #1;		add.nc	lc, r2, #-5
		add	r18, r18, #2;		vextbcst.16	 x7, x0, r18;		vsub.f	dm4, dm3, dm0, r0
		nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd0;		nopv
		nopa	;		nopb	;		movs	dj0, r17;		add	r17, r17, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r1
		nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj0];		vconv.bf16.fp32	 x3, cml4;		add	r19, r18, #1;		vextbcst.16	 x9, x0, r19;		nopv
		nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		add	r18, r18, #2;		vextbcst.16	 x7, x0, r18;		vsub.f	dm4, dm3, dm0, r0
		nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x3, x7, r3
	.LBB0_2:                                // %_ZNK3aie15vector_elem_refI8bfloat16Lj32EEcvS1_Ev.exit
	                                        //   Parent Loop BB0_1 Depth=1
	                                        // =>  This Inner Loop Header: Depth=2
		nopa	;		nopb	;		movs	dj0, r17;		add	r17, r17, #32;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r1
		nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj0];		vconv.bf16.fp32	 x3, cml4;		add	r19, r18, #1;		vextbcst.16	 x9, x0, r19;		vmac.f	dm2, dm2, x5, x9, r3
		nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh4;		add	r18, r18, #2;		vextbcst.16	 x7, x0, r18;		vsub.f	dm4, dm3, dm0, r0
	.L_LEnd0:
		nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm1, dm1, x3, x7, r3
	// %bb.3:                               // %for.cond.cleanup3
	                                        //   in Loop: Header=BB0_1 Depth=1
		nopa	;		nopx	;		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r1
		vconv.bf16.fp32	 x3, cml4;		add	r19, r18, #1;		vextbcst.16	 x9, x0, r19;		vmac.f	dm2, dm2, x5, x9, r3
		vconv.bf16.fp32	 x5, cmh4;		add	r18, r18, #2;		vextbcst.16	 x7, x0, r18;		vsub.f	dm4, dm3, dm0, r0
		vmac.f	dm1, dm1, x3, x7, r3
		vups.4x	dm3, x1, s0, upssign1;		vadd	dm3, dm3, dm0, r1
		vconv.bf16.fp32	 x3, cml4;		add	r19, r18, #1;		vextbcst.16	 x9, x0, r19;		vmac.f	dm2, dm2, x5, x9, r3
		vconv.bf16.fp32	 x5, cmh4;		add	r18, r18, #2;		vextbcst.16	 x7, x0, r18;		vsub.f	dm4, dm3, dm0, r0
		vmac.f	dm1, dm1, x3, x7, r3
		nop
		vconv.bf16.fp32	 x3, cml4;		add	r19, r18, #1;		vextbcst.16	 x9, x0, r19;		vmac.f	dm2, dm2, x5, x9, r3
		vconv.bf16.fp32	 x5, cmh4;		add	r18, r18, #2;		vextbcst.16	 x7, x0, r18
		add	 r16, r16, r4;		vmac.f	dm1, dm1, x3, x7, r3
		eq	 r17, r16, r5
		vconv.bf16.fp32	 x3, cml4;		add	r7, r7, #32;		vextbcst.16	 x9, x0, r19;		vmac.f	dm2, dm2, x5, x9, r3
		vconv.bf16.fp32	 x5, cmh4
		vmac.f	dm1, dm1, x3, x7, r3
		nop
		vmac.f	dm2, dm2, x5, x9, r3
		jz	 r17, #.LBB0_1
		nop	                                //  Delay Slot 5
		nop	                                //  Delay Slot 4
		nop	                                //  Delay Slot 3
		nop	                                //  Delay Slot 2
		nop	                                //  Delay Slot 1
	// %bb.4:                               // %for.cond.cleanup
		vadd.f	dm0, dm1, dm2, r0
		nop
		ret	lr
		nop	                                //  Delay Slot 5
		nop	                                //  Delay Slot 4
		nop	                                //  Delay Slot 3
		vst.conv.bf16.fp32	 cml0, [p2, #0] //  Delay Slot 2
		nop	                                //  Delay Slot 1
	.Lfunc_end0:
		.size	layer_fused_gemv_bcast_kc256_bf16, .Lfunc_end0-layer_fused_gemv_bcast_kc256_bf16
	                                        // -- End function

	// No-op dummy symbol for MLIR linking (#131). Called once per token per tile
	// to force aiecc to link kc256_rr.o, but does no compute.
		.section	.text._ha_noop,"ax",@progbits
		.globl	_ha_noop
		.p2align	4
		.type	_ha_noop,@function
	_ha_noop:
		ret	lr
		nop
		nop
		nop
		nop
		nop
		nop
		.size	_ha_noop, .-_ha_noop

		.section	".linker-options","e",@llvm_linker_options
		.ident	"clang version 21.0.0 (https://github.com/Xilinx/llvm-aie 7bc5ade688821cdfd3af590b084ad055b0275c6e)"
		.section	".note.GNU-stack","",@progbits
		.addrsig
