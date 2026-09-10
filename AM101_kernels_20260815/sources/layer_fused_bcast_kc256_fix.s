	.file	"bcast_gemv.cpp"
	.section	.text.layer_fused_gemv_bcast_kc256_bf16,"ax",@progbits
	.globl	layer_fused_gemv_bcast_kc256_bf16 // -- Begin function layer_fused_gemv_bcast_kc256_bf16
	.p2align	4
	.type	layer_fused_gemv_bcast_kc256_bf16,@function
layer_fused_gemv_bcast_kc256_bf16:     // @layer_fused_gemv_bcast_kc256_bf16
// %bb.0:                               // %entry
	movxm	r0, #19201
	movxm	r1, #4096
	movxm	dj0, #4096
	nopx	;		mov	crrnd, #12
	mov	crunpacksize, #0
	mov	crupsmode, #0
	paddxm	 [sp], #128
	mova	r0, #0;		vbcst.16	 x0, r0
	vbcst.32	 x2, r0
	mov	s0, r0
	vconv.fp32.bf16	cml0, x0
	mova	r2, #16;		vmov	bmll1, x2
	mova	r4, #828;		movx	r3, #60;		vmov	bmlh1, x2
	mova	r6, #1;		vst	 bmll1, [sp, #-128];		movx	r5, #512;		vmov	cmh0, cml0 // 64-byte Folded Spill
	mova	r16, #0;		vst	 bmlh1, [sp, #-64];		movx	r7, #0;		vmov	cml2, cml0 // FIX AM-101 cml1: seed acc from cml0=0 (cml1 = résidu non-init)
	vmov	cml1, cml0 // FIX AM-101 (copie _fix) : seed cml1/cml0=0 — cml1 est LU dans la boucle (L48 vconv cml1) sans être seedé → état résiduel → NaN head4
	vmov	cmh1, cml0 // FIX AM-101 (copie _fix) : seed cmh1/cml0=0 — idem (L49 vconv cmh1)
.LBB0_1:                                // %for.body
                                        // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_2 Depth 2
	nopa	;		nopb	;		lshl	 r17, r7, r6
	vlda	 bmll1, [sp, #-128];		mov	dj1, r17 // 64-byte Folded Reload
	vlda	 bmlh1, [sp, #-64];		vldb	 x0, [p1, dj1];		mov	r18, r16 // 64-byte Folded Reload
	movs	dj1, r18;		add	r18, r18, #32
	vldb.unpack	 x1, unpacksign1, [p0, dj1]
	nop	
	mova	r19, #0
	add	r20, r19, #1
	nop	
	vmov	cml3, cml0 // FIX AM-101 cml1: seed acc from cml0=0
	movs	dj1, r18;		add	r18, r18, #32;		vmov	cml4, cml0 // FIX AM-101 cml1: seed acc from cml0=0
	vldb.unpack	 x1, unpacksign1, [p0, dj1];		vups.4x	dm1, x1, s0, upssign1;		vadd	dm1, dm1, dm0, r0
	add.nc	lc, r2, #-2
	add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm1, dm1, dm0, r3
	add	r20, r19, #1;		vextbcst.16	 x9, x0, r20
	movxm	ls, #.LBB0_2
	nopa	;		nopb	;		nops	;		movxm	le, #.L_LEnd0;		nopv	
.LBB0_2:                                // %_ZNK3aie15vector_elem_refI8bfloat16Lj32EEcvS1_Ev.exit
                                        //   Parent Loop BB0_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	nopa	;		nopb	;		movs	dj1, r18;		add	r18, r18, #32;		nopm	;		nopv	
	nopa	;		vldb.unpack	 x1, unpacksign1, [p0, dj1];		nops	;		nopx	;		vups.4x	dm1, x1, s0, upssign1;		vadd	dm1, dm1, dm0, r0
	nopa	;		nopb	;		vconv.bf16.fp32	 x3, cml1;		nopxm	;		nopv	
	nopa	;		nopb	;		vconv.bf16.fp32	 x5, cmh1;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm1, dm1, dm0, r3
	nopa	;		nopb	;		nops	;		add	r20, r19, #1;		vextbcst.16	 x9, x0, r20;		vmac.f	dm3, dm3, x3, x7, r4
	nopa	;		nopb	;		nops	;		nopxm	;		vmac.f	dm4, dm4, x5, x9, r4
.L_LEnd0:
	nopa	;		nopb	;		nops	;		nopxm	;		nopv	
// %bb.3:                               // %for.cond.cleanup4
                                        //   in Loop: Header=BB0_1 Depth=1
	movs	p3, p0;		mov	m0, r17
	padda	 [p3], m0;		vups.4x	dm1, x1, s0, upssign1;		vadd	dm1, dm1, dm0, r0
	vldb	 x0, [p3, dj0];		vconv.bf16.fp32	 x3, cml1
	vconv.bf16.fp32	 x5, cmh1;		add	r19, r19, #2;		vextbcst.16	 x7, x0, r19;		vsub.f	dm1, dm1, dm0, r3
	add	 r16, r16, r5;		vextbcst.16	 x9, x0, r20;		vmac.f	dm3, dm3, x3, x7, r4
	eq	 r17, r16, r1;		vmac.f	dm4, dm4, x5, x9, r4
	add	r7, r7, #32
	nop	
	nop	
	vconv.bf16.fp32	 x3, cml1
	vconv.bf16.fp32	 x5, cmh1
	vmac.f	dm3, dm3, x3, x7, r4
	vmac.f	dm4, dm4, x5, x9, r4
	nop	
	nop	
	vadd.f	dm3, dm3, dm4, r3
	nop	
	nop	
	nop	
	nop	
	nop	
	vconv.bf16.fp32	 x2, cml3
	nop	
	vmac.f	dm2, dm2, x2, x0, r4
	jz	 r17, #.LBB0_1
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	nop	                                //  Delay Slot 2
	nop	                                //  Delay Slot 1
// %bb.4:                               // %for.cond.cleanup
	ret	lr
	nop	                                //  Delay Slot 5
	nop	                                //  Delay Slot 4
	nop	                                //  Delay Slot 3
	vst.conv.bf16.fp32	 cml2, [p2, #0];		paddxm	 [sp], #-128 //  Delay Slot 2
	nop	                                //  Delay Slot 1
.Lfunc_end0:
	.size	layer_fused_gemv_bcast_kc256_bf16, .Lfunc_end0-layer_fused_gemv_bcast_kc256_bf16
                                        // -- End function

// No-op dummy symbol for MLIR linking (#131). Called once per token per tile
// to force aiecc to link kc256.o, but does no compute.
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
