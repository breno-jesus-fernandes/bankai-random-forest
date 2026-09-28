// Extracted from cargo rustc --release --lib -- --emit=asm
// RUSTFLAGS=-C target-cpu=native; function includes inlined histogram subtraction.
__RNvXs8_NtCsghSPhPCBQnG_20bankai_random_forest5denseNtB5_10DenseInputNtNtCsiE2AcdbpEWu_3xrf7rfinput7RfInput20split_cache_children:
	.cfi_startproc
	sub	sp, sp, #176
	stp	x20, x19, [sp, #144]
	stp	x29, x30, [sp, #160]
	add	x29, sp, #160
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset w19, -24
	.cfi_offset w20, -32
	mov	x20, x1
	mov	x19, x8
	ldr	x8, [x1]
	cmn	x8, #1
	b.eq	LBB191_8
	mov	x1, x0
	ldr	x8, [x3, #16]
	ldr	x9, [x4, #16]
	cmp	x8, x9
	b.ls	LBB191_9
	ldr	x2, [x4, #8]
	add	x0, sp, #72
	mov	x3, x9
	bl	__RNvNtCsghSPhPCBQnG_20bankai_random_forest5dense16build_histograms
	ldr	x8, [x20, #40]
	ldr	x9, [sp, #112]
	cmp	x9, x8
	csel	x8, x9, x8, lo
	cbz	x8, LBB191_18
	ldr	x9, [x20, #32]
	ldr	x10, [sp, #104]
	cmp	x8, #8
	b.lo	LBB191_15
	lsl	x11, x8, #3
	add	x12, x9, x11
	add	x11, x10, x11
	cmp	x9, x11
	ccmp	x10, x12, #2, lo
	b.lo	LBB191_15
	and	x11, x8, #0xfffffffffffffff8
	add	x12, x9, #32
	add	x13, x10, #32
	movi.2d	v0, #0000000000000000
	and	x14, x8, #0xfffffffffffffff8
LBB191_6:
	ldp	q1, q2, [x12, #-32]
	ldp	q3, q4, [x12]
	ldp	q5, q6, [x13, #-32]
	ldp	q7, q16, [x13], #64
	fsub.2d	v1, v1, v5
	fsub.2d	v2, v2, v6
	fsub.2d	v3, v3, v7
	fsub.2d	v4, v4, v16
	fmaxnm.2d	v1, v1, v0
	fmaxnm.2d	v2, v2, v0
	fmaxnm.2d	v3, v3, v0
	fmaxnm.2d	v4, v4, v0
	stp	q1, q2, [x12, #-32]
	stp	q3, q4, [x12], #64
	subs	x14, x14, #8
	b.ne	LBB191_6
	cmp	x8, x11
	b.ne	LBB191_16
	b	LBB191_18
LBB191_8:
	str	x8, [x19]
	str	x8, [x19, #72]
	ldp	x29, x30, [sp, #160]
	ldp	x20, x19, [sp, #144]
	add	sp, sp, #176
	ret
LBB191_9:
	ldr	x2, [x3, #8]
	mov	x0, sp
	mov	x3, x8
	bl	__RNvNtCsghSPhPCBQnG_20bankai_random_forest5dense16build_histograms
	ldr	x8, [x20, #40]
	ldr	x9, [sp, #40]
	cmp	x9, x8
	csel	x8, x9, x8, lo
	cbz	x8, LBB191_27
	ldr	x9, [x20, #32]
	ldr	x10, [sp, #32]
	cmp	x8, #8
	b.lo	LBB191_24
	lsl	x11, x8, #3
	add	x12, x9, x11
	add	x11, x10, x11
	cmp	x9, x11
	ccmp	x10, x12, #2, lo
	b.lo	LBB191_24
	and	x11, x8, #0xfffffffffffffff8
	add	x12, x9, #32
	add	x13, x10, #32
	movi.2d	v0, #0000000000000000
	and	x14, x8, #0xfffffffffffffff8
LBB191_13:
	ldp	q1, q2, [x12, #-32]
	ldp	q3, q4, [x12]
	ldp	q5, q6, [x13, #-32]
	ldp	q7, q16, [x13], #64
	fsub.2d	v1, v1, v5
	fsub.2d	v2, v2, v6
	fsub.2d	v3, v3, v7
	fsub.2d	v4, v4, v16
	fmaxnm.2d	v1, v1, v0
	fmaxnm.2d	v2, v2, v0
	fmaxnm.2d	v3, v3, v0
	fmaxnm.2d	v4, v4, v0
	stp	q1, q2, [x12, #-32]
	stp	q3, q4, [x12], #64
	subs	x14, x14, #8
	b.ne	LBB191_13
	cmp	x8, x11
	b.ne	LBB191_25
	b	LBB191_27
LBB191_15:
	mov	x11, #0
LBB191_16:
	lsl	x12, x11, #3
	add	x10, x10, x12
	add	x9, x9, x12
	sub	x8, x8, x11
	movi.2d	v0, #0000000000000000
LBB191_17:
	ldr	d1, [x10], #8
	ldr	d2, [x9]
	fsub	d1, d2, d1
	fmaxnm	d1, d1, d0
	str	d1, [x9], #8
	subs	x8, x8, #1
	b.ne	LBB191_17
LBB191_18:
	ldr	x8, [x20, #64]
	ldr	x9, [sp, #136]
	cmp	x9, x8
	csel	x8, x9, x8, lo
	cbz	x8, LBB191_36
	ldr	x9, [x20, #56]
	ldr	x10, [sp, #128]
	cmp	x8, #8
	b.lo	LBB191_33
	lsl	x11, x8, #3
	add	x12, x9, x11
	add	x11, x10, x11
	cmp	x9, x11
	ccmp	x10, x12, #2, lo
	b.lo	LBB191_33
	and	x11, x8, #0xfffffffffffffff8
	add	x12, x9, #32
	add	x13, x10, #32
	and	x14, x8, #0xfffffffffffffff8
LBB191_22:
	ldp	q0, q1, [x13, #-32]
	ldp	q2, q3, [x13], #64
	ldp	q4, q5, [x12, #-32]
	ldp	q6, q7, [x12]
	sub.2d	v0, v4, v0
	sub.2d	v1, v5, v1
	sub.2d	v2, v6, v2
	sub.2d	v3, v7, v3
	stp	q0, q1, [x12, #-32]
	stp	q2, q3, [x12], #64
	subs	x14, x14, #8
	b.ne	LBB191_22
	cmp	x8, x11
	b.ne	LBB191_34
	b	LBB191_36
LBB191_24:
	mov	x11, #0
LBB191_25:
	lsl	x12, x11, #3
	add	x10, x10, x12
	add	x9, x9, x12
	sub	x8, x8, x11
	movi.2d	v0, #0000000000000000
LBB191_26:
	ldr	d1, [x10], #8
	ldr	d2, [x9]
	fsub	d1, d2, d1
	fmaxnm	d1, d1, d0
	str	d1, [x9], #8
	subs	x8, x8, #1
	b.ne	LBB191_26
LBB191_27:
	ldr	x8, [x20, #64]
	ldr	x9, [sp, #64]
	cmp	x9, x8
	csel	x8, x9, x8, lo
	cbz	x8, LBB191_40
	ldr	x9, [x20, #56]
	ldr	x10, [sp, #56]
	cmp	x8, #8
	b.lo	LBB191_37
	lsl	x11, x8, #3
	add	x12, x9, x11
	add	x11, x10, x11
	cmp	x9, x11
	ccmp	x10, x12, #2, lo
	b.lo	LBB191_37
	and	x11, x8, #0xfffffffffffffff8
	add	x12, x9, #32
	add	x13, x10, #32
	and	x14, x8, #0xfffffffffffffff8
LBB191_31:
	ldp	q0, q1, [x13, #-32]
	ldp	q2, q3, [x13], #64
	ldp	q4, q5, [x12, #-32]
	ldp	q6, q7, [x12]
	sub.2d	v0, v4, v0
	sub.2d	v1, v5, v1
	sub.2d	v2, v6, v2
	sub.2d	v3, v7, v3
	stp	q0, q1, [x12, #-32]
	stp	q2, q3, [x12], #64
	subs	x14, x14, #8
	b.ne	LBB191_31
	cmp	x8, x11
	b.ne	LBB191_38
	b	LBB191_40
LBB191_33:
	mov	x11, #0
LBB191_34:
	lsl	x12, x11, #3
	add	x10, x10, x12
	add	x9, x9, x12
	sub	x8, x8, x11
LBB191_35:
	ldr	x11, [x10], #8
	ldr	x12, [x9]
	sub	x11, x12, x11
	str	x11, [x9], #8
	subs	x8, x8, #1
	b.ne	LBB191_35
LBB191_36:
	ldp	q0, q1, [x20, #32]
	stp	q0, q1, [x19, #32]
	ldp	q1, q0, [x20]
	stp	q1, q0, [x19]
	ldur	q0, [sp, #88]
	ldur	q1, [sp, #72]
	stur	q0, [x19, #88]
	ldur	q0, [sp, #104]
	ldur	q2, [sp, #120]
	stur	q0, [x19, #104]
	stur	q2, [x19, #120]
	ldr	x8, [x20, #64]
	str	x8, [x19, #64]
	ldr	x8, [sp, #136]
	str	x8, [x19, #136]
	stur	q1, [x19, #72]
	ldp	x29, x30, [sp, #160]
	ldp	x20, x19, [sp, #144]
	add	sp, sp, #176
	ret
LBB191_37:
	mov	x11, #0
LBB191_38:
	lsl	x12, x11, #3
	add	x10, x10, x12
	add	x9, x9, x12
	sub	x8, x8, x11
LBB191_39:
	ldr	x11, [x10], #8
	ldr	x12, [x9]
	sub	x11, x12, x11
	str	x11, [x9], #8
	subs	x8, x8, #1
	b.ne	LBB191_39
LBB191_40:
	ldp	q0, q1, [sp, #32]
	stp	q0, q1, [x19, #32]
	ldp	q1, q0, [sp]
	stp	q1, q0, [x19]
	ldp	q1, q0, [x20]
	stur	q0, [x19, #88]
	ldp	q0, q2, [x20, #32]
	stur	q0, [x19, #104]
	stur	q2, [x19, #120]
	ldr	x8, [sp, #64]
	str	x8, [x19, #64]
	ldr	x8, [x20, #64]
	str	x8, [x19, #136]
	stur	q1, [x19, #72]
	ldp	x29, x30, [sp, #160]
	ldp	x20, x19, [sp, #144]
	add	sp, sp, #176
	ret
	.cfi_endproc
