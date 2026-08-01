//! Eight-lane NEON kernels for VP9's 8- and 16-tap loop filters.
//!
//! The arithmetic follows `vpx_dsp/arm/loopfilter_neon.c`. Each lane is one
//! transverse edge position. The Rust wrapper keeps the scalar slice API and
//! validates the complete 16-pixel neighbourhood before entering NEON code.

use core::arch::aarch64::*;

#[inline(always)]
unsafe fn filter_masks(
    limit: u8,
    blimit: u8,
    thresh: u8,
    values: &[uint8x8_t; 16],
) -> (uint8x8_t, uint8x8_t, uint8x8_t, uint8x8_t) {
    let p7 = values[0];
    let p6 = values[1];
    let p5 = values[2];
    let p4 = values[3];
    let p3 = values[4];
    let p2 = values[5];
    let p1 = values[6];
    let p0 = values[7];
    let q0 = values[8];
    let q1 = values[9];
    let q2 = values[10];
    let q3 = values[11];
    let q4 = values[12];
    let q5 = values[13];
    let q6 = values[14];
    let q7 = values[15];

    let inner_diff = vmax_u8(vabd_u8(p1, p0), vabd_u8(q1, q0));
    let mut max_adjacent = inner_diff;
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(p3, p2));
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(p2, p1));
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(q2, q1));
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(q3, q2));
    let adjacent_ok = vcle_u8(max_adjacent, vdup_n_u8(limit));

    // Widen this expression to match the scalar reference exactly. A
    // saturating u8 approximation differs when blimit == 255.
    let p0q0 = vabdl_u8(p0, q0);
    let p1q1 = vabdl_u8(p1, q1);
    let boundary = vaddq_u16(vshlq_n_u16(p0q0, 1), vshrq_n_u16(p1q1, 1));
    let boundary_ok = vmovn_u16(vcleq_u16(boundary, vdupq_n_u16(u16::from(blimit))));
    let mask = vand_u8(adjacent_ok, boundary_ok);
    let hev = vcgt_u8(inner_diff, vdup_n_u8(thresh));

    let mut flat_diff = inner_diff;
    flat_diff = vmax_u8(flat_diff, vabd_u8(p2, p0));
    flat_diff = vmax_u8(flat_diff, vabd_u8(q2, q0));
    flat_diff = vmax_u8(flat_diff, vabd_u8(p3, p0));
    flat_diff = vmax_u8(flat_diff, vabd_u8(q3, q0));
    let flat = vcle_u8(flat_diff, vdup_n_u8(1));

    let mut flat2_diff = vabd_u8(p4, p0);
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(p5, p0));
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(p6, p0));
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(p7, p0));
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(q4, q0));
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(q5, q0));
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(q6, q0));
    flat2_diff = vmax_u8(flat2_diff, vabd_u8(q7, q0));
    let flat2 = vcle_u8(flat2_diff, vdup_n_u8(1));
    (mask, flat, flat2, hev)
}

#[inline(always)]
unsafe fn filter4(
    mask: uint8x8_t,
    hev: uint8x8_t,
    p1: uint8x8_t,
    p0: uint8x8_t,
    q0: uint8x8_t,
    q1: uint8x8_t,
) -> [uint8x8_t; 4] {
    let sign = vdup_n_u8(0x80);
    let mut ps1 = vreinterpret_s8_u8(veor_u8(p1, sign));
    let mut ps0 = vreinterpret_s8_u8(veor_u8(p0, sign));
    let mut qs0 = vreinterpret_s8_u8(veor_u8(q0, sign));
    let mut qs1 = vreinterpret_s8_u8(veor_u8(q1, sign));

    let mut filter = vqsub_s8(ps1, qs1);
    filter = vand_s8(filter, vreinterpret_s8_u8(hev));
    let inner = vqsub_s8(qs0, ps0);
    filter = vqadd_s8(filter, inner);
    filter = vqadd_s8(filter, inner);
    filter = vqadd_s8(filter, inner);
    filter = vand_s8(filter, vreinterpret_s8_u8(mask));

    let filter1 = vshr_n_s8(vqadd_s8(filter, vdup_n_s8(4)), 3);
    let filter2 = vshr_n_s8(vqadd_s8(filter, vdup_n_s8(3)), 3);
    qs0 = vqsub_s8(qs0, filter1);
    ps0 = vqadd_s8(ps0, filter2);

    let outer = vbic_s8(vrshr_n_s8(filter1, 1), vreinterpret_s8_u8(hev));
    qs1 = vqsub_s8(qs1, outer);
    ps1 = vqadd_s8(ps1, outer);

    [
        veor_u8(vreinterpret_u8_s8(ps1), sign),
        veor_u8(vreinterpret_u8_s8(ps0), sign),
        veor_u8(vreinterpret_u8_s8(qs0), sign),
        veor_u8(vreinterpret_u8_s8(qs1), sign),
    ]
}

#[inline(always)]
unsafe fn update_sum(
    mut sum: uint16x8_t,
    sub0: uint8x8_t,
    sub1: uint8x8_t,
    add0: uint8x8_t,
    add1: uint8x8_t,
) -> uint16x8_t {
    sum = vsubw_u8(sum, sub0);
    sum = vsubw_u8(sum, sub1);
    sum = vaddw_u8(sum, add0);
    vaddw_u8(sum, add1)
}

#[inline(always)]
unsafe fn filter7(values: &[uint8x8_t; 8]) -> [uint8x8_t; 6] {
    let p3 = values[0];
    let p2 = values[1];
    let p1 = values[2];
    let p0 = values[3];
    let q0 = values[4];
    let q1 = values[5];
    let q2 = values[6];
    let q3 = values[7];

    let mut sum = vaddl_u8(p3, p3);
    sum = vaddw_u8(sum, p3);
    sum = vaddw_u8(sum, p2);
    sum = vaddw_u8(sum, p2);
    sum = vaddw_u8(sum, p1);
    sum = vaddw_u8(sum, p0);
    sum = vaddw_u8(sum, q0);
    let op2 = vrshrn_n_u16(sum, 3);
    sum = update_sum(sum, p3, p2, p1, q1);
    let op1 = vrshrn_n_u16(sum, 3);
    sum = update_sum(sum, p3, p1, p0, q2);
    let op0 = vrshrn_n_u16(sum, 3);
    sum = update_sum(sum, p3, p0, q0, q3);
    let oq0 = vrshrn_n_u16(sum, 3);
    sum = update_sum(sum, p2, q0, q1, q3);
    let oq1 = vrshrn_n_u16(sum, 3);
    sum = update_sum(sum, p1, q1, q2, q3);
    let oq2 = vrshrn_n_u16(sum, 3);
    [op2, op1, op0, oq0, oq1, oq2]
}

#[inline(always)]
unsafe fn filter8_lanes(
    values: [uint8x8_t; 8],
    limit: u8,
    blimit: u8,
    thresh: u8,
) -> [uint8x8_t; 8] {
    let p3 = values[0];
    let p2 = values[1];
    let p1 = values[2];
    let p0 = values[3];
    let q0 = values[4];
    let q1 = values[5];
    let q2 = values[6];
    let q3 = values[7];

    let inner_diff = vmax_u8(vabd_u8(p1, p0), vabd_u8(q1, q0));
    let mut max_adjacent = inner_diff;
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(p3, p2));
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(p2, p1));
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(q2, q1));
    max_adjacent = vmax_u8(max_adjacent, vabd_u8(q3, q2));
    let adjacent_ok = vcle_u8(max_adjacent, vdup_n_u8(limit));

    // The scalar oracle widens this expression before comparing it with
    // `blimit`; keeping it in u8 would saturate and differ at blimit 255.
    let p0q0 = vabdl_u8(p0, q0);
    let p1q1 = vabdl_u8(p1, q1);
    let boundary = vaddq_u16(vshlq_n_u16(p0q0, 1), vshrq_n_u16(p1q1, 1));
    let boundary_ok = vmovn_u16(vcleq_u16(boundary, vdupq_n_u16(u16::from(blimit))));
    let mask = vand_u8(adjacent_ok, boundary_ok);
    let hev = vcgt_u8(inner_diff, vdup_n_u8(thresh));

    let mut flat_diff = inner_diff;
    flat_diff = vmax_u8(flat_diff, vabd_u8(p2, p0));
    flat_diff = vmax_u8(flat_diff, vabd_u8(q2, q0));
    flat_diff = vmax_u8(flat_diff, vabd_u8(p3, p0));
    flat_diff = vmax_u8(flat_diff, vabd_u8(q3, q0));
    let flat = vand_u8(mask, vcle_u8(flat_diff, vdup_n_u8(1)));

    let mut output = values;
    let narrow = filter4(mask, hev, p1, p0, q0, q1);
    output[2] = narrow[0];
    output[3] = narrow[1];
    output[4] = narrow[2];
    output[5] = narrow[3];

    if vmaxv_u8(flat) != 0 {
        let wide = filter7(&values);
        for index in 0..6 {
            output[index + 1] = vbsl_u8(flat, wide[index], output[index + 1]);
        }
    }
    output
}

#[inline(always)]
unsafe fn filter15(values: &[uint8x8_t; 16]) -> [uint8x8_t; 14] {
    let p7 = values[0];
    let p6 = values[1];
    let p5 = values[2];
    let p4 = values[3];
    let p3 = values[4];
    let p2 = values[5];
    let p1 = values[6];
    let p0 = values[7];
    let q0 = values[8];
    let q1 = values[9];
    let q2 = values[10];
    let q3 = values[11];
    let q4 = values[12];
    let q5 = values[13];
    let q6 = values[14];
    let q7 = values[15];

    let mut sum = vshll_n_u8(p7, 3);
    sum = vsubw_u8(sum, p7);
    sum = vaddw_u8(sum, p6);
    sum = vaddw_u8(sum, p6);
    sum = vaddw_u8(sum, p5);
    sum = vaddw_u8(sum, p4);
    sum = vaddw_u8(sum, p3);
    sum = vaddw_u8(sum, p2);
    sum = vaddw_u8(sum, p1);
    sum = vaddw_u8(sum, p0);
    sum = vaddw_u8(sum, q0);

    let mut output = [vdup_n_u8(0); 14];
    output[0] = vrshrn_n_u16(sum, 4);
    let updates = [
        (p7, p6, p5, q1),
        (p7, p5, p4, q2),
        (p7, p4, p3, q3),
        (p7, p3, p2, q4),
        (p7, p2, p1, q5),
        (p7, p1, p0, q6),
        (p7, p0, q0, q7),
        (p6, q0, q1, q7),
        (p5, q1, q2, q7),
        (p4, q2, q3, q7),
        (p3, q3, q4, q7),
        (p2, q4, q5, q7),
        (p1, q5, q6, q7),
    ];
    for (index, &(sub0, sub1, add0, add1)) in updates.iter().enumerate() {
        sum = update_sum(sum, sub0, sub1, add0, add1);
        output[index + 1] = vrshrn_n_u16(sum, 4);
    }
    output
}

#[inline(always)]
unsafe fn filter16_lanes(
    values: [uint8x8_t; 16],
    limit: u8,
    blimit: u8,
    thresh: u8,
) -> [uint8x8_t; 16] {
    let (mask, flat, flat2, hev) = filter_masks(limit, blimit, thresh, &values);
    let mut output = values;

    let narrow = filter4(mask, hev, values[6], values[7], values[8], values[9]);
    output[6] = narrow[0];
    output[7] = narrow[1];
    output[8] = narrow[2];
    output[9] = narrow[3];

    let wide7_mask = vand_u8(mask, flat);
    if vmaxv_u8(wide7_mask) == 0 {
        return output;
    }
    let wide7 = filter7(&[
        values[4], values[5], values[6], values[7], values[8], values[9], values[10], values[11],
    ]);
    for index in 0..6 {
        output[index + 5] = vbsl_u8(wide7_mask, wide7[index], output[index + 5]);
    }

    let wide15_mask = vand_u8(wide7_mask, flat2);
    let wide15 = filter15(&values);
    for index in 0..14 {
        output[index + 1] = vbsl_u8(wide15_mask, wide15[index], output[index + 1]);
    }
    output
}

#[inline(always)]
unsafe fn transpose_8x8(rows: [uint8x8_t; 8]) -> [uint8x8_t; 8] {
    let b00 = vtrn1_u8(rows[0], rows[1]);
    let b01 = vtrn2_u8(rows[0], rows[1]);
    let b10 = vtrn1_u8(rows[2], rows[3]);
    let b11 = vtrn2_u8(rows[2], rows[3]);
    let b20 = vtrn1_u8(rows[4], rows[5]);
    let b21 = vtrn2_u8(rows[4], rows[5]);
    let b30 = vtrn1_u8(rows[6], rows[7]);
    let b31 = vtrn2_u8(rows[6], rows[7]);

    let c00 = vtrn1_u16(vreinterpret_u16_u8(b00), vreinterpret_u16_u8(b10));
    let c01 = vtrn2_u16(vreinterpret_u16_u8(b00), vreinterpret_u16_u8(b10));
    let c10 = vtrn1_u16(vreinterpret_u16_u8(b01), vreinterpret_u16_u8(b11));
    let c11 = vtrn2_u16(vreinterpret_u16_u8(b01), vreinterpret_u16_u8(b11));
    let c20 = vtrn1_u16(vreinterpret_u16_u8(b20), vreinterpret_u16_u8(b30));
    let c21 = vtrn2_u16(vreinterpret_u16_u8(b20), vreinterpret_u16_u8(b30));
    let c30 = vtrn1_u16(vreinterpret_u16_u8(b21), vreinterpret_u16_u8(b31));
    let c31 = vtrn2_u16(vreinterpret_u16_u8(b21), vreinterpret_u16_u8(b31));

    let d00 = vreinterpret_u8_u32(vtrn1_u32(
        vreinterpret_u32_u16(c00),
        vreinterpret_u32_u16(c20),
    ));
    let d01 = vreinterpret_u8_u32(vtrn2_u32(
        vreinterpret_u32_u16(c00),
        vreinterpret_u32_u16(c20),
    ));
    let d10 = vreinterpret_u8_u32(vtrn1_u32(
        vreinterpret_u32_u16(c01),
        vreinterpret_u32_u16(c21),
    ));
    let d11 = vreinterpret_u8_u32(vtrn2_u32(
        vreinterpret_u32_u16(c01),
        vreinterpret_u32_u16(c21),
    ));
    let d20 = vreinterpret_u8_u32(vtrn1_u32(
        vreinterpret_u32_u16(c10),
        vreinterpret_u32_u16(c30),
    ));
    let d21 = vreinterpret_u8_u32(vtrn2_u32(
        vreinterpret_u32_u16(c10),
        vreinterpret_u32_u16(c30),
    ));
    let d30 = vreinterpret_u8_u32(vtrn1_u32(
        vreinterpret_u32_u16(c11),
        vreinterpret_u32_u16(c31),
    ));
    let d31 = vreinterpret_u8_u32(vtrn2_u32(
        vreinterpret_u32_u16(c11),
        vreinterpret_u32_u16(c31),
    ));

    [d00, d20, d10, d30, d01, d21, d11, d31]
}

#[inline(always)]
unsafe fn transpose_16x8(rows: [uint8x16_t; 8]) -> [uint8x8_t; 16] {
    let b00 = vtrn1q_u8(rows[0], rows[1]);
    let b01 = vtrn2q_u8(rows[0], rows[1]);
    let b10 = vtrn1q_u8(rows[2], rows[3]);
    let b11 = vtrn2q_u8(rows[2], rows[3]);
    let b20 = vtrn1q_u8(rows[4], rows[5]);
    let b21 = vtrn2q_u8(rows[4], rows[5]);
    let b30 = vtrn1q_u8(rows[6], rows[7]);
    let b31 = vtrn2q_u8(rows[6], rows[7]);

    let c00 = vtrn1q_u16(vreinterpretq_u16_u8(b00), vreinterpretq_u16_u8(b10));
    let c01 = vtrn2q_u16(vreinterpretq_u16_u8(b00), vreinterpretq_u16_u8(b10));
    let c10 = vtrn1q_u16(vreinterpretq_u16_u8(b01), vreinterpretq_u16_u8(b11));
    let c11 = vtrn2q_u16(vreinterpretq_u16_u8(b01), vreinterpretq_u16_u8(b11));
    let c20 = vtrn1q_u16(vreinterpretq_u16_u8(b20), vreinterpretq_u16_u8(b30));
    let c21 = vtrn2q_u16(vreinterpretq_u16_u8(b20), vreinterpretq_u16_u8(b30));
    let c30 = vtrn1q_u16(vreinterpretq_u16_u8(b21), vreinterpretq_u16_u8(b31));
    let c31 = vtrn2q_u16(vreinterpretq_u16_u8(b21), vreinterpretq_u16_u8(b31));

    let d00 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c00),
        vreinterpretq_u32_u16(c20),
    ));
    let d01 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c00),
        vreinterpretq_u32_u16(c20),
    ));
    let d10 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c01),
        vreinterpretq_u32_u16(c21),
    ));
    let d11 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c01),
        vreinterpretq_u32_u16(c21),
    ));
    let d20 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c10),
        vreinterpretq_u32_u16(c30),
    ));
    let d21 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c10),
        vreinterpretq_u32_u16(c30),
    ));
    let d30 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c11),
        vreinterpretq_u32_u16(c31),
    ));
    let d31 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c11),
        vreinterpretq_u32_u16(c31),
    ));

    [
        vget_low_u8(d00),
        vget_low_u8(d20),
        vget_low_u8(d10),
        vget_low_u8(d30),
        vget_low_u8(d01),
        vget_low_u8(d21),
        vget_low_u8(d11),
        vget_low_u8(d31),
        vget_high_u8(d00),
        vget_high_u8(d20),
        vget_high_u8(d10),
        vget_high_u8(d30),
        vget_high_u8(d01),
        vget_high_u8(d21),
        vget_high_u8(d11),
        vget_high_u8(d31),
    ]
}

#[inline(always)]
unsafe fn transpose_8x16(columns: [uint8x8_t; 16]) -> [uint8x16_t; 8] {
    let a0 = vcombine_u8(columns[0], columns[8]);
    let a1 = vcombine_u8(columns[1], columns[9]);
    let a2 = vcombine_u8(columns[2], columns[10]);
    let a3 = vcombine_u8(columns[3], columns[11]);
    let a4 = vcombine_u8(columns[4], columns[12]);
    let a5 = vcombine_u8(columns[5], columns[13]);
    let a6 = vcombine_u8(columns[6], columns[14]);
    let a7 = vcombine_u8(columns[7], columns[15]);

    let b00 = vtrn1q_u8(a0, a1);
    let b01 = vtrn2q_u8(a0, a1);
    let b10 = vtrn1q_u8(a2, a3);
    let b11 = vtrn2q_u8(a2, a3);
    let b20 = vtrn1q_u8(a4, a5);
    let b21 = vtrn2q_u8(a4, a5);
    let b30 = vtrn1q_u8(a6, a7);
    let b31 = vtrn2q_u8(a6, a7);

    let c00 = vtrn1q_u16(vreinterpretq_u16_u8(b00), vreinterpretq_u16_u8(b10));
    let c01 = vtrn2q_u16(vreinterpretq_u16_u8(b00), vreinterpretq_u16_u8(b10));
    let c10 = vtrn1q_u16(vreinterpretq_u16_u8(b01), vreinterpretq_u16_u8(b11));
    let c11 = vtrn2q_u16(vreinterpretq_u16_u8(b01), vreinterpretq_u16_u8(b11));
    let c20 = vtrn1q_u16(vreinterpretq_u16_u8(b20), vreinterpretq_u16_u8(b30));
    let c21 = vtrn2q_u16(vreinterpretq_u16_u8(b20), vreinterpretq_u16_u8(b30));
    let c30 = vtrn1q_u16(vreinterpretq_u16_u8(b21), vreinterpretq_u16_u8(b31));
    let c31 = vtrn2q_u16(vreinterpretq_u16_u8(b21), vreinterpretq_u16_u8(b31));

    let d00 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c00),
        vreinterpretq_u32_u16(c20),
    ));
    let d01 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c00),
        vreinterpretq_u32_u16(c20),
    ));
    let d10 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c01),
        vreinterpretq_u32_u16(c21),
    ));
    let d11 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c01),
        vreinterpretq_u32_u16(c21),
    ));
    let d20 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c10),
        vreinterpretq_u32_u16(c30),
    ));
    let d21 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c10),
        vreinterpretq_u32_u16(c30),
    ));
    let d30 = vreinterpretq_u8_u32(vtrn1q_u32(
        vreinterpretq_u32_u16(c11),
        vreinterpretq_u32_u16(c31),
    ));
    let d31 = vreinterpretq_u8_u32(vtrn2q_u32(
        vreinterpretq_u32_u16(c11),
        vreinterpretq_u32_u16(c31),
    ));

    [d00, d20, d10, d30, d01, d21, d11, d31]
}

#[target_feature(enable = "neon")]
unsafe fn horizontal_8_inner(
    pixels: *mut u8,
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    let start = pixels.add(s_index - 4 * pitch);
    let mut values = [vdup_n_u8(0); 8];
    for (row, value) in values.iter_mut().enumerate() {
        *value = vld1_u8(start.add(row * pitch));
    }
    let output = filter8_lanes(values, limit, blimit, thresh);
    for (row, value) in output.iter().enumerate().take(7).skip(1) {
        vst1_u8(start.add(row * pitch), *value);
    }
}

#[target_feature(enable = "neon")]
unsafe fn vertical_8_inner(
    pixels: *mut u8,
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    let start = pixels.add(s_index - 4);
    let mut rows = [vdup_n_u8(0); 8];
    for (row, value) in rows.iter_mut().enumerate() {
        *value = vld1_u8(start.add(row * pitch));
    }
    let output = transpose_8x8(filter8_lanes(transpose_8x8(rows), limit, blimit, thresh));
    for (row, value) in output.iter().enumerate() {
        vst1_u8(start.add(row * pitch), *value);
    }
}

/// Apply the 8-tap loop filter across eight horizontal edge columns.
pub fn lpf_horizontal_8_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    assert!(pitch >= 8);
    assert!(s_index >= 4 * pitch);
    assert!(s_index + 3 * pitch + 7 < pixels.len());
    // SAFETY: AArch64 always provides NEON. The assertions prove that all
    // eight 8-byte row loads and the six filtered row stores stay in `pixels`.
    unsafe { horizontal_8_inner(pixels.as_mut_ptr(), s_index, pitch, blimit, limit, thresh) }
}

/// Apply two adjacent 8-tap horizontal filters with independent thresholds.
#[allow(clippy::too_many_arguments)]
pub fn lpf_horizontal_8_dual_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    lpf_horizontal_8_neon(pixels, s_index, pitch, blimit0, limit0, thresh0);
    lpf_horizontal_8_neon(pixels, s_index + 8, pitch, blimit1, limit1, thresh1);
}

/// Apply the 8-tap loop filter down eight vertical edge rows.
pub fn lpf_vertical_8_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    assert!(pitch >= 8);
    assert!(s_index >= 4);
    assert!(s_index + 7 * pitch + 3 < pixels.len());
    // SAFETY: AArch64 always provides NEON. The assertions prove that all
    // eight row loads and stores stay in `pixels`.
    unsafe { vertical_8_inner(pixels.as_mut_ptr(), s_index, pitch, blimit, limit, thresh) }
}

/// Apply two vertically adjacent 8-tap filters with independent thresholds.
#[allow(clippy::too_many_arguments)]
pub fn lpf_vertical_8_dual_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit0: u8,
    limit0: u8,
    thresh0: u8,
    blimit1: u8,
    limit1: u8,
    thresh1: u8,
) {
    lpf_vertical_8_neon(pixels, s_index, pitch, blimit0, limit0, thresh0);
    lpf_vertical_8_neon(pixels, s_index + 8 * pitch, pitch, blimit1, limit1, thresh1);
}

#[target_feature(enable = "neon")]
unsafe fn horizontal_inner(
    pixels: *mut u8,
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    let start = pixels.add(s_index - 8 * pitch);
    let mut values = [vdup_n_u8(0); 16];
    for (row, value) in values.iter_mut().enumerate() {
        *value = vld1_u8(start.add(row * pitch));
    }
    let output = filter16_lanes(values, limit, blimit, thresh);
    for (row, value) in output.iter().enumerate().take(15).skip(1) {
        vst1_u8(start.add(row * pitch), *value);
    }
}

#[target_feature(enable = "neon")]
unsafe fn vertical_inner(
    pixels: *mut u8,
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    let start = pixels.add(s_index - 8);
    let mut rows = [vdupq_n_u8(0); 8];
    for (row, value) in rows.iter_mut().enumerate() {
        *value = vld1q_u8(start.add(row * pitch));
    }
    let output = transpose_8x16(filter16_lanes(transpose_16x8(rows), limit, blimit, thresh));
    for (row, value) in output.iter().enumerate() {
        vst1q_u8(start.add(row * pitch), *value);
    }
}

/// Apply the 16-tap loop filter across eight horizontal edge columns.
pub fn lpf_horizontal_16_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    assert!(pitch >= 8);
    assert!(s_index >= 8 * pitch);
    assert!(s_index + 7 * pitch + 7 < pixels.len());
    // SAFETY: AArch64 always provides NEON. The assertions prove that all
    // sixteen 8-byte row loads and the fourteen row stores stay in `pixels`.
    unsafe { horizontal_inner(pixels.as_mut_ptr(), s_index, pitch, blimit, limit, thresh) }
}

/// Apply two adjacent 16-tap horizontal filters with shared thresholds.
pub fn lpf_horizontal_16_dual_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    lpf_horizontal_16_neon(pixels, s_index, pitch, blimit, limit, thresh);
    lpf_horizontal_16_neon(pixels, s_index + 8, pitch, blimit, limit, thresh);
}

/// Apply the 16-tap loop filter down eight vertical edge rows.
pub fn lpf_vertical_16_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    assert!(pitch >= 16);
    assert!(s_index >= 8);
    assert!(s_index + 7 * pitch + 7 < pixels.len());
    // SAFETY: AArch64 always provides NEON. The assertions prove that all
    // eight 16-byte row loads and filtered-pixel stores stay in `pixels`.
    unsafe { vertical_inner(pixels.as_mut_ptr(), s_index, pitch, blimit, limit, thresh) }
}

/// Apply two vertically adjacent 16-tap filters with shared thresholds.
pub fn lpf_vertical_16_dual_neon(
    pixels: &mut [u8],
    s_index: usize,
    pitch: usize,
    blimit: u8,
    limit: u8,
    thresh: u8,
) {
    lpf_vertical_16_neon(pixels, s_index, pitch, blimit, limit, thresh);
    lpf_vertical_16_neon(pixels, s_index + 8 * pitch, pitch, blimit, limit, thresh);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loopfilter::{
        lpf_horizontal_8, lpf_horizontal_8_dual, lpf_horizontal_16, lpf_horizontal_16_dual,
        lpf_vertical_8, lpf_vertical_8_dual, lpf_vertical_16, lpf_vertical_16_dual,
    };

    struct Lcg32(u32);

    impl Lcg32 {
        fn byte(&mut self) -> u8 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (self.0 >> 24) as u8
        }
    }

    #[test]
    fn transpose_8x8_roundtrip_preserves_every_position() {
        let input = core::array::from_fn::<_, 8, _>(|row| {
            core::array::from_fn::<_, 8, _>(|column| (row * 8 + column) as u8)
        });
        // SAFETY: AArch64 always provides NEON, and every load and store uses
        // one complete 8-byte array row.
        let actual = unsafe {
            let mut rows = [vdup_n_u8(0); 8];
            for (row, value) in rows.iter_mut().enumerate() {
                *value = vld1_u8(input[row].as_ptr());
            }
            let output = transpose_8x8(transpose_8x8(rows));
            let mut actual = [[0u8; 8]; 8];
            for (row, value) in output.iter().enumerate() {
                vst1_u8(actual[row].as_mut_ptr(), *value);
            }
            actual
        };
        assert_eq!(actual, input);
    }

    #[test]
    fn transpose_roundtrip_preserves_every_position() {
        let input = core::array::from_fn::<_, 8, _>(|row| {
            core::array::from_fn::<_, 16, _>(|column| (row * 16 + column) as u8)
        });
        // SAFETY: AArch64 always provides NEON, and every load and store uses
        // one complete 16-byte array row.
        let actual = unsafe {
            let mut rows = [vdupq_n_u8(0); 8];
            for (row, value) in rows.iter_mut().enumerate() {
                *value = vld1q_u8(input[row].as_ptr());
            }
            let output = transpose_8x16(transpose_16x8(rows));
            let mut actual = [[0u8; 16]; 8];
            for (row, value) in output.iter().enumerate() {
                vst1q_u8(actual[row].as_mut_ptr(), *value);
            }
            actual
        };
        assert_eq!(actual, input);
    }

    #[test]
    fn differential_random_8_tap_single_and_dual_edges() {
        let mut rng = Lcg32(0x8F11_7E22);
        for iteration in 0..512 {
            let pitch = 32 + iteration % 17;
            let height = 40;
            let input: Vec<u8> = (0..pitch * height).map(|_| rng.byte()).collect();
            let blimit0 = rng.byte();
            let limit0 = rng.byte();
            let thresh0 = rng.byte();
            let blimit1 = rng.byte();
            let limit1 = rng.byte();
            let thresh1 = rng.byte();

            let horizontal_index = 12 * pitch + 5;
            let vertical_index = 5 * pitch + 12;
            let mut scalar = input.clone();
            let mut neon = input.clone();
            lpf_horizontal_8(
                &mut scalar,
                horizontal_index,
                pitch,
                blimit0,
                limit0,
                thresh0,
            );
            lpf_horizontal_8_neon(&mut neon, horizontal_index, pitch, blimit0, limit0, thresh0);
            assert_eq!(neon, scalar, "horizontal iteration {iteration}");

            scalar.clone_from(&input);
            neon.clone_from(&input);
            lpf_vertical_8(&mut scalar, vertical_index, pitch, blimit0, limit0, thresh0);
            lpf_vertical_8_neon(&mut neon, vertical_index, pitch, blimit0, limit0, thresh0);
            assert_eq!(neon, scalar, "vertical iteration {iteration}");

            scalar.clone_from(&input);
            neon.clone_from(&input);
            lpf_horizontal_8_dual(
                &mut scalar,
                horizontal_index,
                pitch,
                blimit0,
                limit0,
                thresh0,
                blimit1,
                limit1,
                thresh1,
            );
            lpf_horizontal_8_dual_neon(
                &mut neon,
                horizontal_index,
                pitch,
                blimit0,
                limit0,
                thresh0,
                blimit1,
                limit1,
                thresh1,
            );
            assert_eq!(neon, scalar, "horizontal dual iteration {iteration}");

            scalar.clone_from(&input);
            neon.clone_from(&input);
            lpf_vertical_8_dual(
                &mut scalar,
                vertical_index,
                pitch,
                blimit0,
                limit0,
                thresh0,
                blimit1,
                limit1,
                thresh1,
            );
            lpf_vertical_8_dual_neon(
                &mut neon,
                vertical_index,
                pitch,
                blimit0,
                limit0,
                thresh0,
                blimit1,
                limit1,
                thresh1,
            );
            assert_eq!(neon, scalar, "vertical dual iteration {iteration}");
        }
    }

    #[test]
    fn differential_8_tap_boundary_does_not_saturate_at_255() {
        let pitch = 32;
        let height = 24;

        let mut horizontal = vec![0u8; pitch * height];
        for row in 12..height {
            horizontal[row * pitch..(row + 1) * pitch].fill(255);
        }
        let mut scalar = horizontal.clone();
        let mut neon = horizontal;
        let horizontal_index = 12 * pitch + 4;
        lpf_horizontal_8(&mut scalar, horizontal_index, pitch, 255, 255, 255);
        lpf_horizontal_8_neon(&mut neon, horizontal_index, pitch, 255, 255, 255);
        assert_eq!(neon, scalar);

        let mut vertical = vec![0u8; pitch * height];
        for row in 0..height {
            vertical[row * pitch + 16..(row + 1) * pitch].fill(255);
        }
        scalar.clone_from(&vertical);
        neon = vertical;
        let vertical_index = 4 * pitch + 16;
        lpf_vertical_8(&mut scalar, vertical_index, pitch, 255, 255, 255);
        lpf_vertical_8_neon(&mut neon, vertical_index, pitch, 255, 255, 255);
        assert_eq!(neon, scalar);
    }

    #[test]
    fn differential_random_single_and_dual_edges() {
        let mut rng = Lcg32(0x16F1_17E2);
        for iteration in 0..256 {
            let pitch = 32 + iteration % 17;
            let height = 40;
            let input: Vec<u8> = (0..pitch * height).map(|_| rng.byte()).collect();
            let blimit = rng.byte();
            let limit = rng.byte();
            let thresh = rng.byte();

            let horizontal_index = 12 * pitch + 5;
            let vertical_index = 5 * pitch + 12;
            let mut scalar = input.clone();
            let mut neon = input.clone();
            lpf_horizontal_16(&mut scalar, horizontal_index, pitch, blimit, limit, thresh);
            lpf_horizontal_16_neon(&mut neon, horizontal_index, pitch, blimit, limit, thresh);
            assert_eq!(neon, scalar, "horizontal iteration {iteration}");

            scalar.clone_from(&input);
            neon.clone_from(&input);
            lpf_vertical_16(&mut scalar, vertical_index, pitch, blimit, limit, thresh);
            lpf_vertical_16_neon(&mut neon, vertical_index, pitch, blimit, limit, thresh);
            assert_eq!(neon, scalar, "vertical iteration {iteration}");

            scalar.clone_from(&input);
            neon.clone_from(&input);
            lpf_horizontal_16_dual(&mut scalar, horizontal_index, pitch, blimit, limit, thresh);
            lpf_horizontal_16_dual_neon(&mut neon, horizontal_index, pitch, blimit, limit, thresh);
            assert_eq!(neon, scalar, "horizontal dual iteration {iteration}");

            scalar.clone_from(&input);
            neon.clone_from(&input);
            lpf_vertical_16_dual(&mut scalar, vertical_index, pitch, blimit, limit, thresh);
            lpf_vertical_16_dual_neon(&mut neon, vertical_index, pitch, blimit, limit, thresh);
            assert_eq!(neon, scalar, "vertical dual iteration {iteration}");
        }
    }

    #[test]
    fn differential_flat_and_step_edges() {
        let pitch = 32;
        let height = 40;
        for delta in 0..=4u8 {
            let mut input = vec![127u8; pitch * height];
            for row in 0..height {
                for col in 16..pitch {
                    input[row * pitch + col] = 127 + delta;
                }
            }
            for &(blimit, limit, thresh) in &[(0, 0, 0), (1, 1, 0), (255, 255, 255)] {
                let mut scalar = input.clone();
                let mut neon = input.clone();
                lpf_horizontal_16(&mut scalar, 16 * pitch + 4, pitch, blimit, limit, thresh);
                lpf_horizontal_16_neon(&mut neon, 16 * pitch + 4, pitch, blimit, limit, thresh);
                assert_eq!(neon, scalar, "flat horizontal delta {delta}");

                scalar.clone_from(&input);
                neon.clone_from(&input);
                lpf_vertical_16(&mut scalar, 4 * pitch + 16, pitch, blimit, limit, thresh);
                lpf_vertical_16_neon(&mut neon, 4 * pitch + 16, pitch, blimit, limit, thresh);
                assert_eq!(neon, scalar, "flat vertical delta {delta}");
            }
        }
    }
}
