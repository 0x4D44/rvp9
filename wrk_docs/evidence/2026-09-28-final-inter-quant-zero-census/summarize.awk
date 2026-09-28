function flush(    plane, key, total, triplet, path, nz) {
    if (frames == 0) return
    for (plane = 1; plane <= 3; plane++) {
        key = (plane == 1 ? "y" : plane == 2 ? "u" : "v")
        triplet = sums[key "_exact_zero"] + sums[key "_quant_zero"] + sums[key "_eob_nonzero"]
        path = sums[key "_source"] + sums[key "_direct"] + sums[key "_direct_deferred"]
        if (triplet != sums[key "_tx"] || path != sums[key "_tx"]) {
            print "bucket mismatch", file, key, triplet, path, sums[key "_tx"] > "/dev/stderr"
            exit 1
        }
    }
    total = sums["y_tx"] + sums["u_tx"] + sums["v_tx"]
    nz = sums["y_eob_nonzero"] + sums["u_eob_nonzero"] + sums["v_eob_nonzero"]
    if (total != sums["tx_blocks"] || nz != sums["nonzero_tx"]) {
        print "total mismatch", file, total, nz > "/dev/stderr"
        exit 1
    }
    printf "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.2f\n", file, frames, sums["tx_blocks"], sums["y_tx"], sums["y_exact_zero"], sums["y_quant_zero"], sums["y_eob_nonzero"], sums["y_source"], sums["y_direct"], sums["y_direct_deferred"], sums["u_tx"], sums["u_exact_zero"], sums["u_quant_zero"], sums["u_eob_nonzero"], sums["u_source"], sums["u_direct"], sums["u_direct_deferred"], sums["v_tx"], sums["v_exact_zero"], sums["v_quant_zero"], sums["v_eob_nonzero"], sums["v_source"], sums["v_direct"], sums["v_direct_deferred"], sums["nonzero_tx"], sums["eob_sum"], sums["quant"]
    delete sums
    frames = 0
}
BEGIN {
    print "clip,frames,tx_blocks,y_tx,y_exact_zero,y_quant_zero,y_eob_nonzero,y_source,y_direct,y_direct_deferred,u_tx,u_exact_zero,u_quant_zero,u_eob_nonzero,u_source,u_direct,u_direct_deferred,v_tx,v_exact_zero,v_quant_zero,v_eob_nonzero,v_source,v_direct,v_direct_deferred,nonzero_tx,eob_sum,quant_ms"
}
FNR == 1 {
    flush()
    file = FILENAME
    sub(/^.*\//, "", file)
    sub(/\.stderr\.txt$/, "", file)
}
/\[EMIT_PROFILE\]/ {
    frames++
    for (i = 1; i <= NF; i++) {
        split($i, part, "=")
        if (length(part[2])) sums[part[1]] += part[2] + 0
    }
}
END { flush() }
