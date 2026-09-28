BEGIN {
    print "clip,frames,tx_blocks,y_tx,y_tx4,y_tx8,y_tx16,y_tx32,u_tx,u_tx4,u_tx8,u_tx16,u_tx32,v_tx,v_tx4,v_tx8,v_tx16,v_tx32"
}

function emit() {
    if (frames == 0) return
    path_parts = split(current, path, "/")
    split(path[path_parts], name, ".")
    printf "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        name[1], frames, sum["tx_blocks"], sum["y_tx"],
        sum["y_tx4"], sum["y_tx8"], sum["y_tx16"], sum["y_tx32"],
        sum["u_tx"], sum["u_tx4"], sum["u_tx8"], sum["u_tx16"], sum["u_tx32"],
        sum["v_tx"], sum["v_tx4"], sum["v_tx8"], sum["v_tx16"], sum["v_tx32"]
}

FNR == 1 {
    if (NR > 1) {
        emit()
        delete sum
        frames = 0
    }
    current = FILENAME
}

/\[EMIT_PROFILE\]/ {
    delete value
    for (i = 1; i <= NF; i++) {
        pair_parts = split($i, pair, "=")
        if (pair_parts == 2) {
            value[pair[1]] = pair[2] + 0
            sum[pair[1]] += value[pair[1]]
        }
    }
    frames++
    if (value["tx_blocks"] != value["y_tx"] + value["u_tx"] + value["v_tx"]) {
        print FILENAME ":" FNR ": plane total mismatch" > "/dev/stderr"
        bad++
    }
    for (p = 1; p <= 3; p++) {
        plane = (p == 1 ? "y" : p == 2 ? "u" : "v")
        shape_sum = 0
        for (s = 1; s <= 5; s++) {
            side = (s == 1 ? 4 : s == 2 ? 8 : s == 3 ? 16 : s == 4 ? 32 : 64)
            shape_sum += value[plane "_tx" side]
        }
        if (shape_sum != value[plane "_tx"]) {
            print FILENAME ":" FNR ": " plane " shape mismatch" > "/dev/stderr"
            bad++
        }
    }
}

END {
    emit()
    if (bad) exit 1
}
