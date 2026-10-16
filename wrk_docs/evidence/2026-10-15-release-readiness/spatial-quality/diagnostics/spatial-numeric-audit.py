#!/usr/bin/env python3
"""Diagnostic cubic least-squares audit; never substitutes for acceptance."""
import csv
import decimal
import hashlib
import json
import pathlib
import sys

decimal.getcontext().prec = 80
D = decimal.Decimal
ROOT = pathlib.Path(sys.argv[1])


def fit(points, mid, span):
    xy = [((D(p[4]) - mid) / span, D(p[3]).ln()) for p in points]
    matrix = [
        [sum(x ** (i + j) for x, _ in xy) for j in range(4)]
        + [sum(y * x**i for x, y in xy)]
        for i in range(4)
    ]
    for i in range(4):
        pivot = max(range(i, 4), key=lambda k: abs(matrix[k][i]))
        matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
        factor = matrix[i][i]
        matrix[i] = [x / factor for x in matrix[i]]
        for k in range(4):
            if k != i:
                factor = matrix[k][i]
                matrix[k] = [x - factor * y for x, y in zip(matrix[k], matrix[i])]
    return [r[-1] for r in matrix]


def bdrate(anchor, test, centered):
    lo = max(min(D(p[4]) for p in anchor), min(D(p[4]) for p in test))
    hi = min(max(D(p[4]) for p in anchor), max(D(p[4]) for p in test))
    assert hi > lo
    mid, span = ((lo + hi) / 2, (hi - lo) / 2) if centered else (D(0), D(1))
    ca, cb = fit(anchor, mid, span), fit(test, mid, span)
    lower, upper = (lo - mid) / span, (hi - mid) / span
    integral = sum(
        (cb[i] - ca[i]) * (upper ** (i + 1) - lower ** (i + 1)) / D(i + 1)
        for i in range(4)
    )
    return ((integral / (upper - lower)).exp() - 1) * 100


results = []
for source in sorted((ROOT / 'quality').glob('*/bdrate.csv')):
    rows = list(csv.reader(source.open()))[1:]
    for gate in (r for r in rows if r[0] == 'bdrate_pct'):
        anchor = [r for r in rows if r[0] == 'anchor' and r[1] == gate[1]]
        test = [r for r in rows if r[0] == 'test' and r[1] == gate[1]]
        assert len(anchor) == len(test) == 6
        value = bdrate(anchor, test, False)
        centered = bdrate(anchor, test, True)
        results.append({
            'curve_id': source.parent.name, 'layer': int(gate[1]),
            'mode': source.parent.name.split('-')[0], 'recorded_pct': D(gate[2]),
            'decimal_original_pct': value, 'decimal_centered_pct': centered,
            'difference_pp': value - D(gate[2]),
            'basis_agreement_pp': abs(value - centered),
            'anchor_points': [(r[3], r[4]) for r in anchor],
            'test_points': [(r[3], r[4]) for r in test],
            'raw_csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        })

summary_rows = list(csv.reader((ROOT / 'quality/quality-summary.csv').open()))
cross = list(csv.DictReader((ROOT / 'quality/cross-decoder.csv').open()))
stats = {}
for mode, ceiling in [('cbr', D('.5')), ('vbr', D(25))]:
    rows = [r for r in results if r['mode'] == mode]
    stats[mode] = {
        'curves': len({r['curve_id'] for r in rows}), 'layers': len(rows),
        'ceiling_pct': ceiling,
        'recorded_mean_pct': sum(r['recorded_pct'] for r in rows) / len(rows),
        'decimal_mean_pct': sum(r['decimal_original_pct'] for r in rows) / len(rows),
        'recorded_failures': sum(r['recorded_pct'] > ceiling for r in rows),
        'decimal_failures': sum(r['decimal_original_pct'] > ceiling for r in rows),
        'decimal_above_production_ceiling': sum(r['decimal_original_pct'] > D('.5') for r in rows),
        'worst': max(rows, key=lambda r: r['decimal_original_pct']),
    }
output = {
    'purpose': 'Diagnostic oracle, not acceptance workaround; rounded CSV points only',
    'method': '80-digit Decimal; degree3 least squares all6 points; ln bitrate against PSNR; normal equations with partial-pivot Gauss-Jordan; exact overlap integral; original and centered bases',
    'evidence_root': str(ROOT), 'decimal_precision': 80,
    'layer_results': len(results), 'stats': stats,
    'max_basis_disagreement_pp': max(r['basis_agreement_pp'] for r in results),
    'strongest_discrepancy': max(results, key=lambda r: abs(r['difference_pp'])),
    'csv_header_fields': len(summary_rows[0]),
    'csv_data_field_widths': sorted({len(r) for r in summary_rows[1:]}),
    'summary_rows': len(summary_rows) - 1,
    'cross_stream_rows': len(cross), 'cross_unique_curve_streams': len({(r['curve_id'],r['stream']) for r in cross}),
    'cross_statuses': sorted({r['status'] for r in cross}),
    'decoder_sha_mismatches': sum(r['rvp9dec_sha256'] != r['vpxdec_sha256'] for r in cross),
    'ffmpeg_statuses': sorted({r['ffmpeg_decode'] for r in cross}),
    'results': results,
}
print(json.dumps(output, default=str, indent=2))
