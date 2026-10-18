# Spatial cross-decoder mismatch failure probe

On runner revision 74b8edbb9a6caeeab9dbafa3ea5aa1b536177b58, the spatial qualification script blob matches product candidate fb5c0c917038db6c640b3dbc7c2b84b567233333. A registered scratch harness extracted the script's cross_check_curve, cross_check_stream, decode, hash and failure helpers verbatim, plus the four constants they use.

The harness supplied the 18 nonempty IVF filenames expected for a two-layer curve across six bitrate points. Its RVP9 and pinned-libvpx decoder stubs emitted different nonempty byte streams; the FFmpeg stub exited successfully. The curve stream-count check passed, the first full-stream CSV row recorded no,ok,FAIL, and the helper exited 1 with a cross-decoder mismatch diagnostic. The raw output and exit capture, exact extracted helper, stub tools, all 18 fixtures and their hashes are retained here and inventoried in qualification.json.

This verifies the exact-source spatial helper's mismatch branch and count preflight, not the full spatial quality phase or phase publication. The data and decoders are synthetic; no codec, compression-quality or performance result is claimed. FLU-00041 and FLU-00042 remain separate quality-campaign blockers.
