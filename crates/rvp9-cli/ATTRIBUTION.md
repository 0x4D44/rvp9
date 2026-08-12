# Attribution

This project is a Rust port of the VP9 encoder and decoder from **libvpx**,
maintained by the WebM Project at Google. libvpx is used as both the
reference implementation (for differential testing and byte-parity work) and,
in early stages, as the source of probability tables, scan orders, and other
encoding-related constants transcribed verbatim.

## libvpx

- **Source:** https://chromium.googlesource.com/webm/libvpx
- **Pinned version:** v1.14.1 (commit `12f3a2a`)
- **License:** BSD 3-Clause (see `third_party/libvpx/LICENSE` after the
  submodule is initialised, or the verbatim notice below)
- **Patent grant:** Additional IP Rights Grant (Patents), see
  `third_party/libvpx/PATENTS` and [`PATENTS`](./PATENTS)

### libvpx LICENSE text

```
Copyright (c) 2010, The WebM Project authors. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * Neither the name of Google, nor the WebM Project, nor the names
    of its contributors may be used to endorse or promote products
    derived from this software without specific prior written
    permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## VP9 Bitstream Specification

The decoder is implemented to the **VP9 Bitstream & Decoding Process
Specification** published by Google / WebM. The spec is the unambiguous
oracle for decoder conformance.

- Spec: https://www.webmproject.org/vp9/
- Specification patent licence: https://www.webmproject.org/license/bitstream/

Google's patent grants are reproduced in [`PATENTS`](./PATENTS). They cover
only the claims and uses described in those grants. They do not represent
clearance of unrelated third-party patent rights.

## Test vectors

The official VP9 conformance test vectors used in `tests/conformance/` are
distributed under the libvpx project's licence. Fetched by
`scripts/fetch-vectors.ps1`.

## Test corpus

Test clips under `tests/corpus/` are sourced from:

- **AOM Common Test Conditions clips:** CC-licensed, fetched by
  `scripts/fetch-corpus.ps1`.
- **Xiph.org test clips:** public domain or CC, fetched by
  `scripts/fetch-corpus.ps1`.

If you add proprietary or differently-licensed clips, update this file.

## Acknowledgements

This project would not be possible without the decades of work poured into
libvpx by the WebM Project authors. The Rust port is an independent
reimplementation; any defects, divergences from spec, or performance
shortfalls are ours, not theirs.
