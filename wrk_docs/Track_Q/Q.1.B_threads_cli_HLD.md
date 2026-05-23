# HLD — Q.1.B: wire `rvp9dec --threads N` to `Decoder::with_threads()`

**Author:** Q.1.B design agent
**Date:** 2026-05-23
**Status:** SIGNED OFF by lead 2026-05-23 (with one revision; see §9)

## 1. Scope and out-of-scope

In-scope: change the `--threads` row in `flag_table()` from `apply: None` to a wired handler that parses N and constructs the decoder via `Decoder::with_threads(NonZeroUsize)`.

Out-of-scope: **frame-parallel decode** (multiple frames in flight). This wave is *tile* threading only — one frame's tiles dispatched in parallel through the existing `WorkerPool` (`crates/rvp9-decoder/src/lib.rs:1476`). Frame-parallel is Q.1.C.

## 2. Behavior matrix (POST LEAD REVISION — see §9)

| Input | Behavior |
|---|---|
| flag absent | **CLI default = auto-detect** via `Decoder::new()` (which calls `available_parallelism()` at `crates/rvp9-decoder/src/lib.rs:615`). Matches libvpx UX. |
| `--threads 0` | **1 thread** (single-threaded), **matching libvpx exactly** (`vpxdec.c:629-630` forwards 0 verbatim; libvpx decoder treats it as "no extra workers"). Users wanting auto omit the flag. |
| `--threads 1` | Single-thread; `WorkerPool::new(1)`, no parallel dispatch (gated by `self.num_threads.get() > 1` at `crates/rvp9-decoder/src/lib.rs:1476`). |
| `--threads N`, 2 ≤ N ≤ `usize::MAX` | `Decoder::with_threads(NonZeroUsize::new(N).unwrap())`. No upper cap — libvpx has no hard cap in `vpxdec` either. |
| negative / `-1` | `parse_value::<usize>` fails → `"rvp9dec: --threads: invalid value '-1'"`, exit 1. |
| non-numeric / `abc` | Same `parse_value` rejection, exit 1. |
| overflow (`> usize::MAX`) | Same `parse_value` rejection, exit 1. |
| missing value (`rvp9dec --threads`) | `"rvp9dec: --threads requires a value"`, exit 1 (`crates/rvp9-cli/src/args.rs:140`). |
| repeated (`--threads 2 --threads 4`) | **Last-wins.** Closure overwrites `opts.threads` on each occurrence. |

## 3. Default thread count — auto-detect when flag absent

`Decoder::new()` already picks `available_parallelism()` (`crates/rvp9-decoder/src/lib.rs:615`). The CLI currently defaults to 1 only because `--threads` errors. Once wired, calling `Decoder::new()` when the flag is absent gives free auto-detect with no special-case code, matches `vpxdec`'s feel, and keeps the existing `Decoder::new()` path covered.

Represent this in `DecOpts` as `threads: Option<NonZeroUsize>` (`None` = default). At construction:
```
match opts.threads {
    Some(n) => Decoder::with_threads(n),
    None    => Decoder::new(),
}
```

## 4. Flag table change (`crates/rvp9-cli/src/bin/rvp9dec.rs:107`)

One row swaps `apply: None` for a closure that:

1. Calls `args::parse_value::<usize>(v, "rvp9dec", "--threads")` for consistent error formatting.
2. **If parsed = 0, store `Some(NonZeroUsize::new(1).unwrap())`** — matches libvpx (0 = 1 thread).
3. Otherwise wraps the parsed `usize` in `NonZeroUsize::new(n).expect(...)` (infallible since n ≥ 1).
4. Stores into `opts.threads: Option<NonZeroUsize>`.

Also add the `threads: Option<NonZeroUsize>` field to `DecOpts` (line 46-56) with default `None`, and update `run()` at line 200 to branch on `Decoder::new()` vs `Decoder::with_threads(n)`.

Help string: `"max decode threads (0 = 1 thread, omit for auto-detect)"` — explicit about the libvpx-parity 0 case and the auto-detect default.

## 5. Validation

All validation happens at the CLI boundary in the `apply` closure; the decoder side already enforces N ≥ 1 via `NonZeroUsize`. No clamping: libvpx doesn't cap in vpxdec.

## 6. Tests

New tests in `crates/rvp9-cli/tests/cli_roundtrip.rs`:

1. **`threads_flag_accepted`** — `rvp9dec --threads 4 <ivf>` exits 0, output byte-identical to no-`--threads` decode.
2. **`threads_zero_means_one_thread`** — `--threads 0` exits 0, same output as `--threads 1`.
3. **`threads_invalid_rejected`** — three sub-asserts: `--threads -1`, `--threads abc`, `--threads` (missing value). Each: non-zero exit, stderr names `--threads` and a sensible message.
4. **`threads_help_tag_supported`** — `--help` for `--threads` no longer flagged as unsupported.

Coverage trade-off recorded: the cross-process `RVP9_G5_1_PARALLEL_JOBS` counter (`crates/rvp9-decoder/src/lib.rs:134`) is not directly observable from CLI integration tests (lives in the child binary's process). We rely on the existing `multi_tile_dispatch_uses_worker_pool` test (`crates/rvp9-harness/tests/tile_threading.rs:270`) which already proves the constructor-level path. The CLI is a thin shim and doesn't need to re-prove pool usage; only that it routes correctly.

## 7. Doc updates

- `--threads` help text update (see §4).
- README `Usage` section: add `--threads` row if a flag table exists there. If not, `--help` is canonical.

## 8. Plan-of-record

**Single wave, one commit:**

1. Add `threads: Option<NonZeroUsize>` to `DecOpts` (default `None`).
2. Swap the `--threads` row's `apply: None` for the closure described in §4.
3. Update help string.
4. Branch `Decoder::new()` vs `Decoder::with_threads(n)` in `run()`.
5. Add CLI tests 1-4.

No sequencing dependencies; no decoder-side changes; no public API changes.

## 9. Lead revision on sign-off

The agent's draft proposed `--threads 0` → auto-detect via `available_parallelism()`. **This diverges from libvpx**, which treats `0` as "no extra workers" (= 1 thread). For CLI parity (users coming from `vpxdec` expect identical semantics), the sign-off version is:

- `--threads 0` → **1 thread** (matches libvpx).
- Flag absent → **auto-detect** (matches libvpx's effective default via codec config).
- Help string explicit about both.

### Critical Files for Implementation

- `D:\language\rvp9-2\crates\rvp9-cli\src\bin\rvp9dec.rs` (flag table row 107, `DecOpts` 46-72, `run()` 200)
- `D:\language\rvp9-2\crates\rvp9-cli\src\args.rs` (`parse_value` helper, line 211)
- `D:\language\rvp9-2\crates\rvp9-cli\tests\cli_roundtrip.rs` (new CLI tests)
- `D:\language\rvp9-2\crates\rvp9-decoder\src\lib.rs` (read-only: `with_threads` at 639, `num_threads_default` at 615)
- `D:\language\rvp9-2\crates\rvp9-harness\tests\tile_threading.rs` (existing `multi_tile_dispatch_uses_worker_pool` at 270)
