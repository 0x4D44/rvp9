//! Shared building blocks for the `rvp9enc` / `rvp9dec` CLIs.
//!
//! The only shared piece today is [`args`] — a tiny `vpxenc`/`vpxdec`-
//! style argument parser whose flag table is the *single source of
//! truth* for which options exist and which are wired up.

pub mod args;
