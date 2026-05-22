//! A small `vpxenc` / `vpxdec`-style argument parser.
//!
//! ## Design: support is emergent, not catalogued
//!
//! Each option is a [`Flag`] entry in a table. The entry carries an
//! optional handler ([`Flag::apply`]):
//!
//! - `Some(handler)` → the flag is **wired**: the handler applies the
//!   value to the tool's config builder.
//! - `None` → the flag is **recognised but not implemented yet**: using
//!   it is a hard error (`"... recognised but not supported yet"`),
//!   never a silent no-op.
//!
//! There is no separate "supported flags" list to keep in sync. When a
//! feature is implemented you swap the flag's `None` for a `Some(...)`
//! handler and it auto-enables — and [`print_help`]'s `supported` /
//! `not yet` tag is *derived from the same field*, so the docs can never
//! drift from reality.
//!
//! ## Syntax accepted
//!
//! - Long: `--name=value`, `--name value`, or `--name` (boolean).
//! - Short: `-x value`, `-xvalue`, or `-x` (boolean).
//! - Everything else is a positional argument (input/output paths).

use std::fmt::Write as _;

/// Whether a flag consumes a value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arg {
    /// Boolean flag — takes no value (e.g. `--ivf`, `--md5`).
    None,
    /// Flag requires a value (e.g. `--target-bitrate=800`).
    Required,
}

/// One command-line option, generic over the tool's config builder `C`.
///
/// The `apply` field is the single source of truth for whether the flag
/// is wired (`Some`) or recognised-but-unsupported (`None`).
pub struct Flag<C: 'static> {
    /// Long name without the leading `--` (e.g. `"target-bitrate"`).
    pub long: &'static str,
    /// Optional single-character short alias (e.g. `Some('o')`).
    pub short: Option<char>,
    /// Whether the flag takes a value.
    pub arg: Arg,
    /// One-line help text shown by [`print_help`].
    pub help: &'static str,
    /// Handler that applies the parsed value to the config builder.
    /// `None` marks the flag as not-yet-implemented.
    pub apply: Option<fn(&mut C, Option<String>) -> Result<(), String>>,
}

impl<C> Flag<C> {
    /// `true` if the flag is wired up (has a handler).
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.apply.is_some()
    }
}

/// Parse `argv` against `table`, applying wired handlers to `cfg`.
/// Returns the positional arguments (in order).
///
/// # Errors
///
/// Returns a human-readable message (already prefixed with `tool`) for
/// unknown options, missing values, values on boolean flags, or use of
/// a recognised-but-unsupported flag.
pub fn parse<C>(
    table: &[Flag<C>],
    cfg: &mut C,
    argv: &[String],
    tool: &str,
) -> Result<Vec<String>, String> {
    let mut positionals = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        let tok = &argv[i];
        if let Some(rest) = tok.strip_prefix("--") {
            // Long option: --name or --name=value.
            let (name, inline_val) = match rest.split_once('=') {
                Some((n, v)) => (n, Some(v.to_string())),
                None => (rest, None),
            };
            let flag = table
                .iter()
                .find(|f| f.long == name)
                .ok_or_else(|| format!("{tool}: unknown option '--{name}'"))?;
            let value = take_value(flag, inline_val, argv, &mut i, tool, &format!("--{name}"))?;
            apply(flag, cfg, value, tool)?;
        } else if tok.len() >= 2 && tok.starts_with('-') {
            // Short option: -x, -xVALUE, or -x VALUE.
            let c = tok[1..].chars().next().unwrap();
            let attached = &tok[1 + c.len_utf8()..];
            let inline_val = if attached.is_empty() {
                None
            } else {
                Some(attached.to_string())
            };
            let flag = table
                .iter()
                .find(|f| f.short == Some(c))
                .ok_or_else(|| format!("{tool}: unknown option '-{c}'"))?;
            let value = take_value(flag, inline_val, argv, &mut i, tool, &format!("-{c}"))?;
            apply(flag, cfg, value, tool)?;
        } else {
            // Positional ("-" included: treated as a path by the caller).
            positionals.push(tok.clone());
        }
        i += 1;
    }
    Ok(positionals)
}

/// Resolve the value for `flag` from an inline `=value`/attached form or
/// the next argv token, honouring the flag's [`Arg`] kind.
fn take_value<C>(
    flag: &Flag<C>,
    inline_val: Option<String>,
    argv: &[String],
    i: &mut usize,
    tool: &str,
    shown: &str,
) -> Result<Option<String>, String> {
    match flag.arg {
        Arg::None => {
            if inline_val.is_some() {
                return Err(format!("{tool}: {shown} takes no value"));
            }
            Ok(None)
        }
        Arg::Required => match inline_val {
            Some(v) => Ok(Some(v)),
            None => {
                *i += 1;
                let v = argv
                    .get(*i)
                    .ok_or_else(|| format!("{tool}: {shown} requires a value"))?;
                Ok(Some(v.clone()))
            }
        },
    }
}

/// Dispatch to the flag's handler, or reject if it's not wired.
fn apply<C>(
    flag: &Flag<C>,
    cfg: &mut C,
    value: Option<String>,
    tool: &str,
) -> Result<(), String> {
    match flag.apply {
        Some(handler) => handler(cfg, value),
        None => Err(format!(
            "{tool}: --{} is recognised but not supported yet",
            flag.long
        )),
    }
}

/// Render a `--help` listing: every flag with its support tag, derived
/// from whether it has a handler. `usage` is the one-line synopsis.
#[must_use]
pub fn render_help<C>(table: &[Flag<C>], tool: &str, usage: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{tool}\n\nUsage: {usage}\n\nOptions:");
    // Column-align the flag names.
    let width = table
        .iter()
        .map(|f| flag_label(f).len())
        .max()
        .unwrap_or(0)
        .max(12);
    for f in table {
        let label = flag_label(f);
        let tag = if f.is_supported() {
            "[supported]"
        } else {
            "[not yet]"
        };
        let _ = writeln!(out, "  {label:width$}  {tag:<12} {}", f.help);
    }
    out
}

/// Print a `--help` listing to stdout.
pub fn print_help<C>(table: &[Flag<C>], tool: &str, usage: &str) {
    print!("{}", render_help(table, tool, usage));
}

/// Format the `--long[, -x] [<arg>]` label for one flag.
fn flag_label<C>(f: &Flag<C>) -> String {
    let mut s = String::new();
    let _ = write!(s, "--{}", f.long);
    if let Some(c) = f.short {
        let _ = write!(s, ", -{c}");
    }
    if f.arg == Arg::Required {
        s.push_str(" <arg>");
    }
    s
}

/// Parse a value as a `T`, mapping failures to a `tool`-prefixed error.
///
/// # Errors
///
/// Returns an error if `value` is `None` or doesn't parse as `T`.
pub fn parse_value<T: std::str::FromStr>(
    value: Option<String>,
    tool: &str,
    flag: &str,
) -> Result<T, String> {
    let v = value.ok_or_else(|| format!("{tool}: {flag} requires a value"))?;
    v.parse::<T>()
        .map_err(|_| format!("{tool}: {flag}: invalid value '{v}'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Cfg {
        width: u32,
        bitrate: u32,
        ivf: bool,
    }

    fn table() -> Vec<Flag<Cfg>> {
        vec![
            Flag {
                long: "width",
                short: Some('w'),
                arg: Arg::Required,
                help: "frame width",
                apply: Some(|c, v| {
                    c.width = parse_value(v, "test", "--width")?;
                    Ok(())
                }),
            },
            Flag {
                long: "target-bitrate",
                short: None,
                arg: Arg::Required,
                help: "target kbps",
                apply: Some(|c, v| {
                    c.bitrate = parse_value(v, "test", "--target-bitrate")?;
                    Ok(())
                }),
            },
            Flag {
                long: "ivf",
                short: None,
                arg: Arg::None,
                help: "ivf output",
                apply: Some(|c, _| {
                    c.ivf = true;
                    Ok(())
                }),
            },
            Flag {
                long: "svc",
                short: None,
                arg: Arg::None,
                help: "spatial layers",
                apply: None, // not yet implemented
            },
        ]
    }

    fn args(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn long_eq_form() {
        let mut c = Cfg::default();
        let pos = parse(&table(), &mut c, &args(&["--width=640", "in.y4m"]), "test").unwrap();
        assert_eq!(c.width, 640);
        assert_eq!(pos, vec!["in.y4m"]);
    }

    #[test]
    fn long_space_form_and_short_attached() {
        let mut c = Cfg::default();
        parse(
            &table(),
            &mut c,
            &args(&["--target-bitrate", "800", "-w1280"]),
            "test",
        )
        .unwrap();
        assert_eq!(c.bitrate, 800);
        assert_eq!(c.width, 1280);
    }

    #[test]
    fn short_space_form() {
        let mut c = Cfg::default();
        parse(&table(), &mut c, &args(&["-w", "320"]), "test").unwrap();
        assert_eq!(c.width, 320);
    }

    #[test]
    fn boolean_flag_sets_and_rejects_value() {
        let mut c = Cfg::default();
        parse(&table(), &mut c, &args(&["--ivf"]), "test").unwrap();
        assert!(c.ivf);

        let mut c2 = Cfg::default();
        let err = parse(&table(), &mut c2, &args(&["--ivf=1"]), "test").unwrap_err();
        assert!(err.contains("takes no value"), "{err}");
    }

    #[test]
    fn unsupported_flag_errors_not_silently_ignored() {
        let mut c = Cfg::default();
        let err = parse(&table(), &mut c, &args(&["--svc"]), "test").unwrap_err();
        assert!(err.contains("not supported yet"), "{err}");
    }

    #[test]
    fn unknown_flag_errors() {
        let mut c = Cfg::default();
        let err = parse(&table(), &mut c, &args(&["--nonsense"]), "test").unwrap_err();
        assert!(err.contains("unknown option"), "{err}");
    }

    #[test]
    fn missing_value_errors() {
        let mut c = Cfg::default();
        let err = parse(&table(), &mut c, &args(&["--width"]), "test").unwrap_err();
        assert!(err.contains("requires a value"), "{err}");
    }

    #[test]
    fn help_tag_derived_from_handler_presence() {
        let t = table();
        let help = render_help(&t, "test", "test [opts] in");
        // Wired flags tagged supported; handler-less flags tagged not yet.
        assert!(help.contains("--width"));
        let svc_line = help
            .lines()
            .find(|l| l.contains("--svc"))
            .expect("svc line");
        assert!(svc_line.contains("[not yet]"), "{svc_line}");
        let width_line = help
            .lines()
            .find(|l| l.contains("--width"))
            .expect("width line");
        assert!(width_line.contains("[supported]"), "{width_line}");
    }
}
