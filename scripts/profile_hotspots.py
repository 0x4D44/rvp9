#!/usr/bin/env python3
"""Reduce a samply Firefox-profiler JSON.gz (+ .syms.json sidecar) into
ranked self-time and inclusive-time tables per function.

Discovery tool for the decode hotspot hunt — see wrk_journals. Joins the
unsymbolicated frame addresses in the profile against the presymbolicate
sidecar (samply record --unstable-presymbolicate).

Usage:
  python profile_hotspots.py <profile.json.gz> [thread_substr]
The sidecar is assumed at <profile.json.gz without .gz> + .syms.json,
i.e. profile_hl.json.gz -> profile_hl.json.syms.json
"""
import bisect
import gzip
import json
import sys
from collections import defaultdict


def load_gz(path):
    with gzip.open(path) as f:
        return json.load(f)


def build_sidecar(syms_path):
    """debugName -> (sorted_rvas, entries[(rva,size,name)])."""
    d = json.load(open(syms_path))
    st = d["string_table"]
    by_name = {}
    for m in d["data"]:
        entries = []
        for s in m["symbol_table"]:
            entries.append((s["rva"], s.get("size", 0), st[s["symbol"]]))
        entries.sort(key=lambda e: e[0])
        rvas = [e[0] for e in entries]
        by_name[m["debug_name"]] = (rvas, entries)
    return by_name


def resolve(by_name, debug_name, rva):
    """Return symbol name covering rva in module debug_name, or None."""
    rec = by_name.get(debug_name)
    if rec is None:
        return None
    rvas, entries = rec
    i = bisect.bisect_right(rvas, rva) - 1
    if i < 0:
        return None
    base, size, name = entries[i]
    if size and rva >= base + size:
        return None  # past the end of this symbol's range (gap)
    return name


def short_mod(modname):
    """Compact module tag for table prefixes."""
    if modname.endswith(".exe") or modname.endswith(".dll") or modname.endswith(".sys"):
        return modname.rsplit(".", 1)[0]
    return modname


def aggregate(prof, by_name, threads, rva_resolver=None, our_module=None):
    """Aggregate self + inclusive counts over a list of thread dicts.
    Returns (total, self_counts, incl_counts, self_by_module).

    rva_resolver: optional DbgHelp Resolver for our_module (samply's own
    PDB parser fails to name Rust symbols on Windows, so we resolve those
    RVAs directly). our_module is the short module tag (e.g. 'decode-headtohead')."""
    libs = prof["libs"]
    self_counts = defaultdict(float)
    incl_counts = defaultdict(float)
    self_by_module = defaultdict(float)
    total = 0.0

    for t in threads:
        ft = t["frameTable"]
        fu = t["funcTable"]
        rt = t["resourceTable"]
        addr = ft["address"]
        ffunc = ft["func"]
        fres = fu["resource"]
        rlib = rt["lib"]

        nframes = ft["length"]
        frame_label = [None] * nframes
        for fi in range(nframes):
            a = addr[fi]
            func = ffunc[fi]
            res = fres[func] if func is not None and func < len(fres) else -1
            libidx = rlib[res] if (res is not None and 0 <= res < len(rlib)) else None
            if libidx is None or libidx < 0 or a is None or a < 0:
                frame_label[fi] = ("?", "<unknown>")
                continue
            lib = libs[libidx]
            modname = short_mod(lib.get("name", lib.get("debugName", "?")))
            sym = None
            if rva_resolver is not None and modname == our_module:
                sym = rva_resolver.name(a)
            if sym is None:
                sym = resolve(by_name, lib.get("debugName", ""), a)
            if sym is None:
                sym = f"+0x{a:x}"
            frame_label[fi] = (modname, f"{modname}!{sym}")

        stab = t["stackTable"]
        sprefix = stab["prefix"]
        sframe = stab["frame"]
        stacks = t["samples"]["stack"]
        weights = t["samples"].get("weight")

        for si, sidx in enumerate(stacks):
            if sidx is None:
                continue
            w = 1.0 if (weights is None or weights[si] is None) else float(weights[si])
            total += w
            mod, sym = frame_label[sframe[sidx]]
            self_counts[sym] += w
            self_by_module[mod] += w
            seen = set()
            cur = sidx
            while cur is not None:
                _, s2 = frame_label[sframe[cur]]
                if s2 not in seen:
                    seen.add(s2)
                    incl_counts[s2] += w
                cur = sprefix[cur]

    return total, self_counts, incl_counts, self_by_module


def report(title, prof, by_name, threads, topn=40, rva_resolver=None, our_module=None):
    total, self_counts, incl_counts, self_by_module = aggregate(
        prof, by_name, threads, rva_resolver, our_module
    )
    print(f"\n########## {title}  (total weighted samples: {total:.0f}) ##########\n")
    if total == 0:
        print("  (no samples)\n")
        return
    print("== SELF time by MODULE ==")
    for mod, c in sorted(self_by_module.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {100.0*c/total:6.2f}%  {c:9.0f}  {mod}")
    print()

    def dump(t2, counts):
        print(f"== {t2} (top {topn}) ==")
        for fn, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:topn]:
            short = fn if len(fn) <= 150 else fn[:147] + "..."
            print(f"  {100.0*c/total:6.2f}%  {c:9.0f}  {short}")
        print()

    dump("SELF time (leaf)", self_counts)
    dump("INCLUSIVE time", incl_counts)


def main(profile_path, exe_path=None):
    syms_path = profile_path[:-3] if profile_path.endswith(".gz") else profile_path
    syms_path = syms_path + ".syms.json"
    prof = load_gz(profile_path)
    by_name = build_sidecar(syms_path)

    # Identify our exe module + build a DbgHelp resolver for it.
    our_module = None
    rva_resolver = None
    for lib in prof["libs"]:
        if lib.get("name", "").endswith(".exe") and "decode" in lib.get("name", ""):
            our_module = short_mod(lib["name"])
            break
    if exe_path is None and our_module is not None:
        exe_path = f"target/release/{our_module}.exe"
    if exe_path:
        try:
            from symbolize_pdb import Resolver

            rva_resolver = Resolver(exe_path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] DbgHelp resolver unavailable ({e}); using sidecar only")

    print("== threads ==")
    for ti, t in enumerate(prof["threads"]):
        n = len(t["samples"]["stack"])
        print(f"  [{ti}] {t.get('name','?')!r:24} main={t.get('isMainThread')} samples={n}")

    workers = [t for t in prof["threads"] if "worker" in t.get("name", "")]
    mains = [t for t in prof["threads"] if t.get("isMainThread")]
    report("ALL THREADS", prof, by_name, prof["threads"], rva_resolver=rva_resolver, our_module=our_module)
    report("WORKER THREADS ONLY (parallel-efficiency view)", prof, by_name, workers, rva_resolver=rva_resolver, our_module=our_module)
    report("MAIN THREAD ONLY", prof, by_name, mains, rva_resolver=rva_resolver, our_module=our_module)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "profile_hl2.json.gz"
    exe = sys.argv[2] if len(sys.argv) > 2 else None
    main(p, exe)
