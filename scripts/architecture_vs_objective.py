#!/usr/bin/env python
"""Does representation geometry track architecture or pretraining objective?

For a similarity matrix S under metric m, contrast the mean similarity of pairs
that share a design factor against pairs that do not:

    delta_obj  = E[S_ij | o_i = o_j] - E[S_ij | o_i != o_j]
    delta_arch = E[S_ij | a_i = a_j] - E[S_ij | a_i != a_j]

A larger positive delta means that factor is more associated with representation
geometry. Annotations come from configs/encoders.yaml -- `family` for the
objective, `architecture` for the backbone.

Architecture is grouped two ways, because neither alone is honest:

  backbone  ViT / hybrid / CNN. Coarse enough to have same-arch pairs on every
            group, but on the flagship it is confounded: the only supervised
            model is also the only pure CNN, so delta_arch there is partly
            delta_obj wearing a different label.
  exact     the full architecture string (ViT-H/14 vs ViT-g/14 vs ...). Almost
            every pair differs, so this yields few same-arch pairs -- but the
            ones it yields are clean.

Usage:
    python scripts/architecture_vs_objective.py
    python scripts/architecture_vs_objective.py --out results/full_run/analysis/arch_vs_objective
"""
from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

METRICS = [
    "linear_cka", "kernel_cka", "svcca", "pwcca",
    "procrustes", "cosine_rsa", "distance_correlation",
]


def backbone(arch: str) -> str:
    """Coarse backbone family from the registry's architecture string."""
    a = arch.lower()
    if "resnet" in a and "swin" not in a:
        return "CNN"
    if "swin" in a or ("cnn" in a and "vit" not in a):
        return "hybrid"
    if "vit" in a or "beit" in a:
        return "ViT"
    return "other"


def load_registry(path: Path) -> dict:
    """display_name -> {objective, arch_exact, arch_backbone}.

    Parsed with a regex rather than PyYAML so this runs with the standard
    library alone; the registry is a flat two-level mapping.
    """
    text = path.read_text()
    reg = {}
    for key, body in re.findall(r"^  ([a-z0-9_]+):\n((?:    .*\n)+)", text, re.M):
        f = dict(re.findall(r"^\s+(\w+):\s*(.+?)\s*$", body, re.M))
        if "dim" not in f:
            continue
        arch = f.get("architecture", "")
        reg[f.get("display_name", key)] = {
            "key": key,
            "objective": f.get("family", "unknown"),
            "arch_exact": re.split(r"[ (@+]", arch)[0] or arch,
            "arch_backbone": backbone(arch),
        }
    return reg


def read_matrix(path: Path) -> dict:
    """Square CSV -> {(row, col): value}, with the row/col names."""
    rows = list(csv.reader(path.open()))
    cols = rows[0][1:]
    return {"names": cols,
            "v": {(r[0], c): float(x) for r in rows[1:] for c, x in zip(cols, r[1:])}}


def deltas(mat: dict, reg: dict, factor: str) -> dict:
    """delta for one similarity matrix and one design factor."""
    names = [n for n in mat["names"] if n in reg]
    same, diff = [], []
    for a, b in itertools.combinations(names, 2):
        v = mat["v"][(a, b)]
        (same if reg[a][factor] == reg[b][factor] else diff).append(v)
    if not same or not diff:
        return {"delta": float("nan"), "n_same": len(same), "n_diff": len(diff),
                "mean_same": float("nan"), "mean_diff": float("nan")}
    ms, md = sum(same) / len(same), sum(diff) / len(diff)
    return {"delta": ms - md, "n_same": len(same), "n_diff": len(diff),
            "mean_same": ms, "mean_diff": md}


def analyse(mdir: Path, reg: dict, label: str) -> list:
    rows = []
    for m in METRICS:
        f = mdir / f"{m}.csv"
        if not f.exists():
            continue
        mat = read_matrix(f)
        for factor, tag in (("objective", "objective"),
                            ("arch_backbone", "architecture (backbone)"),
                            ("arch_exact", "architecture (exact)")):
            d = deltas(mat, reg, factor)
            rows.append({"group": label, "metric": m, "factor": tag, **d})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/encoders.yaml"))
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    reg = load_registry(args.config)

    sources = [(args.results / "full_run/analysis/similarity/matrices",
                "CPTAC 10x/256px (flagship, 6 enc)")]
    for g in sorted((args.results / "groups").glob("*/similarity/matrices")):
        sources.append((g, g.parents[1].name))

    rows = []
    for d, lab in sources:
        if d.exists():
            rows += analyse(d, reg, lab)
    if not rows:
        raise SystemExit("no similarity matrices found")

    factors = ["objective", "architecture (backbone)", "architecture (exact)"]
    for lab in dict.fromkeys(r["group"] for r in rows):
        sub = [r for r in rows if r["group"] == lab]
        if all(r["delta"] != r["delta"] for r in sub):
            continue
        print(f"\n=== {lab}")
        seen = {}
        for r in sub:
            seen.setdefault(r["factor"], (r["n_same"], r["n_diff"]))
        print("    pairs: " + ", ".join(
            f"{f} {n[0]} same / {n[1]} diff" for f, n in seen.items()))
        print(f"    {'metric':22s}" + "".join(f"{f:>26s}" for f in factors))
        for m in METRICS:
            cells = {r["factor"]: r["delta"] for r in sub if r["metric"] == m}
            if not cells:
                continue
            line = f"    {m:22s}"
            for f in factors:
                v = cells.get(f, float("nan"))
                line += f"{'  --  ' if v != v else f'{v:+.3f}':>26s}"
            print(line)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        f = args.out / "arch_vs_objective.csv"
        with f.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {f}")
        m = args.out / "arch_vs_objective.md"
        m.write_text(as_markdown(rows, factors))
        print(f"wrote {m}")


def as_markdown(rows: list, factors: list) -> str:
    """The complete table, every group x metric x factor, as one markdown file."""
    out = [
        "# Δ_obj and Δ_arch — the complete table",
        "",
        "Every group with a similarity matrix, all seven metrics, all three",
        "groupings. Positive means encoders sharing that design factor are more",
        "similar to each other than to encoders that do not.",
        "",
        "    Δ = E[ S_ij | factor_i = factor_j ] − E[ S_ij | factor_i ≠ factor_j ]",
        "",
        "`n same / n diff` is how many encoder pairs each mean is taken over — read",
        "it before reading the Δ. A cell is blank where one side has no pairs: on the",
        "224px groups every encoder is `vision_ssl`, so Δ_obj has no contrast, and on",
        "most groups no two encoders share an exact architecture string.",
        "",
        "Generated by `scripts/architecture_vs_objective.py`. Narrative and caveats:",
        "`README.md` in this folder.",
        "",
    ]
    for lab in dict.fromkeys(r["group"] for r in rows):
        sub = [r for r in rows if r["group"] == lab]
        if all(r["delta"] != r["delta"] for r in sub):
            continue
        counts = {}
        for r in sub:
            counts.setdefault(r["factor"], (r["n_same"], r["n_diff"]))
        out += [f"## {lab}", ""]
        out.append("| metric | " + " | ".join(
            f"Δ {f}<br><sub>{counts[f][0]} same / {counts[f][1]} diff</sub>"
            for f in factors) + " |")
        out.append("|---|" + "---|" * len(factors))
        for metric in METRICS:
            cells = {r["factor"]: r["delta"] for r in sub if r["metric"] == metric}
            if not cells:
                continue
            vals = []
            for f in factors:
                v = cells.get(f, float("nan"))
                vals.append("—" if v != v else f"{v:+.3f}")
            out.append(f"| {metric} | " + " | ".join(vals) + " |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    main()
