"""Similarity recomputed within each subcohort instead of pooling them.

Every similarity matrix in this study is computed on a patch sample drawn
across a whole benchmark cohort, which mixes tissue types: CPTAC is ~50% lung
and TCGA ~52% breast. That makes any CPTAC-vs-TCGA comparison partly a
comparison of tissue composition rather than of cohort, and it cannot be
separated after the fact -- the pooled matrices carry no subcohort structure.

This script recomputes the seven metrics with the patch sample restricted to one
subcohort at a time, so tissue is held fixed. Sampling matches the pooled runs
(``--n-patches 20000 --max-slides 200 --max-samples 5000``) so the two are
directly comparable.

CPTAC-LSCC is skipped: it is dropped from the study.

Examples
--------
    python scripts/similarity_by_subcohort.py --out results/full_run/analysis/similarity_by_subcohort
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.features import FeatureStore  # noqa: E402
from utils.labels import build_cptac_labels, build_tcga_labels  # noqa: E402
from utils.pairwise import compute_all_similarity_matrices  # noqa: E402

#: The GDC manifests label TCGA subtypes bare ("LUAD"), which collides with the
#: CPTAC collection names. Prefixing keeps TCGA-LUAD and CPTAC-LUAD distinct --
#: they are different cohorts of the same tissue and must not be conflated.
#:
#: LUAD and LUSC are kept apart rather than merged into TCGA-NSCLC: NSCLC is a
#: task construct (the LUAD-vs-LUSC classification), not a tissue unit, and the
#: point of splitting by subcohort is to hold tissue fixed.
SUBCOHORT_ALIASES = {
    "LUAD": "TCGA-LUAD",
    "LUSC": "TCGA-LUSC",
    "BRCA": "TCGA-BRCA",
}

#: Every subcohort the study covers. Anything outside this set is refused rather
#: than silently added as a row -- that is how TCGA-RCC went unnoticed.
PAPER_SUBCOHORTS = {
    "TCGA-LUAD", "TCGA-LUSC", "TCGA-BRCA",
    "CPTAC-LUAD", "CPTAC-BRCA", "CPTAC-COAD",
}

#: Dropped as a *task* (see the note above CPTAC_TASKS in utils/labels.py), so
#: it is not broken out here. Note this is narrower than the study-wide
#: withholding in ``configs/excluded_slides.txt``: those slides never reach a
#: group at all, whereas CPTAC-LSCC still contributes patches to the pooled
#: CPTAC matrices. If that is not wanted, add them to the exclusion file.
EXCLUDED_SUBCOHORTS = {"CPTAC-LSCC"}

#: Below this a subcohort cannot support a stable patch sample, and its matrix
#: would say more about which slides were drawn than about the encoders.
MIN_SLIDES = 40


def subcohort_map(cohort: str) -> dict[str, str]:
    """slide_id -> subcohort, for one benchmark cohort."""
    if cohort == "cptac_benchmark":
        df = build_cptac_labels()
        raw = dict(zip(df["slide_id"], df["collection"]))
    else:
        df = build_tcga_labels()
        raw = dict(zip(df["slide_id"], df["subtype"]))
    return {k: SUBCOHORT_ALIASES.get(v, v) for k, v in raw.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", nargs="+", default=None,
                    help="feature-store groups; default: every multi-encoder group")
    ap.add_argument("--n-patches", type=int, default=20000)
    ap.add_argument("--max-slides", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--redo", action="store_true",
                    help="recompute groups that already have matrices on disk")
    ap.add_argument("--slide-encoders", dest="slide_enc", action="store_true",
                    default=True, help="also split the slide-encoder sets")
    args = ap.parse_args()

    store = FeatureStore()
    keys = args.groups or [k for k, g in store.groups.items() if len(g.encoders) >= 2]
    maps = {c: subcohort_map(c) for c in ("cptac_benchmark", "master_benchmark")}

    rows = []

    def run_one(tag, sub, sub_slides, load, n_enc, kind):
        """Compute and write one (set, subcohort) matrix; returns a summary row."""
        dest = args.out / tag / sub.replace("-", "_")
        if not args.redo and (dest / "matrices" / "linear_cka.csv").exists():
            print(f"  have {tag} / {sub} -- skipping (pass --redo to recompute)")
            return None
        print(f"\n=== {tag} / {sub} — {len(sub_slides)} slides, {n_enc} {kind} ===",
              flush=True)
        reps, names = load(sub_slides)
        mats = compute_all_similarity_matrices(
            reps, max_samples=args.max_samples, seed=args.seed, verbose=False,
        )
        (dest / "matrices").mkdir(parents=True, exist_ok=True)
        rec = {"set": tag, "kind": kind, "subcohort": sub,
               "slides": len(sub_slides), "encoders": len(reps)}
        for metric, S in mats.items():
            S.index = names
            S.columns = names
            S.to_csv(dest / "matrices" / f"{metric}.csv")
            v = S.values
            rec[metric] = float(np.mean(v[~np.eye(v.shape[0], dtype=bool)]))
        print("  " + "  ".join(f"{m}={rec[m]:.3f}" for m in sorted(mats)))
        return rec

    print(f"{len(keys)} candidate groups; TCGA-RCC and CPTAC-LSCC are withheld by "
          "configs/excluded_slides.txt\n")
    for key in sorted(keys):
        group = store.group(key)
        cohort = key.split("/")[0]
        smap = maps[cohort]
        slides = group.slides()
        by_sub: dict[str, list[str]] = {}
        for s in slides:
            sub = smap.get(s)
            if sub and sub not in EXCLUDED_SUBCOHORTS:
                if sub not in PAPER_SUBCOHORTS:
                    raise ValueError(
                        f"{key}: slide {s!r} maps to subcohort {sub!r}, which is not "
                        f"one of the subcohorts the study covers ({sorted(PAPER_SUBCOHORTS)}). "
                        "Either it should be withheld in configs/excluded_slides.txt "
                        "or SUBCOHORT_ALIASES needs an entry."
                    )
                by_sub.setdefault(sub, []).append(s)

        usable = {k: v for k, v in by_sub.items() if len(v) >= MIN_SLIDES}
        if len(usable) < 2:
            only = ", ".join(f"{k} ({len(v)})" for k, v in sorted(usable.items()))
            print(f"  skip {key}: only one usable subcohort [{only}] -- "
                  "splitting would just reproduce the pooled result")
            continue

        for sub, sub_slides in sorted(by_sub.items()):
            if len(sub_slides) < MIN_SLIDES:
                print(f"  skip {key} / {sub}: only {len(sub_slides)} slides")
                continue
            def load(sl, _g=group):
                reps = _g.sample_patches(
                    n_patches=args.n_patches, slides=sl,
                    max_slides=args.max_slides, seed=args.seed, verbose=True,
                )
                return reps, store.display_names(list(reps))

            rec = run_one(key.replace("/", "_"), sub, sub_slides, load,
                          len(group.encoders), "patch encoders")
            if rec:
                rows.append(rec)

    # ---- slide encoders -------------------------------------------------
    # A separate accessor from store.groups, so it needs its own loop or the
    # subcohort table silently covers only the patch encoders. Slide encoders
    # emit one vector per slide, so n is the slide count -- splitting by
    # subcohort cuts it further, and the CCA family is already near its floor at
    # the pooled n. Read CKA / RSA / Procrustes from these, not SVCCA or PWCCA.
    if args.slide_enc:
        for cohort in ("cptac_benchmark", "master_benchmark"):
            try:
                se = store.slide_encoders(cohort)
            except Exception as exc:
                print(f"  slide encoders {cohort}: unavailable ({exc})")
                continue
            smap = maps[cohort]
            by_sub: dict[str, list[str]] = {}
            for sl in se.slides():
                sub = smap.get(sl)
                if sub and sub not in EXCLUDED_SUBCOHORTS:
                    by_sub.setdefault(sub, []).append(sl)
            usable = {k: v for k, v in by_sub.items() if len(v) >= MIN_SLIDES}
            if len(usable) < 2:
                only = ", ".join(f"{k} ({len(v)})" for k, v in sorted(usable.items()))
                print(f"\n  skip slide encoders {cohort}: one usable subcohort "
                      f"[{only}] -- splitting reproduces the pooled result")
                continue
            for sub, sub_slides in sorted(usable.items()):
                def load(sl, _se=se):
                    reps, _ = _se.load(slides=sl, seed=args.seed, verbose=True)
                    return reps, store.display_names(list(reps))

                rec = run_one(f"slide_encoders_{cohort}", sub, sub_slides, load,
                              se.n_encoders, "slide encoders")
                if rec:
                    rows.append(rec)

    if rows:
        df = pd.DataFrame(rows)
        args.out.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out / "similarity_by_subcohort.csv", index=False)
        print(f"\nWrote {len(rows)} subcohort matrices -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
