"""Download CPTAC mutation labels from cBioPortal for the Phase VIII gene tasks.

Produces binary mutated / wild-type labels per patient for the standard CPTAC
mutation-prediction panel:

===========  ==============================
Cohort       Genes
===========  ==============================
CPTAC-BRCA   PIK3CA, MAP3K1, GATA3
CPTAC-COAD   KRAS, PIK3CA, TP53
CPTAC-LUAD   TP53, STK11, KRAS
CPTAC-LSCC   TP53, PIK3R1, KEAP1
===========  ==============================

Two correctness points that determine whether the labels mean anything:

* **The denominator is the sequenced cohort, not everyone.** A patient with no
  mutation record for gene X is wild-type *only if they were sequenced at all*.
  Patients absent from the study's ``_sequenced`` sample list have unknown
  status and are excluded rather than silently labelled wild-type — which would
  otherwise inflate the negative class with unprofiled cases.
* **Silent mutations do not count.** Only non-synonymous consequences are
  treated as mutated, matching how these benchmarks are normally defined.

Output: ``configs/clinical/cptac_mutations.csv``, one row per
``(patient, cohort)`` with one column per gene (1 mutated, 0 wild-type, blank
if not sequenced).

Examples
--------
    python scripts/download_cptac_mutations.py
    python scripts/download_cptac_mutations.py --cohorts CPTAC-LUAD
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://www.cbioportal.org/api"

#: cohort -> (cBioPortal study id, genes). Study ids were chosen for maximal
#: patient overlap with the WSI cohorts actually in the feature store.
COHORT_STUDIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "CPTAC-BRCA": ("breast_cptac_gdc", ("PIK3CA", "MAP3K1", "GATA3")),
    "CPTAC-COAD": ("coad_cptac_2019", ("KRAS", "PIK3CA", "TP53")),
    "CPTAC-LUAD": ("luad_cptac_gdc", ("TP53", "STK11", "KRAS")),
    "CPTAC-LSCC": ("lusc_cptac_2021", ("TP53", "PIK3R1", "KEAP1")),
}

#: Consequences that do not change the protein, so do not count as "mutated".
SILENT_TYPES = {
    "Silent",
    "Intron",
    "3'UTR",
    "5'UTR",
    "3'Flank",
    "5'Flank",
    "IGR",
    "RNA",
    "Splice_Region",
}


def _get(path: str, retries: int = 3):
    """GET a cBioPortal endpoint, with retries."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}/{path}", timeout=120) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _post(path: str, body: dict, retries: int = 3):
    """POST to a cBioPortal endpoint, with retries."""
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{API}/{path}",
                data=data,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def sequenced_patients(study_id: str) -> set[str]:
    """Patients with sequencing data, i.e. those who can be called wild-type.

    Parameters
    ----------
    study_id : str
        cBioPortal study id.

    Returns
    -------
    set of str
        Patient ids in the study's ``_sequenced`` sample list. Falls back to
        every patient in the study if that list is absent, with a warning —
        which would make wild-type calls unreliable.
    """
    lists = {s["sampleListId"] for s in _get(f"studies/{study_id}/sample-lists")}
    target = f"{study_id}_sequenced"
    if target not in lists:
        print(f"  WARNING: {target} not found; falling back to all patients")
        return {p["patientId"] for p in _get(f"studies/{study_id}/patients?pageSize=5000")}

    detail = _get(f"sample-lists/{target}")
    sample_ids = set(detail.get("sampleIds", []))
    samples = _get(f"studies/{study_id}/samples?pageSize=5000")
    return {s["patientId"] for s in samples if s["sampleId"] in sample_ids}


def fetch_mutations(study_id: str, genes: tuple[str, ...]) -> dict[str, set[str]]:
    """Fetch which patients carry a non-silent mutation in each gene.

    Parameters
    ----------
    study_id : str
        cBioPortal study id.
    genes : tuple of str
        HUGO gene symbols.

    Returns
    -------
    dict of str to set of str
        ``{gene: {patient_id, ...}}`` for patients with a non-silent mutation.
    """
    records = _post(
        f"molecular-profiles/{study_id}_mutations/mutations/fetch?projection=DETAILED",
        {"sampleListId": f"{study_id}_sequenced"},
    )

    wanted = set(genes)
    hits: dict[str, set[str]] = {g: set() for g in genes}
    for rec in records:
        symbol = (rec.get("gene") or {}).get("hugoGeneSymbol")
        if symbol not in wanted:
            continue
        if rec.get("mutationType") in SILENT_TYPES:
            continue
        hits[symbol].add(rec["patientId"])
    return hits


def main() -> None:
    """Download mutation labels and write the CSV."""
    parser = argparse.ArgumentParser(
        description="Download CPTAC mutation labels from cBioPortal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cohorts", nargs="+", default=list(COHORT_STUDIES), help="cohorts to fetch"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "clinical"
        / "cptac_mutations.csv",
        help="output CSV",
    )
    args = parser.parse_args()

    frames = []
    for cohort in args.cohorts:
        if cohort not in COHORT_STUDIES:
            raise SystemExit(f"unknown cohort {cohort!r}; expected {list(COHORT_STUDIES)}")
        study_id, genes = COHORT_STUDIES[cohort]
        print(f"\n=== {cohort} ({study_id}) ===")

        patients = sequenced_patients(study_id)
        print(f"  {len(patients)} sequenced patients")

        hits = fetch_mutations(study_id, genes)
        rows = []
        for pid in sorted(patients):
            row = {"patient_id": pid, "cohort": cohort, "study_id": study_id}
            for gene in genes:
                row[gene] = int(pid in hits[gene])
            rows.append(row)

        df = pd.DataFrame(rows)
        for gene in genes:
            n = int(df[gene].sum())
            print(f"  {gene:8s} mutated {n:4d}/{len(df):4d}  ({100 * n / max(len(df), 1):.1f}%)")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} patient rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
