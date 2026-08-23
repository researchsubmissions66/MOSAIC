"""Download TCGA clinical data from the GDC API.

Only open-access, case-level clinical fields are requested — no controlled data
and no authentication. The result is one row per patient, written to
``configs/clinical/tcga_clinical.csv``, which :mod:`utils.labels` then joins to
slides by patient barcode.

Multiple diagnoses per case
---------------------------
Some TCGA cases carry more than one diagnosis record (a breast case may be
recorded as both infiltrating duct and infiltrating lobular carcinoma). These
are kept as a semicolon-joined list in ``primary_diagnosis_all`` with a
``n_diagnoses`` count, rather than silently taking the first. A genuinely mixed
IDC/ILC case is not a clean label for an IDC-vs-ILC task and the task builder
drops it.

Examples
--------
    python scripts/download_clinical.py
    python scripts/download_clinical.py --projects TCGA-BRCA TCGA-LUAD TCGA-LUSC
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

GDC_CASES = "https://api.gdc.cancer.gov/cases"

#: Cohorts that actually have features extracted; see scripts/scan_features.py.
DEFAULT_PROJECTS = ["TCGA-BRCA", "TCGA-LUAD", "TCGA-LUSC"]

FIELDS = [
    "submitter_id",
    "project.project_id",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.gender",
    "demographic.race",
    "diagnoses.primary_diagnosis",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.tumor_grade",
    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
]


def fetch_cases(projects: list[str], page_size: int = 500, retries: int = 3) -> list[dict]:
    """Page through the GDC cases endpoint for the given projects.

    Parameters
    ----------
    projects : list of str
        GDC project ids, e.g. ``['TCGA-BRCA']``.
    page_size : int, default 500
        Records per request.
    retries : int, default 3
        Attempts per page before giving up.

    Returns
    -------
    list of dict
        Raw case records.
    """
    filters = {
        "op": "in",
        "content": {"field": "project.project_id", "value": projects},
    }

    records: list[dict] = []
    frm = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": str(page_size),
            "from": str(frm),
        }
        url = f"{GDC_CASES}?{urllib.parse.urlencode(params)}"

        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    payload = json.loads(resp.read().decode())
                break
            except Exception as exc:  # noqa: BLE001 - report and retry
                if attempt == retries - 1:
                    raise
                print(f"  retry {attempt + 1}/{retries} after {type(exc).__name__}")
                time.sleep(2 * (attempt + 1))

        hits = payload["data"]["hits"]
        page = payload["data"]["pagination"]
        records.extend(hits)
        print(f"  fetched {len(records)}/{page['total']}")

        frm += page_size
        if frm >= page["total"]:
            break
    return records


def flatten(records: list[dict]) -> pd.DataFrame:
    """Flatten GDC case records into one row per patient.

    Parameters
    ----------
    records : list of dict
        Output of :func:`fetch_cases`.

    Returns
    -------
    pandas.DataFrame
        One row per patient, indexed by ``patient_id`` (the TCGA barcode).
    """
    rows = []
    for rec in records:
        demo = rec.get("demographic") or {}
        diags = rec.get("diagnoses") or []
        first = diags[0] if diags else {}

        diagnoses = [d.get("primary_diagnosis") for d in diags if d.get("primary_diagnosis")]
        rows.append(
            {
                "patient_id": rec.get("submitter_id"),
                "project": (rec.get("project") or {}).get("project_id"),
                "primary_diagnosis": first.get("primary_diagnosis"),
                "primary_diagnosis_all": ";".join(sorted(set(diagnoses))),
                "n_diagnoses": len(diags),
                "morphology": first.get("morphology"),
                "tissue_or_organ_of_origin": first.get("tissue_or_organ_of_origin"),
                "ajcc_pathologic_stage": first.get("ajcc_pathologic_stage"),
                "ajcc_pathologic_t": first.get("ajcc_pathologic_t"),
                "ajcc_pathologic_n": first.get("ajcc_pathologic_n"),
                "ajcc_pathologic_m": first.get("ajcc_pathologic_m"),
                "tumor_grade": first.get("tumor_grade"),
                "age_at_diagnosis": first.get("age_at_diagnosis"),
                "days_to_last_follow_up": first.get("days_to_last_follow_up"),
                "vital_status": demo.get("vital_status"),
                "days_to_death": demo.get("days_to_death"),
                "gender": demo.get("gender"),
                "race": demo.get("race"),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset="patient_id")

    # Overall-survival convenience columns for later survival analysis.
    df["event"] = (df["vital_status"] == "Dead").astype("Int64")
    df["time"] = df["days_to_death"].fillna(df["days_to_last_follow_up"])
    return df.sort_values("patient_id").reset_index(drop=True)


def main() -> None:
    """Fetch clinical data and write the CSV."""
    parser = argparse.ArgumentParser(
        description="Download open-access TCGA clinical data from the GDC API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--projects", nargs="+", default=DEFAULT_PROJECTS, help="GDC project ids"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "clinical"
        / "tcga_clinical.csv",
        help="output CSV",
    )
    args = parser.parse_args()

    print(f"Fetching clinical data for {args.projects} ...")
    records = fetch_cases(args.projects)
    df = flatten(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {len(df)} patients to {args.out}\n")
    print(df.groupby("project").size().to_string())

    print("\n=== primary_diagnosis by project (top 6) ===")
    for proj, grp in df.groupby("project"):
        top = grp["primary_diagnosis"].value_counts().head(6)
        print(f"\n{proj}:")
        print(top.to_string())

    mixed = df[df["n_diagnoses"] > 1]
    if len(mixed):
        print(f"\n{len(mixed)} patients have multiple diagnosis records.")


if __name__ == "__main__":
    sys.exit(main())
