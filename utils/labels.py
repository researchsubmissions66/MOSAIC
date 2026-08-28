"""Slide-level labels and downstream task definitions.

Builds slide -> label tables from the dataset manifests already on disk, with no
extra annotation files required:

* **TCGA** — GDC manifests are per-subtype (``gdc_manifest.LUAD.txt`` and
  ``gdc_manifest.LUSC.txt``), so the manifest a slide appears in *is* its
  subtype label.
* **CPTAC** — the TCIA manifests carry ``collection``, ``patient_id`` and
  ``cancer_type`` columns directly.

Task scope
----------
Every registered task is confined to a **single cancer type**. Tissue-of-origin
tasks (breast vs lung, or CPTAC's 4-class cancer type) are excluded on purpose:
they are close to trivial for any pathology encoder, so they saturate and
cannot separate one representation from another — which is exactly what Phase
VIII needs them to do.

Patient grouping
----------------
Every table carries a ``patient_id``, and every split must be grouped by it.
TCGA and CPTAC both contain multiple slides per patient, so a random
slide-level split puts the same patient on both sides and inflates every
downstream number. :func:`grouped_split` is the only splitter exposed here, and
it groups by patient by construction.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_CLINICAL_CSV",
    "DEFAULT_DATASETS_ROOT",
    "Task",
    "TASK_REGISTRY",
    "build_tcga_labels",
    "build_cptac_labels",
    "build_tcga_clinical_labels",
    "build_label_table",
    "available_tasks",
    "get_task",
    "grouped_split",
]

DEFAULT_DATASETS_ROOT = Path("/path/to/Datasets")
DEFAULT_CLINICAL_CSV = (
    Path(__file__).resolve().parents[1] / "configs" / "clinical" / "tcga_clinical.csv"
)
DEFAULT_MUTATION_CSV = (
    Path(__file__).resolve().parents[1] / "configs" / "clinical" / "cptac_mutations.csv"
)

#: cohort -> genes for the CPTAC mutation-prediction panel.
#:
#: CPTAC-LSCC is dropped from the study, not pending. Only 134 of its 1081
#: slides ever had features (~12%), which left every LSCC task either
#: unusable or severely imbalanced -- TP53 is mutated in 94% of LSCC, so that
#: task has almost no negative class, and `cptac_nsclc` ran 1139 LUAD vs 134
#: LSCC. Completing the cohort was possible (the slides download fine) but
#: was judged not worth the extraction; do not re-add LSCC tasks here without
#: reversing that decision.
CPTAC_MUTATION_PANEL: dict[str, tuple[str, ...]] = {
    "CPTAC-BRCA": ("PIK3CA", "MAP3K1", "GATA3"),
    "CPTAC-COAD": ("KRAS", "PIK3CA", "TP53"),
    "CPTAC-LUAD": ("TP53", "STK11", "KRAS"),
}


@dataclass(frozen=True)
class Task:
    """A downstream classification task.

    Attributes
    ----------
    name : str
        Registry key.
    store_cohort : str
        Feature-store cohort the slides live in (``'master_benchmark'`` for
        TCGA, ``'cptac_benchmark'`` for CPTAC).
    source : str
        Which label builder supplies the table, ``'tcga'`` or ``'cptac'``.
    column : str
        Label column in the table.
    classes : tuple of str
        Class values to keep; slides with any other value are dropped.
    description : str
        What the task is.
    notes : str
        Caveats worth knowing before reporting results.
    """

    name: str
    store_cohort: str
    source: str
    column: str
    classes: tuple[str, ...]
    description: str
    notes: str = ""
    cohort_filter: tuple[str, str] | None = None
    """Optional ``(column, value)`` restriction applied before labelling, used
    by the mutation tasks so that e.g. TP53 in LUAD and TP53 in LSCC stay
    separate tasks over disjoint slide sets."""


#: The downstream tasks.
#:
#: Tissue-of-origin tasks (TCGA breast-vs-lung, CPTAC 4-class cancer type) are
#: deliberately absent: they are close to trivial for any pathology encoder, so
#: they saturate and cannot discriminate between representations, which is the
#: entire point of Phase VIII here.
#:
#: Staging is registered per cohort (``tcga_brca_stage``, ``tcga_nsclc_stage``)
#: rather than as one pooled BRCA+LUAD+LUSC task, so "late stage" means one
#: thing within a label set rather than three organ-specific things at once.
TASK_REGISTRY: dict[str, Task] = {
    "tcga_nsclc": Task(
        name="tcga_nsclc",
        store_cohort="master_benchmark",
        source="tcga",
        column="subtype",
        classes=("LUAD", "LUSC"),
        description="TCGA NSCLC subtyping: adenocarcinoma vs squamous cell carcinoma",
        notes=(
            "The standard computational-pathology benchmark, and the only "
            "balanced task here (531/512 slides, 946 patients)."
        ),
    ),
    # --- requires configs/clinical/tcga_clinical.csv (scripts/download_clinical.py) ---
    "tcga_brca_subtype": Task(
        name="tcga_brca_subtype",
        store_cohort="master_benchmark",
        source="tcga_clinical",
        column="brca_subtype",
        classes=("IDC", "ILC"),
        description="TCGA breast histological subtype: invasive ductal vs invasive lobular",
        notes=(
            "The task that makes the 1126 BRCA slides useful. Imbalanced "
            "(769/189 slides, ~4:1) and genuinely hard. Cases recorded as mixed "
            "duct-and-lobular are dropped rather than forced into a class."
        ),
    ),
    "tcga_brca_stage": Task(
        name="tcga_brca_stage",
        store_cohort="master_benchmark",
        source="tcga_clinical",
        column="stage_group",
        classes=("early", "late"),
        description="TCGA breast AJCC pathologic stage: I/II vs III/IV",
        notes=(
            "1012 slides, 946 patients, 767/245. Stage is only weakly visible "
            "in morphology, so expect modest AUC — which is precisely why it "
            "discriminates between encoders where tissue typing saturates. "
            "Split out per cohort rather than pooled with lung, so the label "
            "means one thing."
        ),
        cohort_filter=("cohort", "TCGA-BRCA"),
    ),
    "tcga_nsclc_stage": Task(
        name="tcga_nsclc_stage",
        store_cohort="master_benchmark",
        source="tcga_clinical",
        column="stage_group",
        classes=("early", "late"),
        description="TCGA lung AJCC pathologic stage: I/II vs III/IV",
        notes=(
            "831 slides, 752 patients, 661/170. Pools LUAD and LUSC, which is "
            "standard for NSCLC staging — they are one staging entity even "
            "though tcga_nsclc treats them as separate classes."
        ),
        cohort_filter=("cohort", "TCGA-LUNG"),
    ),
}


_MUTATION_CAVEATS: dict[str, str] = {}

_GENE_CAVEATS: dict[tuple[str, str], str] = {}


def _register_mutation_tasks() -> None:
    """Add one binary mutation-prediction task per (cohort, gene) to the registry.

    Named ``cptac_<cohort>_<gene>``, e.g. ``cptac_luad_kras``. Each is confined
    to its own cohort, so TP53 in LUAD and TP53 in LSCC remain separate tasks
    over disjoint slides.
    """
    short = {
        "CPTAC-BRCA": "brca",
        "CPTAC-COAD": "coad",
        "CPTAC-LUAD": "luad",
    }
    for cohort, genes in CPTAC_MUTATION_PANEL.items():
        for gene in genes:
            name = f"cptac_{short[cohort]}_{gene.lower()}"
            TASK_REGISTRY[name] = Task(
                name=name,
                store_cohort="cptac_benchmark",
                source="cptac_mutation",
                column=f"mut_{gene}",
                classes=("WT", "MUT"),
                description=f"{cohort}: predict {gene} mutation status from morphology",
                notes=(
                    "Patients without sequencing are excluded, not called WT. "
                    "Mutation prediction from H&E is a weak-signal task — AUCs "
                    "in the 0.6-0.75 range are the norm — which makes it far "
                    "more discriminative between encoders than tissue typing."
                    + _MUTATION_CAVEATS.get(cohort, "")
                    + _GENE_CAVEATS.get((cohort, gene), "")
                ),
                cohort_filter=("collection", cohort),
            )


def build_tcga_labels(
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
) -> pd.DataFrame:
    """Build the TCGA slide -> subtype table from GDC manifests.

    Each ``gdc_manifest.<SUBTYPE>.txt`` lists the slides belonging to one
    subtype, so membership is the label. Manifests whose name carries a date
    rather than a subtype fall back to the cohort name.

    Parameters
    ----------
    datasets_root : str or pathlib.Path
        Directory containing the ``TCGA-*`` cohort folders.

    Returns
    -------
    pandas.DataFrame
        Columns ``slide_id``, ``patient_id``, ``cohort``, ``subtype``.
    """
    root = Path(datasets_root)
    rows = []
    for path in sorted(glob.glob(str(root / "TCGA-*" / "manifest" / "*.txt"))):
        p = Path(path)
        cohort = p.parts[-3]
        m = re.search(r"gdc_manifest[._]([A-Za-z]\w*)\.txt$", p.name)
        subtype = m.group(1).upper() if m else cohort.split("-", 1)[-1]

        with open(path) as fh:
            next(fh, None)  # header
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2:
                    continue
                slide_id = cols[1][:-4] if cols[1].endswith(".svs") else cols[1]
                rows.append(
                    {
                        "slide_id": slide_id,
                        "patient_id": slide_id[:12],  # TCGA-XX-YYYY
                        "cohort": cohort,
                        "subtype": subtype,
                    }
                )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset="slide_id").reset_index(drop=True)


def build_cptac_labels(
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
) -> pd.DataFrame:
    """Build the CPTAC slide -> cohort table from the TCIA manifests.

    Feature-store slide names are derived from the downloaded filenames and can
    be any of the manifest's identifier spellings (bare ``slide_id``,
    ``patient-slide``, or the URL basename), sometimes truncated. All spellings
    are indexed so the join is exact wherever possible, with a prefix fallback
    that is rejected if it would be ambiguous across cohorts.

    Parameters
    ----------
    datasets_root : str or pathlib.Path
        Directory containing ``CPTAC/CPTAC-*/manifest/*_manifest.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns ``slide_id``, ``patient_id``, ``collection``, ``cancer_type``,
        ``cancer_location``. ``slide_id`` holds every known spelling, so a
        caller joins on whichever the feature store uses.
    """
    root = Path(datasets_root)
    files = sorted(glob.glob(str(root / "CPTAC" / "*" / "manifest" / "*_manifest.csv")))
    if not files:
        raise FileNotFoundError(f"no CPTAC manifests under {root / 'CPTAC'}")

    frames = [pd.read_csv(f) for f in files]
    man = pd.concat(frames, ignore_index=True)

    rows = []
    for r in man.itertuples(index=False):
        base = Path(str(r.wsiimage_url)).name
        if base.endswith(".svs"):
            base = base[:-4]
        keys = {str(r.slide_id), f"{r.patient_id}-{r.slide_id}", base}
        for key in keys:
            rows.append(
                {
                    "slide_id": key,
                    "patient_id": str(r.patient_id),
                    "collection": str(r.collection),
                    "cancer_type": str(r.cancer_type),
                    "cancer_location": str(getattr(r, "cancer_location", "")),
                }
            )
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset="slide_id").reset_index(drop=True)


#: primary_diagnosis -> breast histological subtype. Only the unambiguous
#: strings are mapped; mixed duct-and-lobular and in-situ records map to
#: nothing and are dropped by the task builder.
_BRCA_SUBTYPE = {
    "Infiltrating duct carcinoma, NOS": "IDC",
    "Lobular carcinoma, NOS": "ILC",
    "Infiltrating lobular carcinoma, NOS": "ILC",
}


def _stage_group(stage) -> str | None:
    """Collapse an AJCC stage string to ``'early'`` (I/II) or ``'late'`` (III/IV).

    Parameters
    ----------
    stage : str or None
        e.g. ``'Stage IIIA'``.

    Returns
    -------
    str or None
        ``'early'``, ``'late'``, or None for unstaged / 'Stage X' / 'Stage 0'.

    Notes
    -----
    Substages carry a trailing letter ('Stage IIIA'), so the numeral must be
    matched as a prefix, not a whole word — a ``\\b`` here would silently drop
    every substaged case, which is most of them.
    """
    if not isinstance(stage, str):
        return None
    # Alternation is longest-first so 'IIIA' matches III, not II then I.
    m = re.match(r"Stage\s+(IV|III|II|I)(?![IV])", stage.strip())
    if not m:
        return None
    return "early" if m.group(1) in ("I", "II") else "late"


def build_tcga_clinical_labels(
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    clinical_csv: str | Path = DEFAULT_CLINICAL_CSV,
) -> pd.DataFrame:
    """Join TCGA slides to case-level clinical data downloaded from the GDC.

    Adds the derived columns the clinical tasks use: ``brca_subtype`` and
    ``stage_group``.

    Parameters
    ----------
    datasets_root : str or pathlib.Path
        Directory containing the ``TCGA-*`` cohort folders.
    clinical_csv : str or pathlib.Path
        CSV written by ``scripts/download_clinical.py``.

    Returns
    -------
    pandas.DataFrame
        The slide table with clinical columns joined on ``patient_id``.

    Raises
    ------
    FileNotFoundError
        If the clinical CSV is missing, with the command to produce it.
    """
    path = Path(clinical_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"clinical data not found at {path}. Download it with:\n"
            "    python scripts/download_clinical.py"
        )

    slides = build_tcga_labels(datasets_root)
    clinical = pd.read_csv(path)

    merged = slides.merge(clinical, on="patient_id", how="left", suffixes=("", "_clin"))

    # A case recorded with several distinct diagnoses is not a clean subtype
    # label, so resolve only the unambiguous ones.
    diag = merged["primary_diagnosis"].map(_BRCA_SUBTYPE)
    all_diag = merged["primary_diagnosis_all"].fillna("")
    ambiguous = all_diag.str.split(";").map(
        lambda parts: len({_BRCA_SUBTYPE.get(p) for p in parts if p in _BRCA_SUBTYPE}) > 1
    )
    merged["brca_subtype"] = diag.where(~ambiguous)

    merged["stage_group"] = merged["ajcc_pathologic_stage"].map(_stage_group)
    return merged


def build_cptac_mutation_labels(
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
    mutation_csv: str | Path = DEFAULT_MUTATION_CSV,
) -> pd.DataFrame:
    """Join CPTAC slides to per-patient mutation status.

    Adds one ``mut_<GENE>`` column per gene, valued ``'MUT'`` or ``'WT'``.
    Patients absent from the mutation table were not sequenced, so their status
    is left missing and the task builder drops them — calling them wild-type
    would pad the negative class with unprofiled cases.

    Parameters
    ----------
    datasets_root : str or pathlib.Path
        Directory containing ``CPTAC/``.
    mutation_csv : str or pathlib.Path
        CSV written by ``scripts/download_cptac_mutations.py``.

    Returns
    -------
    pandas.DataFrame
        The CPTAC slide table with ``mut_<GENE>`` columns.

    Raises
    ------
    FileNotFoundError
        If the mutation CSV is missing, with the command to produce it.
    """
    path = Path(mutation_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"CPTAC mutation labels not found at {path}. Download them with:\n"
            "    python scripts/download_cptac_mutations.py"
        )

    slides = build_cptac_labels(datasets_root)
    mut = pd.read_csv(path)

    genes = sorted({g for panel in CPTAC_MUTATION_PANEL.values() for g in panel})
    present = [g for g in genes if g in mut.columns]

    # One row per (patient, cohort); a gene not assayed in a cohort stays NaN.
    wide = mut.set_index(["patient_id", "cohort"])[present]
    merged = slides.merge(
        wide.reset_index().rename(columns={"cohort": "collection"}),
        on=["patient_id", "collection"],
        how="left",
    )

    for gene in present:
        merged[f"mut_{gene}"] = merged[gene].map({1: "MUT", 1.0: "MUT", 0: "WT", 0.0: "WT"})
    return merged


def build_label_table(
    source: str, datasets_root: str | Path = DEFAULT_DATASETS_ROOT
) -> pd.DataFrame:
    """Build a label table for one source.

    Parameters
    ----------
    source : {'tcga', 'cptac'}
        Which builder to use.
    datasets_root : str or pathlib.Path
        Dataset root.

    Returns
    -------
    pandas.DataFrame
        The label table, indexed by ``slide_id``.
    """
    if source == "tcga":
        df = build_tcga_labels(datasets_root)
    elif source == "cptac":
        df = build_cptac_labels(datasets_root)
    elif source == "tcga_clinical":
        df = build_tcga_clinical_labels(datasets_root)
    elif source == "cptac_mutation":
        df = build_cptac_mutation_labels(datasets_root)
    else:
        raise ValueError(
            f"unknown source {source!r}; expected 'tcga', 'cptac', "
            "'tcga_clinical' or 'cptac_mutation'"
        )
    return df.set_index("slide_id")


def available_tasks() -> list[str]:
    """List registered downstream tasks.

    Returns
    -------
    list of str
        Sorted task names.
    """
    return sorted(TASK_REGISTRY)


def get_task(
    name: str,
    slide_ids: Sequence[str] | None = None,
    datasets_root: str | Path = DEFAULT_DATASETS_ROOT,
) -> pd.DataFrame:
    """Resolve a task to a labelled slide table.

    Parameters
    ----------
    name : str
        Task name, from :func:`available_tasks`.
    slide_ids : sequence of str, optional
        Restrict to slides actually present in the feature store. Strongly
        recommended — the manifests list more slides than were extracted, and
        without this the returned table promises data that does not exist.
    datasets_root : str or pathlib.Path
        Dataset root.

    Returns
    -------
    pandas.DataFrame
        Columns ``slide_id``, ``patient_id``, ``label``, restricted to the
        task's classes.

    Raises
    ------
    KeyError
        If the task is unknown.
    """
    try:
        task = TASK_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown task {name!r}; available: {available_tasks()}"
        ) from None

    table = build_label_table(task.source, datasets_root)
    if slide_ids is not None:
        keep = [s for s in slide_ids if s in table.index]
        table = table.loc[keep]

    if task.cohort_filter is not None:
        column, value = task.cohort_filter
        if column not in table.columns:
            raise KeyError(f"cohort filter column {column!r} missing from the table")
        table = table[table[column] == value]

    if task.column not in table.columns:
        raise KeyError(
            f"task {name!r} needs column {task.column!r}, which is missing from "
            f"the {task.source!r} table"
        )

    out = table[[task.column, "patient_id"]].copy()
    out.columns = ["label", "patient_id"]
    out = out[out["label"].isin(task.classes)]
    return out.reset_index().rename(columns={"index": "slide_id"})


def grouped_split(
    labels: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 0,
    group_column: str = "patient_id",
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Split slides into train/test without splitting any patient across sides.

    Patient leakage is the classic way to overstate slide-level results: TCGA
    and CPTAC both have several slides per patient, and a slide-level random
    split lets a model recognise the patient rather than the disease.

    Parameters
    ----------
    labels : pandas.DataFrame
        Table with ``label`` and the group column.
    test_size : float, default 0.25
        Approximate fraction of *patients* held out.
    seed : int, default 0
        RNG seed.
    group_column : str, default 'patient_id'
        Column to group by.
    stratify : bool, default True
        Balance class proportions across the split by allocating patients
        class-by-class. Each patient is assigned by their majority label.

    Returns
    -------
    tuple of numpy.ndarray
        Positional index arrays ``(train_idx, test_idx)`` into ``labels``.
    """
    if group_column not in labels.columns:
        raise KeyError(f"{group_column!r} not in labels; columns: {list(labels.columns)}")

    rng = np.random.default_rng(seed)
    groups = labels[group_column].to_numpy()

    if stratify:
        # One label per patient (the majority), so patients can be allocated
        # per class without a patient landing on both sides.
        per_group = labels.groupby(group_column)["label"].agg(
            lambda s: s.value_counts().idxmax()
        )
        test_groups: set = set()
        for cls, members in per_group.groupby(per_group):
            ids = np.array(sorted(members.index))
            rng.shuffle(ids)
            n_test = int(round(len(ids) * test_size))
            test_groups.update(ids[:n_test].tolist())
    else:
        ids = np.array(sorted(set(groups)))
        rng.shuffle(ids)
        test_groups = set(ids[: int(round(len(ids) * test_size))].tolist())

    is_test = np.array([g in test_groups for g in groups])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


_register_mutation_tasks()
