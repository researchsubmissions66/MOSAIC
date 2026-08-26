"""End-to-end MOSAIC study runner.

Chains every stage of the study into one reproducible run, writing everything
under a single output directory with a manifest recording exactly what was run.

Stages
------
``inventory``
    Scan the feature store and verify coordinate pairing.
``similarity``
    Representational similarity across encoders (7 metrics, heatmaps,
    clustering, MDS).
``magnification``
    The same similarity and alignment analysis repeated at 5x, 10x and 20x.
``alignment``
    Fit and evaluate shared latent spaces.
``transfer``
    Cross-model transfer: encode with A, decode as B, scored by fidelity,
    retrieval against B's real index, and a probe trained on B's real features.
``retrieval``
    Cross-model retrieval, aligned vs unaligned.
``downstream``
    Slide-level MIL across single / concat / shared input conditions, for each
    requested task.

Each stage is a separate subprocess, so one failure does not abort the rest and
stages can be re-run individually. Results land in
``<out>/<stage>/`` and a summary of what succeeded is written to
``<out>/run_manifest.json``.

Examples
--------
Everything, on the flagship feature group::

    python scripts/run_study.py --out results/full

A quick end-to-end smoke test (small samples, few epochs, minutes not hours)::

    python scripts/run_study.py --out results/smoke --preset smoke

Selected stages only::

    python scripts/run_study.py --out results/r1 --stages similarity retrieval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

ALL_STAGES = [
    "inventory",
    "similarity",
    "magnification",
    "alignment",
    "transfer",
    "retrieval",
    "downstream",
]

#: Presets trade run time against thoroughness. 'smoke' exists so the whole
#: chain can be validated end to end before committing hours of compute to it.
PRESETS: dict[str, dict] = {
    "smoke": {
        "n_patches": 3000,
        "max_slides": 40,
        "max_samples": 2000,
        "latent_dim": 32,
        "query_samples": 1000,
        "mil_epochs": 5,
        "mil_max_patches": 500,
        "align_patches": 3000,
        "tasks": ["tcga_nsclc"],
        "methods": ["joint_pca", "gcca"],
        "mil": ["abmil"],  # smoke only: one head, for speed
    },
    "standard": {
        "n_patches": 20000,
        "max_slides": 200,
        "max_samples": 5000,
        "latent_dim": 64,
        "query_samples": 2000,
        "mil_epochs": 50,
        "mil_max_patches": 2000,
        "align_patches": 20000,
        "tasks": ["tcga_nsclc", "tcga_brca_subtype", "cptac_luad_tp53", "cptac_luad_kras"],
        "methods": ["joint_pca", "gcca", "procrustes"],
        "mil": ["abmil", "transmil"],
    },
    "full": {
        "n_patches": 50000,
        "max_slides": 500,
        "max_samples": 8000,
        "latent_dim": 64,
        "query_samples": 5000,
        "mil_epochs": 80,
        "mil_max_patches": 4000,
        "align_patches": 50000,
        "tasks": None,  # every registered task
        "methods": ["joint_pca", "gcca", "mcca", "procrustes", "autoencoder"],
        "mil": ["abmil", "transmil"],
    },
}


def run_stage(name: str, cmd: list[str], log_dir: Path, dry_run: bool) -> dict:
    """Run one stage as a subprocess, teeing its output to a log file.

    Parameters
    ----------
    name : str
        Stage name, used for the log filename.
    cmd : list of str
        Command to run.
    log_dir : pathlib.Path
        Directory for stage logs.
    dry_run : bool
        Print the command without running it.

    Returns
    -------
    dict
        Stage record with command, status, duration and log path.
    """
    printable = " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 78}\n[{name}] {printable}\n{'=' * 78}", flush=True)

    if dry_run:
        return {"stage": name, "command": printable, "status": "skipped (dry run)"}

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    t0 = time.perf_counter()

    with open(log_path, "w") as log:
        log.write(f"$ {printable}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()

    elapsed = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"[{name}] {status} in {elapsed / 60:.1f} min  (log: {log_path})", flush=True)

    return {
        "stage": name,
        "command": printable,
        "status": status,
        "returncode": proc.returncode,
        "minutes": round(elapsed / 60, 2),
        "log": str(log_path),
    }


def build_commands(args, cfg: dict) -> list[tuple[str, list[str]]]:
    """Assemble the command for each requested stage.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    cfg : dict
        Resolved preset configuration.

    Returns
    -------
    list of tuple
        ``(stage_name, command)`` pairs, in execution order.
    """
    py = sys.executable
    out = args.out
    commands: list[tuple[str, list[str]]] = []

    def stage_out(name: str) -> Path:
        return out / name

    if "inventory" in args.stages:
        commands.append(
            (
                "inventory",
                [py, str(SCRIPTS / "scan_features.py"), "--verify", "2",
                 "--out", str(out / "feature_inventory.json")],
            )
        )

    if "similarity" in args.stages:
        commands.append(
            (
                "similarity",
                [py, str(SCRIPTS / "representation_similarity.py"),
                 "--group", args.group,
                 "--n-patches", str(cfg["n_patches"]),
                 "--max-slides", str(cfg["max_slides"]),
                 "--max-samples", str(cfg["max_samples"]),
                 "--seed", str(args.seed),
                 "--no-umap",
                 "--out", str(stage_out("similarity"))],
            )
        )

    if "magnification" in args.stages:
        commands.append(
            (
                "magnification",
                [py, str(SCRIPTS / "magnification_ablation.py"),
                 "--series", args.series,
                 "--n-patches", str(cfg["n_patches"]),
                 "--max-slides", str(cfg["max_slides"]),
                 "--max-samples", str(cfg["max_samples"]),
                 "--latent-dim", str(cfg["latent_dim"]),
                 "--seed", str(args.seed),
                 "--alignment", *cfg["methods"],
                 "--out", str(stage_out("magnification"))],
            )
        )

    if "alignment" in args.stages:
        commands.append(
            (
                "alignment",
                [py, str(SCRIPTS / "shared_latent_space.py"),
                 "--group", args.group,
                 "--n-patches", str(cfg["n_patches"]),
                 "--max-slides", str(cfg["max_slides"]),
                 "--latent-dim", str(cfg["latent_dim"]),
                 "--methods", *cfg["methods"],
                 "--seed", str(args.seed),
                 "--save-aligners",
                 "--out", str(stage_out("alignment"))],
            )
        )

    if "transfer" in args.stages:
        commands.append(
            (
                "transfer",
                [py, str(SCRIPTS / "cross_model_transfer.py"),
                 "--group", args.group,
                 "--n-patches", str(cfg["n_patches"]),
                 "--max-slides", str(cfg["max_slides"]),
                 "--latent-dim", str(cfg["latent_dim"]),
                 "--aligner", cfg["methods"][-1],
                 "--query-samples", str(cfg["query_samples"]),
                 "--seed", str(args.seed),
                 "--out", str(stage_out("transfer"))],
            )
        )

    if "retrieval" in args.stages:
        commands.append(
            (
                "retrieval",
                [py, str(SCRIPTS / "cross_model_retrieval.py"),
                 "--group", args.group,
                 "--n-patches", str(cfg["n_patches"]),
                 "--max-slides", str(cfg["max_slides"]),
                 "--latent-dim", str(cfg["latent_dim"]),
                 "--methods", *cfg["methods"],
                 "--query-samples", str(cfg["query_samples"]),
                 "--seed", str(args.seed),
                 "--out", str(stage_out("retrieval"))],
            )
        )

    if "downstream" in args.stages:
        for task in cfg["tasks"]:
            group = args.group
            if group == "best":
                # Tasks are cohort-specific, so let the driver pick the right
                # group for the task's cohort rather than forcing one.
                group = "best"
            commands.append(
                (
                    f"downstream_{task}",
                    [py, str(SCRIPTS / "downstream_mil.py"),
                     "--task", task,
                     "--group", group,
                     "--latent-dim", str(cfg["latent_dim"]),
                     "--max-patches", str(cfg["mil_max_patches"]),
                     "--epochs", str(cfg["mil_epochs"]),
                     "--align-patches", str(cfg["align_patches"]),
                     "--mil", *cfg["mil"],
                     "--seed", str(args.seed),
                     "--out", str(stage_out("downstream") / task)],
                )
            )

    return commands


def resolve_tasks(cfg: dict) -> dict:
    """Fill in the task list when the preset asks for all of them."""
    if cfg["tasks"] is None:
        sys.path.insert(0, str(ROOT))
        from utils.labels import available_tasks

        cfg = dict(cfg)
        cfg["tasks"] = available_tasks()
    return cfg


def main() -> int:
    """Run the requested stages and write the manifest."""
    parser = argparse.ArgumentParser(
        description="Run the MOSAIC study end to end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--preset",
        default="standard",
        choices=sorted(PRESETS),
        help="'smoke' validates the whole chain in minutes; 'full' is the paper run",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=ALL_STAGES,
        choices=ALL_STAGES,
        help="stages to run",
    )
    parser.add_argument(
        "--group", default="best", help="feature-store group for the patch-level stages"
    )
    parser.add_argument(
        "--series", default="best", help="magnification series for the ablation"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="override the preset's task list"
    )
    parser.add_argument("--seed", type=int, default=0, help="shared seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands without running"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="abort the run at the first failing stage instead of continuing",
    )
    args = parser.parse_args()

    cfg = resolve_tasks(PRESETS[args.preset])
    if args.tasks is not None:
        cfg = dict(cfg, tasks=args.tasks)

    args.out.mkdir(parents=True, exist_ok=True)
    commands = build_commands(args, cfg)

    print(f"MOSAIC study — preset '{args.preset}', {len(commands)} stages")
    print(f"output: {args.out.resolve()}")
    print(f"tasks : {cfg['tasks']}")

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    records = []

    for name, cmd in commands:
        record = run_stage(name, cmd, args.out / "logs", args.dry_run)
        records.append(record)
        if args.stop_on_failure and record.get("returncode", 0) != 0:
            print(f"\nAborting: stage {name!r} failed and --stop-on-failure is set.")
            break

    manifest = {
        "preset": args.preset,
        "config": cfg,
        "group": args.group,
        "series": args.series,
        "seed": args.seed,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "total_minutes": round((time.perf_counter() - t0) / 60, 2),
        "stages": records,
    }
    if not args.dry_run:
        (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    failed = [r for r in records if r.get("returncode", 0) != 0]
    print(f"\n{'=' * 78}")
    print(f"Finished in {manifest['total_minutes']:.1f} min — "
          f"{len(records) - len(failed)}/{len(records)} stages ok")
    for r in records:
        print(f"  {r['stage']:32s} {r['status']}")
    if not args.dry_run:
        print(f"\nManifest: {args.out / 'run_manifest.json'}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
