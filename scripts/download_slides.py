"""Download a handful of TCGA slides from the GDC, for layer-wise analysis.

Phase IV needs actual patch images: the feature store holds only final
embeddings and patch coordinates, so intermediate block activations cannot be
recovered from it. This fetches a few whole-slide images so patches can be
re-cropped at exactly the coordinates trident used.

Slides are chosen to be **already present in the feature store**, which makes
the extraction self-validating: activations pooled from the final block should
reproduce the stored embeddings for the same patches.

Only open-access TCGA diagnostic slides are downloaded, and the default is the
smallest few — a full cohort is terabytes.

Examples
--------
    python scripts/download_slides.py --n 4
    python scripts/download_slides.py --n 2 --out /path/to/slides
"""

from __future__ import annotations

import argparse
import glob
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GDC_DATA = "https://api.gdc.cancer.gov/data"
DEFAULT_OUT = Path("/path/to/Datasets/tcga_slides_layerwise")


def candidate_slides(
    datasets_root: Path, store_glob: str
) -> list[tuple[int, str, str]]:
    """Find slides that are both in the GDC manifests and in the feature store.

    Parameters
    ----------
    datasets_root : pathlib.Path
        Directory holding the ``TCGA-*`` cohort folders with GDC manifests.
    store_glob : str
        Glob matching one encoder's feature files, used to establish which
        slides have features.

    Returns
    -------
    list of tuple
        ``(size_bytes, file_uuid, slide_id)``, ascending by size.
    """
    store = {Path(p).stem for p in glob.glob(store_glob)}
    rows = []
    for manifest in glob.glob(str(datasets_root / "TCGA-*" / "manifest" / "*.txt")):
        with open(manifest) as fh:
            next(fh, None)
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 4:
                    continue
                name = cols[1][:-4] if cols[1].endswith(".svs") else cols[1]
                if name in store:
                    try:
                        rows.append((int(cols[3]), cols[0], name))
                    except ValueError:
                        continue
    rows.sort()
    return rows


def download(uuid: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream one file from the GDC data endpoint.

    Parameters
    ----------
    uuid : str
        GDC file uuid.
    dest : pathlib.Path
        Destination path.
    chunk : int, default 1 MiB
        Streaming chunk size.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{GDC_DATA}/{uuid}", timeout=300) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r    {done / 1e6:7.1f}/{total / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
    print()
    tmp.rename(dest)


def main() -> int:
    """Download the smallest N feature-store slides."""
    parser = argparse.ArgumentParser(
        description="Download a few TCGA slides for layer-wise analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=4, help="how many slides")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="destination")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("/path/to/Datasets"),
        help="root holding the TCGA-* manifest folders",
    )
    parser.add_argument(
        "--store-glob",
        default="/path/to/trident_features/master_benchmark/"
        "10x_256px_0px_overlap/features_uni_v2/*.h5",
        help="glob establishing which slides have features",
    )
    parser.add_argument(
        "--max-mb", type=float, default=200.0, help="skip slides larger than this"
    )
    args = parser.parse_args()

    rows = candidate_slides(args.datasets_root, args.store_glob)
    if not rows:
        print("No slides found that are both in a manifest and in the feature store.")
        return 1

    picked = [r for r in rows if r[0] / 1e6 <= args.max_mb][: args.n]
    total_mb = sum(r[0] for r in picked) / 1e6
    print(f"{len(rows)} candidates; downloading {len(picked)} ({total_mb:.1f} MB total)\n")

    args.out.mkdir(parents=True, exist_ok=True)
    for size, uuid, name in picked:
        dest = args.out / f"{name}.svs"
        if dest.exists():
            print(f"  {name[:44]} already present, skipping")
            continue
        print(f"  {name[:44]}  ({size / 1e6:.1f} MB)")
        download(uuid, dest)

    print(f"\nSlides in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
