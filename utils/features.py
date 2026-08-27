"""Discovery and loading of trident-extracted patch features.

The feature store on ``alt`` is laid out by trident as::

    <feature_root>/<cohort>/<mag>x_<patch>px_<overlap>px_overlap/features_<encoder>/<slide>.h5

with each HDF5 file holding ``features`` of shape ``(n_patches, dim)`` and
``coords`` of shape ``(n_patches, 2)``.

The pairing constraint
----------------------
Everything in Phases I and V requires **row-paired** embeddings: row i must be
the same tissue patch under every model. Trident writes one coordinate grid per
``(magnification, patch_size)`` directory and every encoder run against that
directory inherits it, so encoders sharing a directory are paired by row index
and encoders in different directories are not — a 224px grid and a 256px grid
cover different tissue.

That makes ``(cohort, magnification, patch_size)`` the natural unit of
analysis. This module calls it a :class:`FeatureGroup`, and a group is the
largest set of models that can be compared without re-extracting anything.
:meth:`FeatureGroup.verify_alignment` checks the coordinate arrays really do
match rather than trusting the layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DEFAULT_CONFIG",
    "EncoderInfo",
    "FeatureGroup",
    "FeatureStore",
    "MagnificationSeries",
    "SlideEncoderSet",
    "load_encoder_config",
]

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "encoders.yaml"

#: trident directory naming, e.g. "20x_256px_0px_overlap".
_DIR_RE = re.compile(r"^(?P<mag>[\d.]+)x_(?P<patch>\d+)px_(?P<overlap>\d+)px_overlap$")


#: Slides withheld from every analysis. The features remain on disk; they are
#: simply never sampled. Filtering here rather than in each script means the
#: exclusion cannot be forgotten by one caller -- similarity, alignment,
#: retrieval, transfer, subspace and downstream all resolve slides through
#: :meth:`FeatureGroup.slides`.
EXCLUDED_SLIDES_FILE = Path(__file__).resolve().parents[1] / "configs" / "excluded_slides.txt"


@lru_cache(maxsize=1)
def excluded_slides(path: str | None = None) -> frozenset[str]:
    """Slide ids listed in ``configs/excluded_slides.txt``.

    Returns
    -------
    frozenset of str
        Excluded slide ids; empty if the file is absent.
    """
    f = Path(path) if path else EXCLUDED_SLIDES_FILE
    if not f.exists():
        return frozenset()
    return frozenset(
        line.strip()
        for line in f.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


@dataclass(frozen=True)
class EncoderInfo:
    """Static metadata for one foundation model.

    Attributes
    ----------
    name : str
        Registry key, matching the ``features_<name>`` directory on disk.
    display_name : str
        Human-readable name for figures.
    dim : int
        Embedding dimensionality.
    family : str
        ``'vision_ssl'``, ``'vision_language'`` or ``'supervised'`` — the
        grouping used to test whether the pretraining objective drives
        representational similarity.
    objective : str
        Pretraining objective.
    architecture : str
        Backbone description.
    hf_id : str or None
        Hugging Face model id, or None for models without one.
    extra : dict
        Any remaining fields from the config.
    """

    name: str
    display_name: str
    dim: int
    family: str = "unknown"
    objective: str = "unknown"
    architecture: str = "unknown"
    hf_id: str | None = None
    extra: dict = field(default_factory=dict)


def load_encoder_config(path: str | Path | None = None) -> tuple[Path, dict[str, EncoderInfo]]:
    """Read the encoder registry YAML.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Config file. Defaults to ``configs/encoders.yaml``.

    Returns
    -------
    tuple
        ``(feature_root, {name: EncoderInfo})``.
    """
    import yaml

    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    root = Path(cfg["feature_root"])
    encoders = {}
    for name, spec in (cfg.get("encoders") or {}).items():
        spec = dict(spec or {})
        encoders[name] = EncoderInfo(
            name=name,
            display_name=spec.pop("display_name", name),
            dim=int(spec.pop("dim")),
            family=spec.pop("family", "unknown"),
            objective=spec.pop("objective", "unknown"),
            architecture=spec.pop("architecture", "unknown"),
            hf_id=spec.pop("hf_id", None),
            extra=spec,
        )
    return root, encoders


@dataclass
class FeatureGroup:
    """A set of encoders sharing one coordinate grid, hence row-pairable.

    Attributes
    ----------
    cohort : str
        Top-level dataset directory, e.g. ``'cptac_benchmark'``.
    magnification : float
        Target magnification.
    patch_size : int
        Patch size in pixels at that magnification.
    path : pathlib.Path
        Directory holding the ``features_<encoder>`` subdirectories.
    encoders : dict of str to pathlib.Path
        Available encoders and their feature directories.
    """

    cohort: str
    magnification: float
    patch_size: int
    path: Path
    encoders: dict[str, Path]

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``'cptac_benchmark/10x_256px'``."""
        mag = int(self.magnification) if self.magnification == int(self.magnification) else self.magnification
        return f"{self.cohort}/{mag}x_{self.patch_size}px"

    @property
    def n_encoders(self) -> int:
        """Number of available encoders."""
        return len(self.encoders)

    def slides(self, encoders: Sequence[str] | None = None) -> list[str]:
        """Slide ids present for *every* requested encoder.

        Slides missing from any one encoder are excluded, because a slide that
        is not in all of them cannot contribute paired rows.

        Parameters
        ----------
        encoders : sequence of str, optional
            Restrict to these encoders. Defaults to all in the group.

        Returns
        -------
        list of str
            Sorted slide ids.
        """
        names = list(encoders) if encoders else list(self.encoders)
        self._check_encoders(names)

        sets = []
        for name in names:
            sets.append({p.stem for p in self.encoders[name].glob("*.h5")})
        if not sets:
            return []
        return sorted(set.intersection(*sets) - excluded_slides())

    def load_slide(
        self,
        slide_id: str,
        encoders: Sequence[str] | None = None,
        with_coords: bool = False,
    ) -> dict[str, np.ndarray]:
        """Load every encoder's features for one slide.

        Parameters
        ----------
        slide_id : str
            Slide id (the h5 filename without extension).
        encoders : sequence of str, optional
            Restrict to these encoders.
        with_coords : bool, default False
            Also return the patch coordinates under key ``'coords'``.

        Returns
        -------
        dict of str to numpy.ndarray
            ``{encoder: (n_patches, dim)}``, row-paired across encoders.

        Raises
        ------
        ValueError
            If the encoders disagree on the patch count for this slide, which
            means the rows are not paired and the slide must be skipped.
        """
        import h5py

        names = list(encoders) if encoders else list(self.encoders)
        self._check_encoders(names)

        out: dict[str, np.ndarray] = {}
        coords = None
        for name in names:
            with h5py.File(self.encoders[name] / f"{slide_id}.h5", "r") as h:
                out[name] = h["features"][:]
                if coords is None and with_coords:
                    coords = h["coords"][:]

        counts = {name: arr.shape[0] for name, arr in out.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"patch counts disagree for slide {slide_id!r}: {counts}. "
                "Rows are not paired; exclude this slide."
            )
        if with_coords:
            out["coords"] = coords
        return out

    def verify_alignment(
        self, slide_id: str, encoders: Sequence[str] | None = None
    ) -> bool:
        """Check that all encoders share identical coordinates for a slide.

        Row-pairing is an assumption everything downstream rests on, so it is
        worth verifying on a few slides rather than inferring it from the
        directory layout.

        Parameters
        ----------
        slide_id : str
            Slide to check.
        encoders : sequence of str, optional
            Restrict to these encoders.

        Returns
        -------
        bool
            True if every encoder reports the same coordinate array.
        """
        import h5py

        names = list(encoders) if encoders else list(self.encoders)
        self._check_encoders(names)

        ref = None
        for name in names:
            with h5py.File(self.encoders[name] / f"{slide_id}.h5", "r") as h:
                c = h["coords"][:]
            if ref is None:
                ref = c
            elif c.shape != ref.shape or not np.array_equal(c, ref):
                return False
        return True

    def sample_patches(
        self,
        n_patches: int = 20_000,
        encoders: Sequence[str] | None = None,
        slides: Sequence[str] | None = None,
        max_slides: int | None = 200,
        seed: int = 0,
        dtype=np.float32,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
        """Draw a row-paired patch sample spanning many slides.

        This is the entry point for Phases I and V: the returned dict plugs
        straight into :func:`utils.compute_all_similarity_matrices` and into any
        aligner's ``fit``.

        Patches are drawn per slide rather than from a global pool, so the
        sample is spread across the cohort instead of concentrated in whichever
        slides happen to be largest — a real risk here, where patch counts per
        slide range from a few hundred to nearly 20,000.

        Parameters
        ----------
        n_patches : int, default 20000
            Total patches to return.
        encoders : sequence of str, optional
            Restrict to these encoders. Defaults to all in the group.
        slides : sequence of str, optional
            Restrict to these slides. Defaults to all shared slides.
        max_slides : int or None, default 200
            Sample from at most this many slides (each open is a separate file
            read, so this bounds I/O). ``None`` uses all.
        seed : int, default 0
            RNG seed. Fixing it keeps the patch set identical across metrics
            and across aligners, which is required for their results to be
            comparable.
        dtype : numpy dtype, default ``np.float32``
            Output dtype. Features are stored as float32; float64 doubles
            memory and the metrics upcast internally anyway.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        dict of str to numpy.ndarray
            ``{encoder: (n_patches, dim)}``, row-paired.
        """
        names = list(encoders) if encoders else list(self.encoders)
        self._check_encoders(names)

        available = list(slides) if slides else self.slides(names)
        if not available:
            raise ValueError(f"no shared slides for encoders {names}")

        rng = np.random.default_rng(seed)
        if max_slides is not None and len(available) > max_slides:
            chosen = sorted(
                rng.choice(len(available), size=max_slides, replace=False).tolist()
            )
            available = [available[i] for i in chosen]

        per_slide = max(1, int(np.ceil(n_patches / len(available))))
        chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
        total = 0

        for i, slide_id in enumerate(available):
            if total >= n_patches:
                break
            try:
                feats = self.load_slide(slide_id, encoders=names)
            except (ValueError, OSError) as exc:
                if verbose:
                    print(f"  skipping {slide_id}: {exc}")
                continue

            n_avail = feats[names[0]].shape[0]
            take = min(per_slide, n_avail, n_patches - total)
            idx = np.sort(rng.choice(n_avail, size=take, replace=False))

            for name in names:
                chunks[name].append(feats[name][idx].astype(dtype, copy=False))
            total += take

            if verbose and (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(available)} slides, {total}/{n_patches} patches")

        if total == 0:
            raise ValueError("no patches could be loaded")
        if verbose:
            print(f"  collected {total} patches from {len(chunks[names[0]])} slides")

        return {name: np.vstack(parts) for name, parts in chunks.items()}

    def _check_encoders(self, names: Iterable[str]) -> None:
        missing = [n for n in names if n not in self.encoders]
        if missing:
            raise KeyError(
                f"{missing} not available in {self.key}; "
                f"present: {sorted(self.encoders)}"
            )

    def __repr__(self) -> str:
        return (
            f"FeatureGroup({self.key}, {self.n_encoders} encoders: "
            f"{sorted(self.encoders)})"
        )


@dataclass
class MagnificationSeries:
    """One experiment repeated at several magnifications.

    Holds the groups that share a cohort and patch size but differ in
    magnification, together with the encoder set common to all of them.

    Two things have to be held fixed for the comparison to isolate
    magnification, and both are enforced here rather than left to the caller:

    * **The encoder set.** ``cptac_benchmark`` has six encoders at 10x/256px
      but five at 5x and 20x, because CONCH was only run at 10x. Comparing a
      six-model similarity matrix against a five-model one would confound
      magnification with which models are in the matrix, so :attr:`encoders`
      is the intersection.
    * **The slides.** :meth:`shared_slides` returns those present at every
      magnification, so the same tissue is sampled throughout.

    What cannot be held fixed is the patches themselves: each magnification has
    its own coordinate grid (at 256px, one 5x patch covers the same tissue as
    four 10x patches or sixteen 20x ones), so the samples are different views
    of the same slides rather than the same patches. That is inherent to the
    question, not a flaw — each magnification is an independent replication of
    the experiment.

    Attributes
    ----------
    cohort : str
        Dataset the series is drawn from.
    patch_size : int
        Patch size in pixels, constant across the series.
    groups : dict of float to FeatureGroup
        ``{magnification: group}``.
    encoders : list of str
        Encoders available at *every* magnification in the series.
    """

    cohort: str
    patch_size: int
    groups: dict[float, FeatureGroup]
    encoders: list[str]

    @property
    def magnifications(self) -> list[float]:
        """Magnifications in the series, ascending."""
        return sorted(self.groups)

    @property
    def key(self) -> str:
        """Stable identifier, e.g. ``'cptac_benchmark/256px'``."""
        return f"{self.cohort}/{self.patch_size}px"

    def shared_slides(self) -> list[str]:
        """Slides present at every magnification, for all common encoders.

        Returns
        -------
        list of str
            Sorted slide ids.
        """
        sets = [
            set(self.groups[m].slides(self.encoders)) for m in self.magnifications
        ]
        return sorted(set.intersection(*sets)) if sets else []

    def sample(
        self,
        n_patches: int = 20_000,
        max_slides: int | None = 200,
        seed: int = 0,
        slides: Sequence[str] | None = None,
        dtype=np.float32,
        verbose: bool = False,
    ) -> dict[float, dict[str, np.ndarray]]:
        """Draw a patch sample at each magnification from the same slides.

        Parameters
        ----------
        n_patches : int, default 20000
            Patches per magnification.
        max_slides : int or None, default 200
            Slides to read per magnification.
        seed : int, default 0
            Sampling seed, shared across magnifications so the same slides are
            selected at each.
        slides : sequence of str, optional
            Restrict to these slides. Defaults to :meth:`shared_slides`.
        dtype : numpy dtype, default ``np.float32``
            Output dtype.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        dict of float to dict
            ``{magnification: {encoder: (n_patches, dim)}}``, row-paired within
            each magnification.
        """
        pool = list(slides) if slides else self.shared_slides()
        if not pool:
            raise ValueError(f"no shared slides across {self.key}")

        # Choose the slide subset once so every magnification sees the same
        # tissue; only the grid changes.
        rng = np.random.default_rng(seed)
        if max_slides is not None and len(pool) > max_slides:
            idx = sorted(rng.choice(len(pool), size=max_slides, replace=False).tolist())
            pool = [pool[i] for i in idx]

        out = {}
        for mag in self.magnifications:
            if verbose:
                print(f"  {mag:g}x ...")
            out[mag] = self.groups[mag].sample_patches(
                n_patches=n_patches,
                encoders=self.encoders,
                slides=pool,
                max_slides=None,
                seed=seed,
                dtype=dtype,
                verbose=verbose,
            )
        return out

    def __repr__(self) -> str:
        mags = ", ".join(f"{m:g}x" for m in self.magnifications)
        return (
            f"MagnificationSeries({self.key}, [{mags}], "
            f"{len(self.encoders)} common encoders: {self.encoders})"
        )


class FeatureStore:
    """Discovers what features exist on disk and hands back pairable groups.

    Parameters
    ----------
    config : str or pathlib.Path, optional
        Encoder registry YAML. Defaults to ``configs/encoders.yaml``.
    feature_root : str or pathlib.Path, optional
        Override the root in the config — useful when the store is mounted
        elsewhere (e.g. after syncing to ``raj``).
    known_only : bool, default True
        Ignore ``features_*`` directories whose encoder is not in the registry.
        Set False to discover encoders the config does not yet describe.

    Examples
    --------
    >>> store = FeatureStore()                                   # doctest: +SKIP
    >>> store.summary()                                          # doctest: +SKIP
    >>> group = store.best_group()                               # doctest: +SKIP
    >>> views = group.sample_patches(n_patches=20000)            # doctest: +SKIP
    """

    def __init__(
        self,
        config: str | Path | None = None,
        feature_root: str | Path | None = None,
        known_only: bool = True,
    ):
        root, encoders = load_encoder_config(config)
        self.feature_root = Path(feature_root) if feature_root else root
        self.encoder_info = encoders
        self.known_only = known_only

        if not self.feature_root.exists():
            raise FileNotFoundError(
                f"feature_root {self.feature_root} does not exist; pass "
                "feature_root= or edit configs/encoders.yaml"
            )
        self._groups = self._discover()

    def _discover(self) -> dict[str, FeatureGroup]:
        """Walk the feature root and build one group per coordinate grid."""
        groups: dict[str, FeatureGroup] = {}
        for cohort_dir in sorted(p for p in self.feature_root.iterdir() if p.is_dir()):
            for grid_dir in sorted(p for p in cohort_dir.iterdir() if p.is_dir()):
                m = _DIR_RE.match(grid_dir.name)
                if not m:
                    continue

                encoders = {}
                for fd in sorted(grid_dir.glob("features_*")):
                    if not fd.is_dir():
                        continue
                    name = fd.name[len("features_") :]
                    if self.known_only and name not in self.encoder_info:
                        continue
                    if next(fd.glob("*.h5"), None) is None:
                        continue
                    encoders[name] = fd

                if not encoders:
                    continue
                group = FeatureGroup(
                    cohort=cohort_dir.name,
                    magnification=float(m.group("mag")),
                    patch_size=int(m.group("patch")),
                    path=grid_dir,
                    encoders=encoders,
                )
                groups[group.key] = group
        return groups

    @property
    def groups(self) -> dict[str, FeatureGroup]:
        """All discovered groups, keyed by ``cohort/MAGx_PATCHpx``."""
        return self._groups

    def group(self, key: str) -> FeatureGroup:
        """Fetch one group by key.

        Parameters
        ----------
        key : str
            e.g. ``'cptac_benchmark/10x_256px'``.

        Returns
        -------
        FeatureGroup
        """
        try:
            return self._groups[key]
        except KeyError:
            raise KeyError(
                f"unknown group {key!r}; available: {sorted(self._groups)}"
            ) from None

    def best_group(
        self, cohort: str | None = None, family: str | None = None
    ) -> FeatureGroup:
        """Return the group with the most encoders.

        Parameters
        ----------
        cohort : str, optional
            Restrict to one cohort.
        family : str, optional
            Count only encoders of this family when ranking.

        Returns
        -------
        FeatureGroup
            The group offering the widest model comparison.
        """
        candidates = [
            g
            for g in self._groups.values()
            if cohort is None or g.cohort == cohort
        ]
        if not candidates:
            raise ValueError(f"no groups for cohort {cohort!r}")

        def score(g: FeatureGroup) -> tuple[int, int]:
            if family is None:
                n = g.n_encoders
            else:
                n = sum(
                    1
                    for e in g.encoders
                    if self.encoder_info.get(e)
                    and self.encoder_info[e].family == family
                )
            return (n, g.patch_size)

        return max(candidates, key=score)

    def magnification_series(
        self,
        cohort: str | None = None,
        patch_size: int | None = None,
        min_magnifications: int = 2,
        min_encoders: int = 2,
    ) -> list[MagnificationSeries]:
        """Find experiments that can be repeated across magnifications.

        Groups sharing a cohort and patch size form a series; the encoder set
        is intersected across magnifications so the comparison isolates
        magnification alone.

        Parameters
        ----------
        cohort : str, optional
            Restrict to one cohort.
        patch_size : int, optional
            Restrict to one patch size.
        min_magnifications : int, default 2
            Discard series with fewer magnifications than this.
        min_encoders : int, default 2
            Discard series whose common encoder set is smaller than this.

        Returns
        -------
        list of MagnificationSeries
            Sorted by encoder count then magnification count, both descending,
            so the most informative series comes first.
        """
        buckets: dict[tuple[str, int], dict[float, FeatureGroup]] = {}
        for g in self._groups.values():
            if cohort is not None and g.cohort != cohort:
                continue
            if patch_size is not None and g.patch_size != patch_size:
                continue
            buckets.setdefault((g.cohort, g.patch_size), {})[g.magnification] = g

        series = []
        for (coh, ps), groups in buckets.items():
            if len(groups) < min_magnifications:
                continue
            common = sorted(
                set.intersection(*[set(g.encoders) for g in groups.values()])
            )
            if len(common) < min_encoders:
                continue
            series.append(
                MagnificationSeries(
                    cohort=coh, patch_size=ps, groups=groups, encoders=common
                )
            )

        return sorted(
            series, key=lambda s: (len(s.encoders), len(s.groups)), reverse=True
        )

    def best_series(self, cohort: str | None = None) -> MagnificationSeries:
        """Return the magnification series covering the most encoders.

        Parameters
        ----------
        cohort : str, optional
            Restrict to one cohort.

        Returns
        -------
        MagnificationSeries
        """
        series = self.magnification_series(cohort=cohort)
        if not series:
            raise ValueError("no magnification series available")
        return series[0]

    def slide_encoders(self, cohort: str) -> "SlideEncoderSet":
        """Collect the slide-level encoders for a cohort, across all grids.

        Slide encoders pair by slide id rather than by patch index, so unlike
        :meth:`group` this deliberately gathers encoders from *every*
        ``(magnification, patch_size)`` directory into one comparable set.

        Parameters
        ----------
        cohort : str
            Store cohort, e.g. ``'master_benchmark'``.

        Returns
        -------
        SlideEncoderSet

        Raises
        ------
        ValueError
            If the cohort has no slide-level features.
        """
        root = self.feature_root / cohort
        if not root.exists():
            raise ValueError(f"cohort {cohort!r} not found under {self.feature_root}")

        encoders, grids = {}, {}
        for grid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if not _DIR_RE.match(grid_dir.name):
                continue
            for fd in sorted(grid_dir.glob("slide_features_*")):
                if not fd.is_dir() or next(fd.glob("*.h5"), None) is None:
                    continue
                name = fd.name[len("slide_features_") :]
                encoders[name] = fd
                grids[name] = grid_dir.name.replace("_0px_overlap", "")

        if not encoders:
            raise ValueError(f"no slide-level features under {root}")
        return SlideEncoderSet(cohort=cohort, encoders=encoders, grids=grids)

    def inventory(self, count_slides: bool = True):
        """Tabulate every group and its encoders.

        Parameters
        ----------
        count_slides : bool, default True
            Also count the slides shared by all encoders in each group. This
            touches the filesystem for every encoder directory, so it is the
            slow part; set False for a quick look.

        Returns
        -------
        pandas.DataFrame
            One row per group, with encoder list, encoder count, feature
            dimensions and (optionally) shared slide count.
        """
        import pandas as pd

        rows = []
        for key, g in sorted(self._groups.items()):
            encs = sorted(g.encoders)
            rows.append(
                {
                    "group": key,
                    "cohort": g.cohort,
                    "magnification": g.magnification,
                    "patch_size": g.patch_size,
                    "n_encoders": len(encs),
                    "encoders": ",".join(encs),
                    "dims": ",".join(
                        str(self.encoder_info[e].dim)
                        if e in self.encoder_info
                        else "?"
                        for e in encs
                    ),
                    "n_shared_slides": len(g.slides()) if count_slides else None,
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["n_encoders", "cohort"], ascending=[False, True]
        )

    def summary(self, count_slides: bool = True) -> str:
        """Human-readable inventory, sorted by how many models can be compared.

        Parameters
        ----------
        count_slides : bool, default True
            Include shared slide counts.

        Returns
        -------
        str
            Formatted table.
        """
        df = self.inventory(count_slides=count_slides)
        return df.to_string(index=False)

    def describe_encoders(self):
        """Registry metadata as a table, for figure labels and grouping.

        Returns
        -------
        pandas.DataFrame
            One row per registered encoder.
        """
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "encoder": e.name,
                    "display_name": e.display_name,
                    "dim": e.dim,
                    "family": e.family,
                    "objective": e.objective,
                    "architecture": e.architecture,
                    "hf_id": e.hf_id,
                }
                for e in self.encoder_info.values()
            ]
        ).set_index("encoder")

    def display_names(self, encoders: Sequence[str]) -> list[str]:
        """Map registry keys to display names for plotting.

        Parameters
        ----------
        encoders : sequence of str
            Registry keys.

        Returns
        -------
        list of str
            Display names, falling back to the key when unregistered.
        """
        return [
            self.encoder_info[e].display_name if e in self.encoder_info else e
            for e in encoders
        ]

    def families(self, encoders: Sequence[str]) -> dict[str, str]:
        """Map encoders to their pretraining family.

        Parameters
        ----------
        encoders : sequence of str
            Registry keys.

        Returns
        -------
        dict of str to str
            ``{encoder: family}``.
        """
        return {
            e: (self.encoder_info[e].family if e in self.encoder_info else "unknown")
            for e in encoders
        }

    def __repr__(self) -> str:
        return (
            f"FeatureStore(root={self.feature_root}, "
            f"{len(self._groups)} groups, {len(self.encoder_info)} registered encoders)"
        )


@dataclass
class SlideEncoderSet:
    """Slide-level encoders for one cohort, paired by slide id.

    Slide encoders consume a whole slide and emit **one vector per slide**, so
    the unit of comparison is the slide rather than the patch. That dissolves
    the constraint that governs the patch-level analysis: a patch encoder can
    only be compared against others sharing its coordinate grid, but slide
    encoders built on *different* grids still describe the same slides and are
    therefore directly comparable.

    In this store that means all six slide encoders can be compared at once —
    CHIEF and Madeleine from 10x/256px, PRISM from 20x/224px, GigaPath-slide
    from 20x/256px, TITAN and Feather from 20x/512px — across every slide in the
    cohort, where the patch encoders top out at six on one grid.

    The trade-off is sample size: n is the number of *slides* (2169 or 2296),
    not patches, against dimensions of 512-1280. That is enough for the
    similarity metrics but close to the floor for the CCA family, which
    saturates as n approaches d.

    Attributes
    ----------
    cohort : str
        Store cohort.
    encoders : dict of str to pathlib.Path
        ``{encoder: directory}``.
    grids : dict of str to str
        ``{encoder: the patch grid it was built on}``, for reporting — it does
        not restrict which encoders can be compared.
    """

    cohort: str
    encoders: dict[str, Path]
    grids: dict[str, str]

    @property
    def n_encoders(self) -> int:
        """Number of slide encoders available."""
        return len(self.encoders)

    def slides(self, encoders: Sequence[str] | None = None) -> list[str]:
        """Slide ids present for every requested encoder.

        Parameters
        ----------
        encoders : sequence of str, optional
            Restrict to these encoders.

        Returns
        -------
        list of str
            Sorted slide ids.
        """
        names = list(encoders) if encoders else list(self.encoders)
        missing = [n for n in names if n not in self.encoders]
        if missing:
            raise KeyError(f"{missing} not available; present: {sorted(self.encoders)}")
        sets = [{p.stem for p in self.encoders[n].glob("*.h5")} for n in names]
        if not sets:
            return []
        # Same withholding as FeatureGroup.slides; this store resolves slides
        # independently, so the filter has to be applied here too.
        return sorted(set.intersection(*sets) - excluded_slides())

    def load(
        self,
        encoders: Sequence[str] | None = None,
        slides: Sequence[str] | None = None,
        max_slides: int | None = None,
        seed: int = 0,
        dtype=np.float32,
        verbose: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Load one embedding per slide for each encoder.

        Parameters
        ----------
        encoders : sequence of str, optional
            Restrict to these encoders. Defaults to all.
        slides : sequence of str, optional
            Restrict to these slides. Defaults to those shared by all encoders.
        max_slides : int or None, default None
            Subsample this many slides.
        seed : int, default 0
            Subsampling seed.
        dtype : numpy dtype, default ``np.float32``
            Output dtype.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        tuple
            ``({encoder: (n_slides, dim)}, slide_ids)`` — row-paired by slide,
            ready for any of the patch-level analyses.
        """
        import h5py

        names = list(encoders) if encoders else sorted(self.encoders)
        pool = list(slides) if slides else self.slides(names)
        if not pool:
            raise ValueError(f"no shared slides across {names}")

        if max_slides is not None and len(pool) > max_slides:
            rng = np.random.default_rng(seed)
            idx = sorted(rng.choice(len(pool), size=max_slides, replace=False).tolist())
            pool = [pool[i] for i in idx]

        out: dict[str, list] = {n: [] for n in names}
        kept: list[str] = []
        for slide_id in pool:
            vectors = {}
            for name in names:
                path = self.encoders[name] / f"{slide_id}.h5"
                try:
                    with h5py.File(path, "r") as h:
                        vectors[name] = np.asarray(h["features"]).reshape(-1)
                except (OSError, KeyError):
                    vectors = {}
                    break
            if not vectors:
                if verbose:
                    print(f"  skipping {slide_id}")
                continue
            for name, vec in vectors.items():
                out[name].append(vec)
            kept.append(slide_id)

        if not kept:
            raise ValueError("no slides could be loaded")
        if verbose:
            print(f"  loaded {len(kept)} slides for {len(names)} encoders")
        return {n: np.vstack(v).astype(dtype, copy=False) for n, v in out.items()}, kept

    def __repr__(self) -> str:
        return (
            f"SlideEncoderSet({self.cohort}, {self.n_encoders} encoders: "
            f"{sorted(self.encoders)})"
        )
