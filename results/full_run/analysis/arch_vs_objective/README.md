# Architecture or pretraining objective? (RQ2)

Does representation geometry track the **architectural family** or the
**pretraining objective**? For a similarity matrix `S` under metric `m`:

    Δ_obj  = E[ S_ij | o_i = o_j ] − E[ S_ij | o_i ≠ o_j ]
    Δ_arch = E[ S_ij | a_i = a_j ] − E[ S_ij | a_i ≠ a_j ]

Annotations come from `configs/encoders.yaml` — `family` for the objective
(`vision_ssl`, `vision_language`, `supervised`), `architecture` for the
backbone. Produced by `scripts/architecture_vs_objective.py`; full table in
`arch_vs_objective.csv`.

Architecture is grouped **two ways**, because neither alone is honest:

- **backbone** — ViT / hybrid / CNN. Coarse enough to yield same-architecture
  pairs on every group, but on the six-encoder groups it is confounded (below).
- **exact** — the full architecture string. Almost every pair differs, so it
  yields few same-architecture pairs; the ones it yields are clean.

## Answer

**The pretraining objective, not the architecture.** Three lines of evidence,
strongest last.

### 1. Δ_obj is positive everywhere; Δ_arch is not

Over the six groups where both are defined, 7 metrics each:

| | n | all positive | min | max | mean |
|---|---|---|---|---|---|
| Δ_obj | 42 | **yes** | +0.026 | +0.192 | **+0.076** |
| Δ_arch (backbone) | 42 | no — 7 negative | −0.047 | +0.180 | +0.051 |

Encoders sharing a supervision paradigm are more similar than encoders that do
not, in **every group and every metric measured**. Sharing a backbone family is
a weaker signal and reverses seven times.

### 2. Where the two separate, objective wins

Linear CKA, per group:

| group | Δ_obj | Δ_arch (backbone) |
|---|---|---|
| CPTAC 10×/256px (flagship) | +0.138 | +0.134 |
| CPTAC 20×/256px | +0.192 | +0.180 |
| CPTAC 5×/256px | +0.098 | +0.092 |
| TCGA 10×/256px | +0.054 | +0.023 |
| TCGA 5×/256px | +0.032 | +0.002 |
| TCGA 20×/256px | +0.080 | **−0.014** |

On CPTAC the two track each other closely — that is the confound, not agreement.
**The only supervised encoder (ResNet50) is also the only pure CNN**, so both
partitions largely reduce to "is the ImageNet control in this pair", and both
deltas mostly measure how far that control sits from everything else. On TCGA,
where the pooled control gap is smaller, the two separate and objective is the
survivor: 2.3× larger at 10×, 16× at 5×, and at 20× architecture goes negative
while objective stays positive.

### 3. The clean test: hold objective constant, vary architecture

The **224px groups** are a natural experiment. All four encoders there — GPFM,
H-optimus-0, Virchow, Virchow2 — are `vision_ssl`, so Δ_obj is undefined (6 same
/ 0 different pairs) and *any* variation must come from something else. Exactly
one pair shares an architecture: **Virchow and Virchow2, both ViT-H/14, both
2560-d, same lab, differing only in pretraining scale (1.5M → 3.1M WSI).**

Δ_arch (exact), that pair against the other five:

| metric | CPTAC 20×/224px | TCGA 20×/224px |
|---|---|---|
| Linear CKA | **−0.216** | **−0.197** |
| Kernel CKA | −0.188 | −0.182 |
| Cosine RSA | −0.185 | −0.150 |
| Distance Correlation | −0.130 | −0.133 |
| PWCCA | −0.055 | −0.050 |
| Procrustes | −0.022 | +0.024 |
| SVCCA | −0.017 | −0.005 |

**Negative, and large.** The two encoders that share an architecture exactly are
the *least* similar pair in the group — less similar than any cross-architecture
pair — and it replicates on both cohorts. Identical architecture, identical
dimension, same authors, and the representations are further apart than
ViT-L/14 is from ViT-g/14.

## What this does not show

**Δ_arch(exact) rests on one pair.** n_same = 1. It replicates across two
cohorts, but that is the same pair measured on two datasets, not two independent
pairs. It is a strong existence proof that shared architecture does not imply
shared geometry; it is not an estimate of an average effect.

**Δ_obj rests on 3-6 pairs per group** — 4 at the flagship: one vision-language
pair (CONCH↔KEEP) and three vision-SSL pairs. With one encoder per cell in some
categories, "objective" and "which specific model" are not fully separable.

**The flagship's agreement between the two deltas is a confound, not
corroboration.** Any partition that isolates ResNet50 scores well there. Cite
the TCGA 256px groups or the 224px test, not the flagship, for this claim.

**Effect sizes are metric-dependent in the expected way.** Δ_obj is largest on
the geometry block (linear CKA, kernel CKA, cosine RSA) and smallest on the CCA
block (SVCCA, PWCCA) — the same two-block split documented in
`results/figures/groups/tcga_20x_224px/README.md`. A paper reporting only PWCCA
would find a much weaker objective effect (+0.026 to +0.076) than one reporting
linear CKA (+0.032 to +0.192).

## Reproducing

```
python scripts/architecture_vs_objective.py --out results/full_run/analysis/arch_vs_objective
```

Standard library only — no pandas or PyYAML — so it runs anywhere the committed
matrices do.
