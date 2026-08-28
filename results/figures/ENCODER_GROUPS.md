# Patch encoder groups

Twelve patch encoders are registered in `configs/encoders.yaml`. They do **not**
form one comparison — they form four, and which encoders land together is
imposed by the data rather than chosen.

Similarity needs row-paired patches: row *i* of one feature matrix and row *i*
of another must be the same patch of the same slide. Trident writes one
coordinate grid per `(magnification, patch_size)`, and every encoder run against
that grid inherits its row order (`utils/features.py:10-23`). Encoders extracted
at different patch sizes share no row index, so they cannot be compared at all —
not with more samples, not with resampling. **Patch size therefore partitions
the registry into four disjoint comparison families.**

## The four families

| family | encoders | n | dims | objectives | slides |
|---|---|---|---|---|---|
| **256px** | CONCH, CTransPath, Prov-GigaPath, KEEP, ResNet50 (ImageNet), UNI2-h | 6 | 512–1536 | 2 VL, 3 SSL, 1 supervised | 2162 / 2169 |
| **224px** | GPFM, H-optimus-0, Virchow, Virchow2 | 4 | 1024–2560 | 4 SSL | 2162 / 2169 |
| **512px** | CONCH, CONCH v1.5 | 2 | 512, 768 | 2 VL | 2162 / 2169 |
| **384px** | MUSK | 1 | 1024 | 1 VL | — |

Slides are CPTAC / TCGA after the withholding in `configs/excluded_slides.txt`.

**384px yields no similarity result at all.** MUSK is alone on that grid, and a
one-encoder matrix has no off-diagonal entry. This is why MUSK appears in no
similarity, alignment, retrieval or transfer figure despite being registered and
fully extracted — it is a data-layout consequence, not a decision about the
model. MUSK is evaluated on its own in `results/musk_baseline/`.

**The 256px family is the flagship** and the only one spanning more than one
supervision paradigm. Every cross-family claim in the study — vision-language
against vision-SSL, pathology against the ImageNet control — can only be made
here. The 224px family is four vision-SSL models and carries no control at all.

## Where each encoder lives

`both` = the grid exists for CPTAC and TCGA; `TCGA` = that cohort only.

| encoder | 5×/224 | 10×/224 | 20×/224 | 5×/256 | 10×/256 | 20×/256 | 5×/384 | 10×/384 | 20×/384 | 5×/512 | 10×/512 | 20×/512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPFM | both | both | both | · | · | · | · | · | · | · | · | · |
| H-optimus-0 | both | both | both | · | · | · | · | · | · | · | · | · |
| Virchow2 | both | both | both | · | · | · | · | · | · | · | · | · |
| Virchow | · | · | both | · | · | · | · | · | · | · | · | · |
| CTransPath | · | · | · | both | both | both | · | · | · | · | · | · |
| Prov-GigaPath | · | · | · | both | both | both | · | · | · | · | · | · |
| KEEP | · | · | · | both | both | both | · | · | · | · | · | · |
| ResNet50 (ImageNet) | · | · | · | both | both | both | · | · | · | · | · | · |
| UNI2-h | · | · | · | both | both | both | · | · | · | · | · | · |
| CONCH | · | · | · | · | both | · | · | · | · | both | both | both |
| CONCH v1.5 | · | · | · | · | · | **TCGA** | · | · | · | both | both | both |
| MUSK | · | · | · | · | · | · | both | both | both | · | · | · |

### Three asymmetries this table makes visible

**Virchow exists only at 20×.** The 224px family is four encoders at 20× and
three at 5× and 10×. Not a sampling choice — a missing extraction.

**CONCH is at 10×/256px but not 5× or 20×.** So the 256px family is six encoders
at 10× and five at 5× and 20×, and the magnification sweep at 256px runs on the
five that span all three (`fig4`, `fig4b`). Including CONCH would have collapsed
the sweep to a single magnification.

**CONCH v1.5 at 20×/256px is TCGA-only.** It is the one place where the two
cohorts do not have matching coverage. The consequence is that `tcga_20x_256` is
a six-encoder group whose set is *not* the flagship six — CONCH v1.5 stands in
for CONCH — while `cptac_20x_256` has five. Two rows of a cross-grid table both
reading "256px, 6 encoders" are therefore not the same six. That group is also
1126 slides of pure BRCA, so it differs in tissue as well.

## Encoder properties

| encoder | dim | architecture | objective | pretraining | params |
|---|---|---|---|---|---|
| CONCH | 512 | ViT-B/16 + text | CoCa (contrastive + captioning) | 1.17M image–caption pairs | — |
| CONCH v1.5 | 768 | ViT-L/16 @ 448px | vision–language (TITAN patch encoder) | TITAN corpus | — |
| CTransPath | 768 | CNN + Swin-T hybrid | SRCL contrastive | ~15M patches (TCGA + PAIP) | — |
| Prov-GigaPath | 1536 | ViT-g/14 | DINOv2 | 1.3B tiles / 171k WSI | — |
| UNI2-h | 1536 | ViT-H/14 (registers) | DINOv2 | ~200M patches / ~350k WSI | — |
| KEEP | 768 | ViT-L/16 | knowledge-enhanced VL | image–text + disease knowledge graph | — |
| ResNet50 (ImageNet) | 1024 | ResNet50 (to layer3) | supervised classification | ImageNet-1k, **no pathology** | — |
| GPFM | 1024 | ViT-L/14 | multi-teacher distillation | ~190k WSI | 307M |
| H-optimus-0 | 1536 | ViT-g/14 | DINOv2 | ~500k WSI | 1.1B |
| Virchow | 2560 | ViT-H/14 | DINOv2 | 1.5M WSI | 632M |
| Virchow2 | 2560 | ViT-H/14 | DINOv2 + pathology augmentations | 3.1M WSI | 632M |
| MUSK | 1024 | BEiT-3 multimodal @ 384px | masked modelling + VL alignment | 50M images + 1B text tokens | — |

## Not registered, and why

`plip`, `quiltnet-b16`, `clip_rn50`, `uni_v1`, `virchow2-cls` are present in the
feature store but excluded from the registry, so `FeatureGroup` never sees them.
The first three sit at 224px and would restore family diversity to that
family — PLIP and QuiltNet are vision-language, CLIP-RN50 a control — but **they
exist for TCGA only**. Registering them would give the two cohorts different
encoder sets, which is the defect that got the `care` and `prism2` slide
encoders excluded. See `AGENT.md` under *Out of scope*.
