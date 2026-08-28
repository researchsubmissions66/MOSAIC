# Slide-encoder similarity

All seven metrics, one figure per cohort.

Slide encoders emit one vector per slide, so the patch-grid pairing
constraint does not apply and all of them are comparable at once --
unlike the patch encoders, which split across four grids.

**Sample-size caveat.** n here is the number of *slides*, not patches,
against dimensions of 512-1280. That is ample for CKA and RSA but close
to the floor for SVCCA and PWCCA, which saturate as n approaches d.

| figure | cohort | encoders | source |
|---|---|---|---|
| `tcga_slide_encoders.pdf` | TCGA | 6 | `results/slide_encoders/master_benchmark/matrices` |
