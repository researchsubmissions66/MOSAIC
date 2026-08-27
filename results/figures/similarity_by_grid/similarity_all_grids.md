# Every similarity result in one table

Mean **off-diagonal** similarity per group.

- **encoders** — how many encoders share that group's coordinate grid;
  the similarity matrix is `encoders x encoders`.
- **pairs** — distinct encoder pairs, `encoders x (encoders - 1) / 2`.
  This is what each mean is actually averaged over, and it is why a
  2-encoder row is a single number rather than a distribution.
- The unit diagonal is excluded from every mean.

Read down a column, not across a row: the seven metrics sit at different
levels, and the groups hold different encoder sets, so a 2-encoder 512px
row is not on the same footing as a 6-encoder 256px one.

| grid | cohort | mag | encoders | pairs | Linear CKA | Kernel CKA | SVCCA | PWCCA | Procrustes | Cosine RSA | Distance Correlation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 224px | CPTAC | 5x | 3 | 3 | 0.543 | 0.581 | 0.596 | 0.836 | 0.711 | 0.520 | 0.755 |
| 224px | CPTAC | 10x | 3 | 3 | 0.532 | 0.562 | 0.620 | 0.843 | 0.722 | 0.501 | 0.744 |
| 224px | CPTAC | 20x | 4 | 6 | 0.486 | 0.516 | 0.567 | 0.779 | 0.683 | 0.477 | 0.714 |
| 224px | TCGA | 5x | 3 | 3 | 0.517 | 0.554 | 0.602 | 0.836 | 0.705 | 0.490 | 0.732 |
| 224px | TCGA | 10x | 3 | 3 | 0.532 | 0.571 | 0.619 | 0.838 | 0.707 | 0.478 | 0.745 |
| 224px | TCGA | 20x | 4 | 6 | 0.459 | 0.488 | 0.568 | 0.767 | 0.650 | 0.396 | 0.693 |
| 256px | CPTAC | 10x | 6 | 15 | 0.600 | 0.627 | 0.567 | 0.637 | 0.609 | 0.567 | 0.789 |
| 256px | CPTAC | 20x | 5 | 10 | 0.543 | 0.571 | 0.510 | 0.608 | 0.578 | 0.503 | 0.749 |
| 256px | TCGA | 10x | 6 | 15 | 0.646 | 0.669 | 0.618 | 0.694 | 0.619 | 0.604 | 0.819 |
| 256px | TCGA | 20x | 6 | 15 | 0.649 | 0.675 | 0.610 | 0.662 | 0.590 | 0.645 | 0.821 |
| 512px | CPTAC | 5x | 2 | 1 | 0.842 | 0.853 | 0.578 | 0.683 | 0.775 | 0.830 | 0.924 |
| 512px | CPTAC | 10x | 2 | 1 | 0.846 | 0.855 | 0.582 | 0.691 | 0.781 | 0.831 | 0.925 |
| 512px | CPTAC | 20x | 2 | 1 | 0.827 | 0.837 | 0.570 | 0.686 | 0.767 | 0.808 | 0.915 |
| 512px | TCGA | 5x | 2 | 1 | 0.890 | 0.899 | 0.572 | 0.691 | 0.796 | 0.898 | 0.948 |
| 512px | TCGA | 10x | 2 | 1 | 0.850 | 0.863 | 0.613 | 0.708 | 0.800 | 0.826 | 0.931 |
| 512px | TCGA | 20x | 2 | 1 | 0.860 | 0.870 | 0.579 | 0.696 | 0.786 | 0.849 | 0.934 |

Source: `similarity_all_grids.csv` in this folder.
