# False Positive Cluster Analysis Summary

## PCA Analysis
- Number of clusters identified: 2
- Explained variance by first 3 PCs: 58.01%
- PC1 explains 28.31% of variance
- PC2 explains 15.05% of variance
- PC3 explains 14.65% of variance

## Cluster Sizes
- Cluster 0: 1398 samples (53.96%)
- Cluster 1: 1193 samples (46.04%)

## Feature Contributions to Principal Components

### PC1 (28.31% variance)
Top contributing features:
- 1. ReTx: 74.97%
- 2. MCS: 68.87%
- 3. CRC: 39.85%

### PC2 (15.05% variance)
Top contributing features:
- 1. HARQ: 41.53%
- 2. SFN: 29.86%
- 3. NDI: 17.19%

### PC3 (14.65% variance)
Top contributing features:
- 1. NDI: 50.81%
- 2. Slot: 46.91%
- 3. CRC: 3.09%

## Discriminative Features Between Clusters
Features that best separate the clusters (ordered by effect size):
- HARQ: effect size = 0.09 *
  - Mean in Cluster 0: 10.82
  - Mean in Cluster 1: 10.46
- ReTx: effect size = 0.09 *
  - Mean in Cluster 0: 0.51
  - Mean in Cluster 1: 0.57
- MCS: effect size = 0.08 *
  - Mean in Cluster 0: 17.89
  - Mean in Cluster 1: 18.93
- CRC: effect size = 0.04 ns
  - Mean in Cluster 0: 0.65
  - Mean in Cluster 1: 0.67
- Slot: effect size = 0.02 ns
  - Mean in Cluster 0: 8.25
  - Mean in Cluster 1: 8.16

## Cluster Interpretation
Based on the analysis, the following interpretations can be made:

### Cluster 0
Distinctive characteristics:
- HARQ tends to be higher than in other cluster(s) (10.82 vs 10.46)
- ReTx tends to be lower than in other cluster(s) (0.51 vs 0.57)
- MCS tends to be lower than in other cluster(s) (17.89 vs 18.93)

### Cluster 1
Distinctive characteristics:
- HARQ tends to be lower than in other cluster(s) (10.46 vs 10.82)
- ReTx tends to be higher than in other cluster(s) (0.57 vs 0.51)
- MCS tends to be higher than in other cluster(s) (18.93 vs 17.89)

## Conclusion
The analysis of false positive clusters reveals distinct patterns that the model identified as anomalous despite being labeled as normal in the dataset. These patterns likely represent edge cases that exhibit characteristics similar to known anomalies but weren't explicitly labeled as such in the original data.

This suggests that the model may be identifying genuine anomalies that were missed during the labeling process, rather than simply making classification errors. Further investigation with domain experts could help determine if these false positives should be relabeled as true anomalies in future iterations of the model.