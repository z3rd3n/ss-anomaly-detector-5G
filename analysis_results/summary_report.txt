# Anomaly Detection Model Evaluation Summary

## Model Information
- Model: BiGRUAnomalyDetector
- Parameters: 1,543,549

## Dataset Statistics
- Total sequences: 7807
- Total timesteps: 687016
- Anomaly counts:
  - Normal: 679799 (98.95%)
  - Type 1: 6471 (0.94%)
  - Type 2: 303 (0.04%)
  - Type 3: 202 (0.03%)
  - Type 4: 241 (0.04%)

## Performance Metrics
- Accuracy: 0.9958
- Precision: 0.7278
- Recall: 0.9598
- F1 Score: 0.8278
- ROC AUC: 0.9986

## Class Detection Rates
- Normal: 677208/679799 = 0.9962
- Type 1: 6338/6471 = 0.9794
- Type 2: 184/303 = 0.6073
- Type 3: 164/202 = 0.8119
- Type 4: 241/241 = 1.0000

## False Positive Analysis
- False positives: 2591 (0.38% of all samples)
- False negatives: 290 (0.04% of all samples)
- True positives: 6927 (1.01% of all samples)
- True negatives: 677208 (98.57% of all samples)

### False Positive Clusters
- Number of clusters: 2
- Cluster 0: 1398 samples (53.96%)
  - Top features: ReTx (0.80), MCS (0.75), CRC (0.47)
  - Possible anomaly type: Unknown Anomaly Pattern
- Cluster 1: 1193 samples (46.04%)
  - Top features: ReTx (0.93), MCS (0.88), CRC (0.55)
  - Possible anomaly type: Unknown Anomaly Pattern

## Conclusion
The model achieved an overall anomaly detection rate of 95.98%.

The model performed best on Type 4 anomalies with a detection rate of 100.00%.
The model struggled most with Type 2 anomalies with a detection rate of 60.73%.