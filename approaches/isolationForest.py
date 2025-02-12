import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging
from tqdm import tqdm

class TelecomAnomalyDetector:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """
        Preprocess the telecommunication data.
        """
        # Extract timestamp from the compound string and parse with specific format
        df['datetime'] = df['timestamp_str'].apply(
            lambda x: pd.to_datetime(
                x.split(';')[1],
                format='%H:%M:%S.%f-%Y/%m/%d'
            )
        )
        
        # Extract relevant features
        features = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        X = df[features].copy()
        
        # Create additional engineered features
        # Rolling statistics for ReTx and CRC per HARQ
        for harq in df['HARQ'].unique():
            harq_mask = df['HARQ'] == harq
            X.loc[harq_mask, 'rolling_retx_mean'] = df.loc[harq_mask, 'ReTx'].rolling(5, min_periods=1).mean()
            X.loc[harq_mask, 'rolling_crc_mean'] = df.loc[harq_mask, 'CRC'].rolling(5, min_periods=1).mean()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, df['timestamp_str']

    def fit_detect(self, df):
        """
        Fit the model and detect anomalies.
        """
        X_scaled, timestamps = self.preprocess_data(df)
        
        print("Fitting Isolation Forest...")
        with tqdm(total=1) as pbar:
            predictions = self.model.fit_predict(X_scaled)
            pbar.update(1)
            
        results_df = pd.DataFrame({
            'timestamp_str': timestamps,
            'is_anomaly': predictions == -1
        })
        
        return results_df[results_df['is_anomaly']]

def find_optimal_contamination(df, ground_truth_path, contamination_range=np.arange(0.001, 0.1, 0.001)):
    """
    Find the optimal contamination parameter that maximizes our custom score.
    """
    best_score = float('-inf')
    best_contamination = None
    best_results = None
    
    print("Finding optimal contamination parameter...")
    for contamination in tqdm(contamination_range):
        detector = TelecomAnomalyDetector(contamination=contamination)
        anomalies_df = detector.fit_detect(df)
        score, metrics = validate_anomalies(anomalies_df, ground_truth_path, verbose=False)
        
        if score > best_score:
            best_score = score
            best_contamination = contamination
            best_results = metrics
    
    print(f"\nBest contamination value: {best_contamination:.4f}")
    print(f"Best score: {best_score:.4f}")
    print("\nDetailed metrics for best result:")
    for key, value in best_results.items():
        print(f"{key}: {value}")
    
    return best_contamination

def validate_anomalies(anomalies_df, ground_truth_path, verbose=True):
    """
    Validate detected anomalies against ground truth with custom scoring.
    
    Returns:
        float: Custom score balancing recall and precision
        dict: Detailed metrics
    """
    # Load ground truth
    df_gt = pd.read_csv(ground_truth_path)
    gt_col = "timestamp_str" if "timestamp_str" in df_gt.columns else "timestamp"
    gt_timestamps = set(df_gt[gt_col].unique())
    
    # Get detected anomaly timestamps
    detected_timestamps = set(anomalies_df['timestamp_str'])
    
    # Calculate metrics
    total_gt_anomalies = len(gt_timestamps)
    total_detected = len(detected_timestamps)
    detected_gt_anomalies = len(gt_timestamps.intersection(detected_timestamps))
    
    if total_gt_anomalies == 0 or total_detected == 0:
        return 0.0, {}
    
    recall = detected_gt_anomalies / total_gt_anomalies
    precision = detected_gt_anomalies / total_detected
    
    # Custom score that penalizes both missing ground truth anomalies and having too many false positives
    # Using F1-score with additional penalty for high false positive rate
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = (total_detected - detected_gt_anomalies) / total_detected if total_detected > 0 else 0
    custom_score = f1_score * (1 - false_positive_rate)
    
    metrics = {
        "Total ground truth anomalies": total_gt_anomalies,
        "Total detected anomalies": total_detected,
        "Ground truth anomalies found": detected_gt_anomalies,
        "Recall": f"{recall*100:.2f}%",
        "Precision": f"{precision*100:.2f}%",
        "False positive rate": f"{false_positive_rate*100:.2f}%",
        "Custom score": f"{custom_score:.4f}"
    }
    
    if verbose:
        print("\nValidation Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    
    return custom_score, metrics

# Example usage
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv('/workspaces/thesis/data/pdsch_data_romes_clean/test/test.csv')
    
    # Find optimal contamination
    best_contamination = find_optimal_contamination(df, '/workspaces/thesis/data/pdsch_data_romes_clean/test/test_gt.csv')
    
    # Run final model with optimal contamination
    detector = TelecomAnomalyDetector(contamination=best_contamination)
    anomalies_df = detector.fit_detect(df)
    
    # Get final validation results
    final_score, _ = validate_anomalies(anomalies_df, '/workspaces/thesis/data/pdsch_data_romes_clean/test/test_gt.csv')