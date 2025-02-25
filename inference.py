import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze PDSCH Anomaly Detection Results')
    
    parser.add_argument('--predictions_file', type=str, required=True,
                        help='Path to CSV file with predictions')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for anomaly detection')
    parser.add_argument('--report_file', type=str, default='detailed_anomaly_report.csv',
                        help='Filename for the detailed report')
    parser.add_argument('--plot_figures', action='store_true',
                        help='Generate plots for the analysis')
    
    return parser.parse_args()

def load_predictions(predictions_file, threshold=0.5):
    """
    Load predictions from a CSV file.
    
    Args:
        predictions_file: Path to the CSV file with predictions
        threshold: Probability threshold for anomaly detection
    
    Returns:
        DataFrame with predictions and potentially reclassified anomalies
    """
    if not os.path.exists(predictions_file):
        print(f"Error: File {predictions_file} not found.")
        sys.exit(1)
        
    print(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Ensure required columns exist
    required_columns = ['anomaly_probability']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns in {predictions_file}")
        print(f"Required columns: {required_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Apply threshold if is_anomaly column doesn't exist
    if 'is_anomaly' not in df.columns:
        df['is_anomaly'] = (df['anomaly_probability'] >= threshold).astype(int)
        print(f"Applied threshold {threshold} to create is_anomaly column")
    
    # Check if original_label and anomaly_type columns exist
    has_ground_truth = ('original_label' in df.columns and 'anomaly_type' in df.columns)
    if not has_ground_truth:
        print("Warning: Missing ground truth information (original_label or anomaly_type)")
        print("Limited analysis will be performed")
        
    return df, has_ground_truth

def analyze_anomaly_types(df, output_dir, has_ground_truth=True):
    """
    Analyze detection performance by anomaly type.
    
    Args:
        df: DataFrame with predictions
        output_dir: Directory to save results
        has_ground_truth: Whether ground truth labels are available
    
    Returns:
        DataFrame with detection statistics by anomaly type
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not has_ground_truth:
        print("Cannot analyze anomaly types: missing ground truth information")
        return None
    
    # Count total samples by anomaly type
    type_stats = df.groupby('anomaly_type').agg({
        'original_label': 'count',
        'is_anomaly': 'sum'
    }).reset_index()
    
    # Rename columns for clarity
    type_stats.columns = ['anomaly_type', 'total_samples', 'detected_as_anomaly']
    
    # Calculate detection rate
    type_stats['detection_rate'] = type_stats.apply(
        lambda row: row['detected_as_anomaly'] / row['total_samples'] 
                   if row['anomaly_type'] != 'normal' else
                   (row['total_samples'] - row['detected_as_anomaly']) / row['total_samples'],
        axis=1
    )
    
    # Calculate adjusted metrics
    type_stats['correctly_detected'] = type_stats.apply(
        lambda row: row['detected_as_anomaly'] 
                   if row['anomaly_type'] != 'normal' else
                   (row['total_samples'] - row['detected_as_anomaly']),
        axis=1
    )
    
    type_stats['missed'] = type_stats['total_samples'] - type_stats['correctly_detected']
    
    # Sort by anomaly type with normal first
    type_stats['sort_order'] = type_stats['anomaly_type'].apply(
        lambda x: 0 if x == 'normal' else 1
    )
    type_stats = type_stats.sort_values(['sort_order', 'anomaly_type']).reset_index(drop=True)
    
    # Drop intermediate column
    type_stats = type_stats.drop('sort_order', axis=1)
    
    # Save to CSV
    type_stats.to_csv(os.path.join(output_dir, 'anomaly_type_stats.csv'), index=False)
    print(f"Saved anomaly type statistics to {os.path.join(output_dir, 'anomaly_type_stats.csv')}")
    
    return type_stats

def find_unexpected_anomalies(df, output_dir, has_ground_truth=True):
    """
    Find anomalies detected by the model but not in the original ground truth.
    
    Args:
        df: DataFrame with predictions
        output_dir: Directory to save results
        has_ground_truth: Whether ground truth labels are available
    
    Returns:
        DataFrame with unexpected anomalies
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not has_ground_truth:
        print("Cannot find unexpected anomalies: missing ground truth information")
        return None
    
    # Find unexpected anomalies
    unexpected = df[(df['is_anomaly'] == 1) & (df['original_label'] == 0)]
    
    if len(unexpected) == 0:
        print("No unexpected anomalies found")
        return None
    
    print(f"Found {len(unexpected)} unexpected anomalies")
    
    # Save to CSV
    unexpected.to_csv(os.path.join(output_dir, 'unexpected_anomalies.csv'), index=False)
    print(f"Saved unexpected anomalies to {os.path.join(output_dir, 'unexpected_anomalies.csv')}")
    
    # Try to find patterns in unexpected anomalies
    if 'harq_id' in unexpected.columns:
        harq_counts = unexpected['harq_id'].value_counts()
        
        print("\nHARQ ID distribution in unexpected anomalies:")
        for harq_id, count in harq_counts.head(5).items():
            print(f"  HARQ {harq_id}: {count} occurrences")
            
        harq_stats = pd.DataFrame({
            'harq_id': harq_counts.index,
            'count': harq_counts.values
        })
        harq_stats.to_csv(os.path.join(output_dir, 'unexpected_anomalies_harq.csv'), index=False)
    
    return unexpected

def analyze_missed_anomalies(df, output_dir, has_ground_truth=True):
    """
    Analyze anomalies that the model failed to detect.
    
    Args:
        df: DataFrame with predictions
        output_dir: Directory to save results
        has_ground_truth: Whether ground truth labels are available
    
    Returns:
        DataFrame with missed anomalies grouped by type
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not has_ground_truth:
        print("Cannot analyze missed anomalies: missing ground truth information")
        return None
    
    # Find missed anomalies (false negatives)
    missed = df[(df['is_anomaly'] == 0) & (df['original_label'] != 0)]
    
    if len(missed) == 0:
        print("No missed anomalies found")
        return None
    
    print(f"Found {len(missed)} missed anomalies")
    
    # Group by anomaly type
    missed_by_type = missed.groupby('anomaly_type').size().reset_index(name='count')
    
    # Save to CSV
    missed.to_csv(os.path.join(output_dir, 'missed_anomalies.csv'), index=False)
    missed_by_type.to_csv(os.path.join(output_dir, 'missed_anomalies_by_type.csv'), index=False)
    print(f"Saved missed anomalies to {os.path.join(output_dir, 'missed_anomalies.csv')}")
    
    # Report by type
    print("\nMissed anomalies by type:")
    for _, row in missed_by_type.iterrows():
        print(f"  {row['anomaly_type']}: {row['count']} missed")
    
    return missed_by_type

def create_detailed_report(df, type_stats, unexpected, missed_by_type, output_dir, report_file):
    """
    Create a detailed report of anomaly detection performance.
    
    Args:
        df: DataFrame with predictions
        type_stats: Statistics by anomaly type
        unexpected: Unexpected anomalies
        missed_by_type: Missed anomalies by type
        output_dir: Directory to save results
        report_file: Filename for the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate overall statistics
    total_samples = len(df)
    total_anomalies = (df['original_label'] != 0).sum() if 'original_label' in df.columns else None
    total_normal = (df['original_label'] == 0).sum() if 'original_label' in df.columns else None
    
    detected_anomalies = df['is_anomaly'].sum()
    
    if 'original_label' in df.columns:
        true_positives = ((df['is_anomaly'] == 1) & (df['original_label'] != 0)).sum()
        false_positives = ((df['is_anomaly'] == 1) & (df['original_label'] == 0)).sum()
        true_negatives = ((df['is_anomaly'] == 0) & (df['original_label'] == 0)).sum()
        false_negatives = ((df['is_anomaly'] == 0) & (df['original_label'] != 0)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total_samples
    else:
        true_positives = false_positives = true_negatives = false_negatives = None
        precision = recall = f1 = accuracy = None
    
    # Create report
    with open(os.path.join(output_dir, report_file), 'w') as f:
        f.write("# PDSCH Anomaly Detection Detailed Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overall Statistics\n\n")
        f.write(f"Total samples: {total_samples}\n")
        if total_anomalies is not None:
            f.write(f"Total anomalies in ground truth: {total_anomalies}\n")
            f.write(f"Total normal samples in ground truth: {total_normal}\n")
        f.write(f"Total anomalies detected by model: {detected_anomalies}\n\n")
        
        if precision is not None:
            f.write("## Classification Metrics\n\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            
            f.write("## Confusion Matrix\n\n")
            f.write(f"True Positives: {true_positives}\n")
            f.write(f"False Positives: {false_positives}\n")
            f.write(f"True Negatives: {true_negatives}\n")
            f.write(f"False Negatives: {false_negatives}\n\n")
        
        if type_stats is not None:
            f.write("## Detection Performance by Anomaly Type\n\n")
            for _, row in type_stats.iterrows():
                detection_rate = row['detection_rate'] * 100
                anomaly_type = row['anomaly_type']
                total = row['total_samples']
                detected = row['correctly_detected']
                
                if anomaly_type == 'normal':
                    f.write(f"Normal samples: {detected}/{total} correctly classified as normal ")
                    f.write(f"({detection_rate:.2f}%)\n")
                else:
                    f.write(f"{anomaly_type}: {detected}/{total} detected ")
                    f.write(f"({detection_rate:.2f}%)\n")
            f.write("\n")
                
        if unexpected is not None and len(unexpected) > 0:
            f.write("## Unexpected Anomalies\n\n")
            f.write(f"Found {len(unexpected)} samples detected as anomalies but labeled as normal in ground truth\n\n")
            
        if missed_by_type is not None and len(missed_by_type) > 0:
            f.write("## Missed Anomalies by Type\n\n")
            for _, row in missed_by_type.iterrows():
                f.write(f"{row['anomaly_type']}: {row['count']} missed\n")
            f.write("\n")
            
        f.write("## Conclusions\n\n")
        if type_stats is not None:
            best_detected = type_stats[type_stats['anomaly_type'] != 'normal'].sort_values('detection_rate', ascending=False)
            worst_detected = type_stats[type_stats['anomaly_type'] != 'normal'].sort_values('detection_rate')
            
            if len(best_detected) > 0:
                best_type = best_detected.iloc[0]['anomaly_type']
                best_rate = best_detected.iloc[0]['detection_rate'] * 100
                f.write(f"Best detected anomaly type: {best_type} ({best_rate:.2f}%)\n")
                
            if len(worst_detected) > 0:
                worst_type = worst_detected.iloc[0]['anomaly_type']
                worst_rate = worst_detected.iloc[0]['detection_rate'] * 100
                f.write(f"Worst detected anomaly type: {worst_type} ({worst_rate:.2f}%)\n\n")
                
        f.write("### Summary\n\n")
        if precision is not None:
            if f1 > 0.8:
                f.write("The model performs well for binary anomaly detection.\n")
            elif f1 > 0.6:
                f.write("The model performs adequately for binary anomaly detection but could be improved.\n")
            else:
                f.write("The model's performance for binary anomaly detection needs significant improvement.\n")
                
            if false_positives > 0.2 * detected_anomalies:
                f.write("There is a relatively high false positive rate, suggesting the model may be too sensitive.\n")
            if false_negatives > 0.2 * total_anomalies:
                f.write("There is a relatively high false negative rate, suggesting the model may be missing important anomalies.\n")
    
    print(f"Saved detailed report to {os.path.join(output_dir, report_file)}")

def plot_detection_by_type(type_stats, output_dir):
    """
    Plot detection rate by anomaly type.
    
    Args:
        type_stats: Statistics by anomaly type
        output_dir: Directory to save plots
    """
    if type_stats is None or len(type_stats) == 0:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # Plot detection rate by type
    sns.barplot(x='anomaly_type', y='detection_rate', data=type_stats)
    
    plt.title('Anomaly Detection Rate by Type')
    plt.xlabel('Anomaly Type')
    plt.ylabel('Detection Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'detection_rate_by_type.png'))
    plt.close()
    
    # Plot counts by type
    plt.figure(figsize=(12, 6))
    
    # Filter out normal samples for better visualization
    anomaly_stats = type_stats[type_stats['anomaly_type'] != 'normal'].copy()
    
    if len(anomaly_stats) > 0:
        # Add total count
        anomaly_stats['total'] = anomaly_stats['total_samples']
        anomaly_stats['missed'] = anomaly_stats['missed']
        anomaly_stats['detected'] = anomaly_stats['correctly_detected']
        
        # Plot stacked bar chart
        ax = anomaly_stats.plot(
            x='anomaly_type',
            y=['detected', 'missed'],
            kind='bar',
            stacked=True,
            figsize=(12, 6),
            color=['#4CAF50', '#FF5722']
        )
        
        plt.title('Anomaly Detection Count by Type')
        plt.xlabel('Anomaly Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(['Detected', 'Missed'])
        plt.tight_layout()
        
        # Add counts as text
        for i, (_, row) in enumerate(anomaly_stats.iterrows()):
            ax.text(
                i, 
                row['total_samples'] + 1, 
                str(row['total_samples']),
                ha='center'
            )
        
        plt.savefig(os.path.join(output_dir, 'detection_count_by_type.png'))
        plt.close()
    
    print(f"Saved detection plots to {output_dir}")

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    df, has_ground_truth = load_predictions(args.predictions_file, args.threshold)
    
    # Analyze anomaly types
    type_stats = analyze_anomaly_types(df, args.output_dir, has_ground_truth)
    
    # Find unexpected anomalies
    unexpected = find_unexpected_anomalies(df, args.output_dir, has_ground_truth)
    
    # Analyze missed anomalies
    missed_by_type = analyze_missed_anomalies(df, args.output_dir, has_ground_truth)
    
    # Create detailed report
    create_detailed_report(
        df, 
        type_stats, 
        unexpected, 
        missed_by_type, 
        args.output_dir, 
        args.report_file
    )
    
    # Plot figures if requested
    if args.plot_figures:
        plot_detection_by_type(type_stats, args.output_dir)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()