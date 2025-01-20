import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import logging
from collections import defaultdict
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class to store all paths and parameters"""
    anomalies_csv: Path
    processed_insights_folder: Path
    output_folder: Path
    matched_csv: Path
    unmatched_csv: Path
    missed_anomalies_csv: Path  # New: for anomalies found in insights but not by model
    plots_folder: Path
    n_workers: int = 4
    timestamp_format: str = "%H:%M:%S.%f-%Y/%m/%d"

class AnomalyAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self._setup_folders()
        
    def _setup_folders(self) -> None:
        """Create necessary output folders if they don't exist"""
        self.config.output_folder.mkdir(parents=True, exist_ok=True)
        self.config.plots_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_filename_from_timestamp(timestamp_str: str) -> Optional[str]:
        """Extract filename from timestamp string with improved error handling"""
        try:
            parts = timestamp_str.split(';')
            if not parts:
                return None
            file_part = parts[0]
            digits = re.findall(r'\d+', file_part)
            return f"{digits[0]}_processed_insights.csv" if digits else None
        except Exception as e:
            logger.warning(f"Error parsing filename from timestamp: {e}")
            return None

    @staticmethod
    def unify_processed_timestamp(ts: str) -> str:
        """Standardize timestamp format"""
        return ts.replace("_processed.csv;", ".csv;")

    def _extract_datetime(self, timestamp_str: str) -> Optional[datetime]:
        """Extract datetime object from timestamp string"""
        try:
            time_part = timestamp_str.split(';')[1]
            return datetime.strptime(time_part, self.config.timestamp_format)
        except Exception:
            return None

    def process_single_file(self, args: Tuple[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process a single file and return matched, unmatched, and missed anomalies DataFrames"""
        processed_file_name, subset_df = args
        
        if not isinstance(processed_file_name, str):
            return pd.DataFrame(), subset_df.copy(), pd.DataFrame()

        processed_path = self.config.processed_insights_folder / processed_file_name
        if not processed_path.exists():
            return pd.DataFrame(), subset_df.copy(), pd.DataFrame()

        # Read processed insights file
        processed_df = pd.read_csv(processed_path)
        processed_df['timestamp_str'] = processed_df['timestamp_str'].apply(self.unify_processed_timestamp)
        
        # Get timestamps from model-detected anomalies
        model_timestamps = set(subset_df['timestamp_str'])
        
        # Find anomalies in processed insights
        insight_anomalies = processed_df[processed_df['insight'].notna()]
        
        # Create insights dictionary for matching
        insights_dict = dict(zip(processed_df['timestamp_str'], processed_df['insight']))
        
        # Add insights to model-detected anomalies and split matched/unmatched
        subset_df['insight'] = subset_df['timestamp_str'].map(insights_dict)
        matched_mask = subset_df['insight'].notna()
        
        # Find missed anomalies (in insights but not detected by model)
        missed_anomalies = insight_anomalies[~insight_anomalies['timestamp_str'].isin(model_timestamps)].copy()
        
        return (
            subset_df[matched_mask].copy(),
            subset_df[~matched_mask].copy(),
            missed_anomalies
        )

    def analyze(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Main analysis function"""
        logger.info("Loading anomalies data...")
        anomalies_df = pd.read_csv(self.config.anomalies_csv)
        anomalies_df['processed_file'] = anomalies_df['timestamp_str'].apply(self.parse_filename_from_timestamp)
        
        # Add temporal information
        anomalies_df['datetime'] = anomalies_df['timestamp_str'].apply(self._extract_datetime)
        
        # Group files
        grouped = list(anomalies_df.groupby('processed_file', dropna=False))
        
        # Process files in parallel
        all_matched_list = []
        all_unmatched_list = []
        all_missed_list = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_single_file, grouped),
                total=len(grouped),
                desc="Processing files"
            ))
        
        for matched, unmatched, missed in results:
            if not matched.empty:
                all_matched_list.append(matched)
            if not unmatched.empty:
                all_unmatched_list.append(unmatched)
            if not missed.empty:
                all_missed_list.append(missed)

        matched_df = pd.concat(all_matched_list, ignore_index=True) if all_matched_list else pd.DataFrame()
        unmatched_df = pd.concat(all_unmatched_list, ignore_index=True) if all_unmatched_list else pd.DataFrame()
        missed_df = pd.concat(all_missed_list, ignore_index=True) if all_missed_list else pd.DataFrame()
        
        # Save results
        matched_df.to_csv(self.config.matched_csv, index=False)
        unmatched_df.drop(columns=['processed_file'], inplace=True)
        unmatched_df.to_csv(self.config.unmatched_csv, index=False)
        missed_df.to_csv(self.config.missed_anomalies_csv, index=False)
        
        # Log statistics
        logger.info(f"Found {len(matched_df)} matched anomalies")
        logger.info(f"Found {len(unmatched_df)} unmatched anomalies")
        logger.info(f"Found {len(missed_df)} anomalies in insights but not detected by model")
        
        if len(matched_df) > 0:
            detection_rate = len(matched_df) / (len(matched_df) + len(missed_df)) * 100
            logger.info(f"Model detection rate: {detection_rate:.2f}%")
        
        return matched_df, unmatched_df, missed_df

    def generate_visualizations(self, matched_df: pd.DataFrame, missed_df: pd.DataFrame) -> None:
        """Generate comprehensive visualizations focusing on file and rule analysis"""
        logger.info("Generating visualizations...")
        
        self._plot_file_rule_distribution(matched_df)
        self._plot_rule_analysis(matched_df)
        self._plot_model_performance_by_file(matched_df, missed_df)
        self._plot_detection_performance(matched_df, missed_df)

    def _simplify_filename(self, filename: str) -> str:
        """Extract numeric part from filename"""
        if pd.isna(filename):
            return "unknown"
        match = re.search(r'(\d+)', filename)
        return f"{match.group(1)}" if match else filename

    def _plot_file_rule_distribution(self, df: pd.DataFrame) -> None:
        """Generate file vs rule distribution heatmap with better space utilization"""
        rule_by_file = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            file = self._simplify_filename(row['processed_file'])
            if pd.isna(row['insight']):
                continue
            rules = [r.strip() for r in row['insight'].split(',')]
            for rule in rules:
                rule_by_file[file][rule] += 1
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(rule_by_file).fillna(0)
        
        def format_value(val):
            if val == 0:
                return ''
            elif val >= 1000:
                return f'{val/1000:.1f}k'
            else:
                return str(int(val))
        
        annotations = [[format_value(val) for val in row] for row in heatmap_df.values]
        
        # Create figure with specific dimensions
        plt.figure(figsize=(30, 10))  # Adjusted height for better aspect ratio
        
        # Create heatmap with adjusted parameters
        ax = sns.heatmap(
            heatmap_df,
            annot=annotations,
            fmt='',
            cmap='YlOrRd',
            annot_kws={'size': 8},
            cbar_kws={
                'label': 'Count',
                'orientation': 'vertical',
                'pad': 0.02  # Reduce padding between heatmap and colorbar
            },
            square=False  # Allow rectangular cells
        )
        
        plt.title('Distribution of Rules Across Files matched', pad=10)
        plt.xlabel('File Number')
        plt.ylabel('Rule Type')
        
        # Rotate x-axis labels and adjust their position
        plt.xticks(rotation=45, ha='right')
        
        # Adjust subplot parameters
        plt.subplots_adjust(
            left=0.1,    # Increase if labels are cut off
            right=0.95,  # Decrease if colorbar is cut off
            bottom=0.15, # Increase if x labels are cut off
            top=0.95     # Decrease if title is cut off
        )
        
        # Save with high DPI
        plt.savefig(
            self.config.plots_folder / 'file_rule_distribution.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()
    
    def _plot_detection_performance(self, matched_df: pd.DataFrame, missed_df: pd.DataFrame) -> None:
        """Generate overall detection performance analysis"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Overall detection pie chart
        plt.subplot(2, 1, 1)
        counts = [len(matched_df), len(missed_df)]
        labels = ['Detected by Model', 'Missed by Model']
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        plt.title('Overall Model Detection Performance matched')
        
        # Plot 2: Rule detection performance
        plt.subplot(2, 1, 2)
        
        def get_rule_counts(df: pd.DataFrame) -> pd.Series:
            rule_counts = defaultdict(int)
            for insight in df['insight'].dropna():
                for rule in insight.split(','):
                    rule_counts[rule.strip()] += 1
            return pd.Series(rule_counts)
        
        matched_rules = get_rule_counts(matched_df)
        missed_rules = get_rule_counts(missed_df)
        
        # Calculate detection rate per rule
        all_rules = set(matched_rules.index) | set(missed_rules.index)
        rule_performance = []
        
        for rule in all_rules:
            detected = matched_rules.get(rule, 0)
            missed = missed_rules.get(rule, 0)
            total = detected + missed
            detection_rate = (detected / total * 100) if total > 0 else 0
            rule_performance.append({
                'Rule': rule,
                'Detection Rate': detection_rate,
                'Total Cases': total
            })
        
        rule_perf_df = pd.DataFrame(rule_performance).sort_values('Total Cases', ascending=False)
        
        sns.barplot(data=rule_perf_df, x='Rule', y='Detection Rate')
        plt.title('Detection Rate by Rule Type')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Detection Rate (%)')
        
        plt.tight_layout()
        plt.savefig(self.config.plots_folder / 'overall_performance.png')
        plt.close()

    def _plot_rule_analysis(self, df: pd.DataFrame) -> None:
        """Generate rule-based analysis plots with simplified filenames"""
        # Rule occurrence by file
        rule_counts = defaultdict(lambda: defaultdict(int))
        total_files = df['processed_file'].nunique()
        
        for _, row in df.iterrows():
            if pd.isna(row['insight']):
                continue
            rules = [r.strip() for r in row['insight'].split(',')]
            file = self._simplify_filename(row['processed_file'])
            for rule in rules:
                rule_counts[rule][file] += 1
        
        # Calculate statistics
        rule_stats = []
        for rule in rule_counts:
            files_with_rule = len(rule_counts[rule])
            total_occurrences = sum(rule_counts[rule].values())
            avg_per_file = total_occurrences / files_with_rule
            coverage = (files_with_rule / total_files) * 100
            
            rule_stats.append({
                'Rule': rule,
                'Total Occurrences': total_occurrences,
                'Files Affected': files_with_rule,
                'Avg per File': avg_per_file,
                'File Coverage %': coverage
            })
        
        stats_df = pd.DataFrame(rule_stats).sort_values('Total Occurrences', ascending=False)
        
        # Plot rule statistics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Total occurrences and file coverage
        stats_df.plot(x='Rule', y=['Total Occurrences', 'File Coverage %'], 
                    kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Rule Occurrence and File Coverage matched')
        ax1.set_xticklabels(stats_df['Rule'], rotation=45, ha='right')
        ax1.legend(['Total Occurrences', 'File Coverage %'])
        
        # Average occurrences per affected file
        stats_df.plot(x='Rule', y='Avg per File', kind='bar', ax=ax2, color='green')
        ax2.set_title('Average Occurrences per Affected File matched')
        ax2.set_xticklabels(stats_df['Rule'], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.config.plots_folder / 'rule_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_performance_by_file(self, matched_df: pd.DataFrame, missed_df: pd.DataFrame) -> None:
        """Generate plots showing model performance per file with simplified filenames"""
        # Calculate detection rates per file
        file_stats = defaultdict(lambda: {'detected': 0, 'missed': 0})
        
        for _, row in matched_df.iterrows():
            file = self._simplify_filename(row['processed_file'])
            file_stats[file]['detected'] += 1
            
        for _, row in missed_df.iterrows():
            file = self._simplify_filename(self.parse_filename_from_timestamp(row['timestamp_str']))
            if file:
                file_stats[file]['missed'] += 1
        
        # Convert to DataFrame
        stats_list = []
        for file, stats in file_stats.items():
            total = stats['detected'] + stats['missed']
            detection_rate = (stats['detected'] / total * 100) if total > 0 else 0
            stats_list.append({
                'File': file,
                'Detection Rate': detection_rate,
                'Detected': stats['detected'],
                'Missed': stats['missed'],
                'Total': total
            })
        
        performance_df = pd.DataFrame(stats_list).sort_values('Total', ascending=False)
        
        # Plot detection performance by file
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Detection rate by file
        sns.barplot(data=performance_df, x='File', y='Detection Rate', ax=ax1, color='blue')
        ax1.set_title('Model Detection Rate by File matched')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_xlabel('File Number')
        ax1.set_ylabel('Detection Rate (%)')
        
        # Detected vs Missed by file
        performance_df.plot(x='File', y=['Detected', 'Missed'], 
                        kind='bar', stacked=True, ax=ax2)
        ax2.set_title('Detected vs Missed Anomalies by File matched')
        ax2.set_xlabel('File Number')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.config.plots_folder / 'model_performance_by_file.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    config = Config(
        anomalies_csv=Path("/workspaces/thesis/mlruns/894675987261661306/809606034f6e4682b2f327d249eff203/artifacts/detection_results/anomalies_p95q95v99_th0.1741.csv"),
        processed_insights_folder=Path("/workspaces/thesis/data/processed_pdsch_insights/"),
        output_folder=Path("/workspaces/thesis/insights"),
        matched_csv=Path("/workspaces/thesis/insights/matched_anomalies.csv"),
        unmatched_csv=Path("/workspaces/thesis/insights/unmatched_anomalies.csv"),
        missed_anomalies_csv=Path("/workspaces/thesis/insights/missed_anomalies.csv"),
        plots_folder=Path("/workspaces/thesis/insights/plots"),
        n_workers=4
    )
    
    analyzer = AnomalyAnalyzer(config)
    matched_df, unmatched_df, missed_df = analyzer.analyze()
    analyzer.generate_visualizations(matched_df, missed_df)
    
    logger.info("Analysis complete. Check the plots folder for visualizations.")

if __name__ == "__main__":
    main()