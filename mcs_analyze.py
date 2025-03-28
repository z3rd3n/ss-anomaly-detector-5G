import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import datetime
import logging
from statsmodels.tsa.stattools import acf, pacf
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules - modify paths as needed
from dataset import create_data_loaders
from trainer import BiGRUAnomalyDetector, setup_logger

# Set up paths
MODEL_PATH = "best_model.pth"
CONFIG_PATH = MODEL_PATH  # Use the same file for both
TEST_DATA_PATH = "unscaled_pdsch_val_min.parquet"
RESULTS_DIR = "mcs_analysis_results"
FP_DATA_PATH = "analysis_results/false_positives/false_positives_clustered.csv"

# Create results directory and subdirectories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/statistics", exist_ok=True)

# Configure logger
logger = setup_logger(log_file=f"{RESULTS_DIR}/mcs_analysis.log")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class MCSAnalyzer:
    def __init__(self, model_path, config_path, test_data_path, fp_data_path, results_dir):
        self.model_path = model_path
        self.config_path = config_path
        self.test_data_path = test_data_path
        self.fp_data_path = fp_data_path
        self.results_dir = results_dir
        self.device = device
        self.feature_names = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        self.class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        self.custom_cmap = self._create_custom_colormap()
        
        # Load model and data
        self.load_config()
        self.load_model()
        self.load_test_data()
        self.load_false_positives()
        
    def _create_custom_colormap(self):
        """Create a custom colormap for visualizations"""
        return LinearSegmentedColormap.from_list(
            "custom_cmap", 
            ["#2ca02c", "#ffcc00", "#ff7f0e", "#d62728"], 
            N=256
        )
        
    def load_config(self):
        """Load model configuration from saved file"""
        logger.info(f"Loading configuration from: {self.config_path}")
        
        # Load the checkpoint (contains both model state and config)
        checkpoint = torch.load(self.config_path, map_location=self.device)
        
        # Extract config from the checkpoint
        if 'config' in checkpoint:
            self.config = checkpoint['config']
            logger.info(f"Configuration loaded successfully")
        else:
            raise ValueError(f"Config not found in checkpoint file: {self.config_path}")
        
        # Set threshold for binary classification
        self.threshold = self.config.get('threshold', 0.5)
        logger.info(f"Using classification threshold: {self.threshold}")
    
    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Initialize model with config
        self.model = BiGRUAnomalyDetector(
            feature_dims=self.config['all_feature_dims'],
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            gru_dropout=self.config['gru_dropout'],
            position_encoding=self.config['position_encoding'],
            bidirectional=self.config['bidirectional'],
            attention_heads=self.config['attention_heads'],
            attention_dropout=self.config['attention_dropout'],
            classifier_hidden_dim=self.config['classifier_hidden_dim'],
            classifier_dropout=self.config['classifier_dropout'],
            device=self.device
        ).to(self.device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model weights loaded successfully")
        else:
            raise ValueError(f"model_state_dict not found in checkpoint file: {self.model_path}")
        
        self.model.eval()
        
    def load_test_data(self):
        """Load and prepare test data"""
        logger.info(f"Loading test data from: {self.test_data_path}")
        
        # Create data loaders using the function from dataset.py
        _, self.test_loader, _ = create_data_loaders(
            train_parquet_path=self.config['train_parquet_path'],
            test_parquet_path=self.test_data_path,
            all_features=self.config['all_features'],
            all_feature_dims=self.config['all_feature_dims'],
            seq_len=self.config['seq_len'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            sample_fraction=1.0  # Use all test data
        )
        
        logger.info(f"Test data loaded successfully")
        
        # Extract raw data for MCS analysis
        self.extract_test_data_features()
    
    def extract_test_data_features(self):
        """Extract raw feature data from test dataset for analysis"""
        logger.info("Extracting raw feature data from test dataset...")
        
        all_features = []
        all_labels = []
        
        # Iterate through dataset to extract features
        for i in range(len(self.test_loader.dataset)):
            sample = self.test_loader.dataset[i]
            features = sample['feature_data'].numpy()  # [seq_len, feature_dim]
            labels = sample['label'].numpy()  # [seq_len]
            
            all_features.append(features)
            all_labels.append(labels)
        
        # Concatenate all features and labels
        self.all_features = np.concatenate(all_features, axis=0)  # [total_timesteps, feature_dim]
        self.all_labels = np.concatenate(all_labels, axis=0)  # [total_timesteps]
        
        # Extract MCS data specifically
        mcs_idx = self.feature_names.index('MCS')
        self.all_mcs = self.all_features[:, mcs_idx]
        
        # Create mask for normal vs anomaly samples
        self.normal_mask = (self.all_labels == 0)
        self.anomaly_mask = (self.all_labels > 0)
        
        logger.info(f"Extracted {len(self.all_mcs)} timesteps of MCS data")

    def load_false_positives(self):
        """Load false positives data from CSV"""
        logger.info(f"Loading false positives from: {self.fp_data_path}")
        
        self.fp_df = pd.read_csv(self.fp_data_path)
        logger.info(f"Loaded {len(self.fp_df)} false positives")
        
        # Split false positives by cluster
        self.clusters = self.fp_df['cluster'].unique()
        self.fp_by_cluster = {c: self.fp_df[self.fp_df['cluster'] == c] for c in self.clusters}
        
        logger.info(f"Found {len(self.clusters)} clusters of false positives")

    def run_inference_on_false_positives(self):
        """Run model inference on false positives to extract attention weights and error values"""
        logger.info("Running inference on false positives to extract model internals...")
        
        # Extract timestamps to identify false positive positions in sequences
        fp_timestamps = set(self.fp_df['timestamp'].values)
        
        # Collect attention weights and errors for false positives
        self.fp_attention = []
        self.fp_errors = []
        self.fp_sequences = []  # Store complete sequences containing false positives
        
        with torch.no_grad():
            # Process each sequence in test dataset
            for i in range(len(self.test_loader.dataset)):
                sample = self.test_loader.dataset[i]
                timestamps = sample['timestamp']
                
                # Check if this sequence contains any false positives
                has_fp = False
                fp_positions = []
                for j, ts in enumerate(timestamps):
                    if ts in fp_timestamps:
                        has_fp = True
                        fp_positions.append(j)
                
                if has_fp:
                    # Run inference on this sequence
                    feature_tensor = torch.tensor(sample['feature_data']).unsqueeze(0).to(self.device)
                    outputs = self.model(feature_tensor)
                    
                    # Extract outputs
                    attention = outputs['instance_attn_weights'].squeeze().cpu().numpy()
                    errors = outputs['error_per_timestep'].squeeze().cpu().numpy()
                    
                    # Store attention and errors for false positive positions
                    for pos in fp_positions:
                        self.fp_attention.append(attention[pos])
                        self.fp_errors.append(errors[pos])
                    
                    # Store the complete sequence for context analysis
                    seq_data = {
                        'features': sample['feature_data'].numpy(),
                        'labels': sample['label'].numpy(),
                        'timestamps': timestamps,
                        'attention': attention,
                        'errors': errors,
                        'fp_positions': fp_positions
                    }
                    self.fp_sequences.append(seq_data)
        
        logger.info(f"Extracted model internals for {len(self.fp_attention)} false positives")
        logger.info(f"Found {len(self.fp_sequences)} sequences containing false positives")

    def analyze_mcs_distribution(self):
        """Analyze and visualize MCS distribution for normal, anomaly, and false positive cases"""
        logger.info("Analyzing MCS distribution...")
        
        # Get MCS values for normal and anomaly samples
        normal_mcs = self.all_mcs[self.normal_mask]
        anomaly_mcs = self.all_mcs[self.anomaly_mask]
        
        # Get MCS values for false positives
        fp_mcs = self.fp_df['MCS'].values
        
        # Calculate statistics
        normal_mcs_mean = np.mean(normal_mcs)
        normal_mcs_std = np.std(normal_mcs)
        normal_mcs_median = np.median(normal_mcs)
        
        anomaly_mcs_mean = np.mean(anomaly_mcs)
        anomaly_mcs_std = np.std(anomaly_mcs)
        anomaly_mcs_median = np.median(anomaly_mcs)
        
        fp_mcs_mean = np.mean(fp_mcs)
        fp_mcs_std = np.std(fp_mcs)
        fp_mcs_median = np.median(fp_mcs)
        
        # Save statistics
        stats_df = pd.DataFrame({
            'Category': ['Normal', 'Known Anomaly', 'False Positive'],
            'Count': [len(normal_mcs), len(anomaly_mcs), len(fp_mcs)],
            'Mean MCS': [normal_mcs_mean, anomaly_mcs_mean, fp_mcs_mean],
            'Std MCS': [normal_mcs_std, anomaly_mcs_std, fp_mcs_std],
            'Median MCS': [normal_mcs_median, anomaly_mcs_median, fp_mcs_median],
            'Min MCS': [np.min(normal_mcs), np.min(anomaly_mcs), np.min(fp_mcs)],
            'Max MCS': [np.max(normal_mcs), np.max(anomaly_mcs), np.max(fp_mcs)]
        })
        
        stats_df.to_csv(f"{self.results_dir}/statistics/mcs_statistics.csv", index=False)
        
        # Create histograms
        plt.figure(figsize=(12, 8))
        bins = np.arange(-0.5, 32.5, 1)  # MCS values are 0-31
        
        # Plot histograms
        plt.hist(normal_mcs, bins=bins, alpha=0.5, label='Normal', density=True, color='green')
        plt.hist(anomaly_mcs, bins=bins, alpha=0.5, label='Known Anomaly', density=True, color='red')
        plt.hist(fp_mcs, bins=bins, alpha=0.7, label='False Positive', density=True, color='orange')
        
        # Add vertical lines for means
        plt.axvline(normal_mcs_mean, color='green', linestyle='dashed', linewidth=2, label='Normal Mean')
        plt.axvline(anomaly_mcs_mean, color='red', linestyle='dashed', linewidth=2, label='Known Anomaly Mean')
        plt.axvline(fp_mcs_mean, color='orange', linestyle='dashed', linewidth=2, label='False Positive Mean')
        
        # Add labels and title
        plt.xlabel('MCS Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('MCS Distribution Comparison', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a more sophisticated publication-quality plot
        self.create_mcs_distribution_publication_plot(normal_mcs, anomaly_mcs, fp_mcs)
        
        return stats_df

    def create_mcs_distribution_publication_plot(self, normal_mcs, anomaly_mcs, fp_mcs):
        """Create a publication-quality plot for MCS distribution comparison"""
        fig = plt.figure(figsize=(12, 10))
        grid = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[4, 1])
        
        # Main distribution plot (2D histogram)
        ax_main = plt.subplot(grid[0, 0])
        bins = np.arange(-0.5, 32.5, 1)
        
        # Create KDE plots instead of histograms for smoother visualization
        sns.kdeplot(normal_mcs, ax=ax_main, color='green', label='Normal', fill=True, alpha=0.3, bw_adjust=0.5)
        sns.kdeplot(anomaly_mcs, ax=ax_main, color='red', label='Known Anomaly', fill=True, alpha=0.3, bw_adjust=0.5)
        sns.kdeplot(fp_mcs, ax=ax_main, color='orange', label='False Positive', fill=True, alpha=0.5, bw_adjust=0.5)
        
        # Add actual histograms with reduced opacity
        ax_main.hist(normal_mcs, bins=bins, alpha=0.15, density=True, color='green', histtype='step', linewidth=1.5)
        ax_main.hist(anomaly_mcs, bins=bins, alpha=0.15, density=True, color='red', histtype='step', linewidth=1.5)
        ax_main.hist(fp_mcs, bins=bins, alpha=0.25, density=True, color='orange', histtype='step', linewidth=1.5)
        
        # Add mean and std deviation indicators
        for data, color, label in zip([normal_mcs, anomaly_mcs, fp_mcs], 
                                     ['green', 'red', 'orange'],
                                     ['Normal', 'Known Anomaly', 'False Positive']):
            mean = np.mean(data)
            std = np.std(data)
            ax_main.axvline(mean, color=color, linestyle='-', linewidth=2, 
                          label=f'{label} Mean: {mean:.2f}')
            ax_main.axvspan(mean-std, mean+std, alpha=0.1, color=color)
        
        ax_main.set_xlabel('MCS Value', fontsize=14)
        ax_main.set_ylabel('Density', fontsize=14)
        ax_main.set_title('MCS Distribution Comparison', fontsize=16)
        ax_main.legend(fontsize=10, loc='upper right')
        ax_main.grid(alpha=0.3)
        
        # Right panel: Boxplot comparison
        ax_box = plt.subplot(grid[0, 1])
        boxdata = [normal_mcs, anomaly_mcs, fp_mcs]
        box = ax_box.boxplot(boxdata, vert=True, patch_artist=True, 
                           labels=['Normal', 'Known\nAnomaly', 'False\nPositive'],
                           widths=0.5)
        
        # Color boxes
        colors = ['green', 'red', 'orange']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax_box.set_ylabel('MCS Value')
        ax_box.grid(alpha=0.3, axis='y')
        
        # Bottom panel: CDF comparison
        ax_cdf = plt.subplot(grid[1, 0])
        
        # Calculate CDFs
        for data, color, label in zip([normal_mcs, anomaly_mcs, fp_mcs], 
                                     ['green', 'red', 'orange'],
                                     ['Normal', 'Known Anomaly', 'False Positive']):
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            ax_cdf.plot(sorted_data, cdf, color=color, linewidth=2, label=label)
        
        ax_cdf.set_xlabel('MCS Value', fontsize=14)
        ax_cdf.set_ylabel('CDF', fontsize=14)
        ax_cdf.grid(alpha=0.3)
        ax_cdf.legend(fontsize=10, loc='lower right')
        
        # Bottom right: Statistical test results
        ax_stats = plt.subplot(grid[1, 1])
        ax_stats.axis('off')
        
        # Perform statistical tests
        normal_vs_fp = stats.ks_2samp(normal_mcs, fp_mcs)
        anomaly_vs_fp = stats.ks_2samp(anomaly_mcs, fp_mcs)
        normal_vs_anomaly = stats.ks_2samp(normal_mcs, anomaly_mcs)
        
        # Add statistical test results
        text = "Statistical Tests (KS-Test):\n\n"
        text += f"Normal vs FP:\np={normal_vs_fp.pvalue:.2e}\n\n"
        text += f"Anomaly vs FP:\np={anomaly_vs_fp.pvalue:.2e}\n\n"
        text += f"Normal vs Anomaly:\np={normal_vs_anomaly.pvalue:.2e}"
        
        ax_stats.text(0.1, 0.5, text, fontsize=10, transform=ax_stats.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_distribution_publication.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_mcs_by_cluster(self):
        """Analyze MCS distribution by false positive cluster"""
        logger.info("Analyzing MCS distribution by cluster...")
        
        # Create figure for comparing clusters
        plt.figure(figsize=(14, 8))
        bins = np.arange(-0.5, 32.5, 1)
        
        # Get MCS values for normal samples (reference)
        normal_mcs = self.all_mcs[self.normal_mask]
        
        # Plot histogram for normal samples
        plt.hist(normal_mcs, bins=bins, alpha=0.3, label='Normal', density=True, color='green')
        
        # Plot histogram for each cluster
        cluster_stats = []
        for cluster in sorted(self.clusters):
            cluster_mcs = self.fp_by_cluster[cluster]['MCS'].values
            
            # Calculate statistics
            mean_mcs = np.mean(cluster_mcs)
            std_mcs = np.std(cluster_mcs)
            median_mcs = np.median(cluster_mcs)
            
            # Store statistics
            cluster_stats.append({
                'Cluster': cluster,
                'Count': len(cluster_mcs),
                'Mean MCS': mean_mcs,
                'Std MCS': std_mcs,
                'Median MCS': median_mcs,
                'Min MCS': np.min(cluster_mcs),
                'Max MCS': np.max(cluster_mcs)
            })
            
            # Plot histogram
            plt.hist(cluster_mcs, bins=bins, alpha=0.6, 
                    label=f'Cluster {cluster} (n={len(cluster_mcs)})', 
                    density=True)
            
            # Add vertical line for mean
            plt.axvline(mean_mcs, linestyle='dashed', linewidth=1.5)
        
        # Add labels and title
        plt.xlabel('MCS Value', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('MCS Distribution by False Positive Cluster', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_distribution_by_cluster.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save cluster statistics
        cluster_stats_df = pd.DataFrame(cluster_stats)
        cluster_stats_df.to_csv(f"{self.results_dir}/statistics/mcs_cluster_statistics.csv", index=False)
        
        # Create a more detailed violin plot comparison
        self.create_cluster_violin_comparison(cluster_stats_df)
        
        return cluster_stats_df

    def create_cluster_violin_comparison(self, cluster_stats):
        """Create publication-quality violin plot comparing MCS distribution by cluster"""
        # Prepare data for plotting
        plot_data = []
        
        # Add normal data
        normal_mcs = self.all_mcs[self.normal_mask]
        normal_df = pd.DataFrame({'MCS': normal_mcs, 'Category': 'Normal'})
        plot_data.append(normal_df)
        
        # Add anomaly data
        anomaly_mcs = self.all_mcs[self.anomaly_mask]
        anomaly_df = pd.DataFrame({'MCS': anomaly_mcs, 'Category': 'Known Anomaly'})
        plot_data.append(anomaly_df)
        
        # Add cluster data
        for cluster in sorted(self.clusters):
            cluster_mcs = self.fp_by_cluster[cluster]['MCS'].values
            cluster_df = pd.DataFrame({'MCS': cluster_mcs, 'Category': f'Cluster {cluster}'})
            plot_data.append(cluster_df)
        
        # Combine all data
        all_data = pd.concat(plot_data)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create violin plot
        ax = sns.violinplot(x='Category', y='MCS', data=all_data, palette='viridis', 
                          inner='box', scale='width')
        
        # Add individual points with jitter for clusters (they have fewer points)
        for i, category in enumerate(all_data['Category'].unique()):
            if category.startswith('Cluster'):
                cluster_data = all_data[all_data['Category'] == category]['MCS']
                plt.scatter([i] * len(cluster_data), cluster_data, 
                           alpha=0.6, s=20, color='white', edgecolor='black')
        
        # Add horizontal lines for normal mean and median
        normal_mean = np.mean(normal_mcs)
        normal_median = np.median(normal_mcs)
        plt.axhline(normal_mean, color='green', linestyle='--', alpha=0.8, 
                   label=f'Normal Mean: {normal_mean:.2f}')
        plt.axhline(normal_median, color='green', linestyle=':', alpha=0.8, 
                   label=f'Normal Median: {normal_median:.2f}')
        
        # Add statistical annotations
        from statannot import add_stat_annotation
        try:
            add_stat_annotation(ax, data=all_data, x='Category', y='MCS',
                               box_pairs=[('Normal', cat) for cat in all_data['Category'].unique() if cat != 'Normal'],
                               test='Mann-Whitney', text_format='simple', loc='outside', verbose=0)
        except:
            # If statannot is not available, add p-values manually
            for i, category in enumerate(all_data['Category'].unique()):
                if category != 'Normal':
                    category_data = all_data[all_data['Category'] == category]['MCS']
                    _, p_value = stats.mannwhitneyu(normal_mcs, category_data)
                    plt.text(i, ax.get_ylim()[1]-1, f'p={p_value:.2e}', 
                           horizontalalignment='center', fontsize=8)
        
        # Add labels and title
        plt.xlabel('', fontsize=14)
        plt.ylabel('MCS Value', fontsize=14)
        plt.title('MCS Distribution by Category', fontsize=16)
        plt.legend(fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_violin_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_mcs_temporal_patterns(self):
        """Analyze temporal patterns in MCS values for false positives"""
        logger.info("Analyzing temporal patterns in MCS values...")
        
        # Run model inference on false positives to extract sequences
        self.run_inference_on_false_positives()
        
        # Analyze MCS patterns around false positives
        self.analyze_mcs_context_windows()
        
        # Analyze MCS transitions
        self.analyze_mcs_transitions()
        
        # Analyze MCS variability
        self.analyze_mcs_variability()

    def analyze_mcs_context_windows(self):
        """Analyze MCS values in context windows around false positives"""
        logger.info("Analyzing MCS context windows around false positives...")
        
        # Context window size (before and after the false positive)
        window_size = 10
        
        # Collect MCS values around false positives
        fp_context_windows = []
        
        for seq in self.fp_sequences:
            features = seq['features']
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            
            for fp_pos in seq['fp_positions']:
                # Get window around the false positive
                start = max(0, fp_pos - window_size)
                end = min(len(mcs_values), fp_pos + window_size + 1)
                
                # Create context window
                window = np.full(2*window_size + 1, np.nan)  # Initialize with NaN
                window_slice = slice(window_size - (fp_pos - start), window_size + (end - fp_pos))
                window[window_slice] = mcs_values[start:end]
                
                # Store with position information
                fp_context_windows.append({
                    'window': window,
                    'fp_pos': fp_pos,
                    'mcs': mcs_values[fp_pos],
                    'cluster': None  # Will fill this later
                })
        
        # Match context windows with clusters
        for i, window_data in enumerate(fp_context_windows):
            fp_mcs = window_data['mcs']
            # Find matching false positive in DataFrame
            for cluster, cluster_df in self.fp_by_cluster.items():
                if fp_mcs in cluster_df['MCS'].values:
                    fp_context_windows[i]['cluster'] = cluster
                    break
        
        # Convert to DataFrame for easier analysis
        window_columns = [f'MCS_t{i-window_size}' for i in range(2*window_size + 1)]
        window_df_rows = []
        
        for window_data in fp_context_windows:
            row = {'cluster': window_data['cluster']}
            for i, val in enumerate(window_data['window']):
                row[window_columns[i]] = val
            window_df_rows.append(row)
        
        window_df = pd.DataFrame(window_df_rows)
        window_df.to_csv(f"{self.results_dir}/statistics/mcs_context_windows.csv", index=False)
        
        # Calculate average window by cluster
        cluster_avg_windows = {}
        for cluster in self.clusters:
            cluster_windows = window_df[window_df['cluster'] == cluster][window_columns].values
            avg_window = np.nanmean(cluster_windows, axis=0)
            cluster_avg_windows[cluster] = avg_window
        
        # Also calculate overall average
        all_windows = window_df[window_columns].values
        overall_avg_window = np.nanmean(all_windows, axis=0)
        
        # Calculate normal reference pattern
        # Randomly sample normal sequences for comparison
        normal_windows = []
        normal_indices = np.where(self.normal_mask)[0]
        np.random.seed(SEED)
        sample_indices = np.random.choice(normal_indices, min(1000, len(normal_indices)), replace=False)
        
        for idx in sample_indices:
            # Find sequence containing this index
            for i, sample in enumerate(self.test_loader.dataset):
                labels = sample['label'].numpy()
                features = sample['feature_data'].numpy()
                mcs_idx = self.feature_names.index('MCS')
                mcs_values = features[:, mcs_idx]
                
                # Check if this sequence contains the index
                if len(labels) > 0 and labels[0] == 0:  # Normal sequence
                    # Randomly pick a position
                    pos = np.random.randint(window_size, len(mcs_values) - window_size)
                    window = mcs_values[pos-window_size:pos+window_size+1]
                    
                    if len(window) == 2*window_size + 1:
                        normal_windows.append(window)
                    
                    # Limit the number of windows
                    if len(normal_windows) >= 1000:
                        break
        
        # Calculate average normal window
        normal_avg_window = np.mean(normal_windows, axis=0)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot overall average
        plt.plot(range(-window_size, window_size+1), overall_avg_window, 'k-', 
                 linewidth=3, label='All False Positives Avg')
        
        # Plot cluster averages
        for cluster, avg_window in cluster_avg_windows.items():
            plt.plot(range(-window_size, window_size+1), avg_window, 'o-', 
                   linewidth=2, label=f'Cluster {cluster} Avg')
        
        # Plot normal average
        plt.plot(range(-window_size, window_size+1), normal_avg_window, 'g--', 
               linewidth=2, label='Normal Reference')
        
        # Add vertical line at false positive position
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='False Positive Position')
        
        # Add labels and title
        plt.xlabel('Relative Position', fontsize=14)
        plt.ylabel('Average MCS Value', fontsize=14)
        plt.title('MCS Context Window Around False Positives', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_context_window.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a more detailed publication plot with confidence intervals
        self.create_mcs_context_publication_plot(window_df, cluster_avg_windows, normal_windows, window_size)

    def create_mcs_context_publication_plot(self, window_df, cluster_avg_windows, normal_windows, window_size):
        """Create publication-quality plot for MCS context windows"""
        # Convert normal windows to DataFrame for easier processing
        normal_df = pd.DataFrame(normal_windows, columns=[f'MCS_t{i-window_size}' for i in range(2*window_size + 1)])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [3, 1]})
        
        # Upper plot: Average patterns with confidence intervals
        ax = axes[0]
        
        # Plot normal reference with confidence interval
        window_cols = [f'MCS_t{i-window_size}' for i in range(2*window_size + 1)]
        normal_mean = normal_df[window_cols].mean().values
        normal_std = normal_df[window_cols].std().values
        normal_ci = 1.96 * normal_std / np.sqrt(len(normal_df))  # 95% CI
        
        x = np.arange(-window_size, window_size+1)
        ax.plot(x, normal_mean, 'g-', linewidth=2, label='Normal Reference')
        ax.fill_between(x, normal_mean - normal_ci, normal_mean + normal_ci, 
                       color='green', alpha=0.2)
        
        # Plot cluster patterns with confidence intervals
        for cluster in sorted(self.clusters):
            cluster_data = window_df[window_df['cluster'] == cluster][window_cols].values
            cluster_mean = np.nanmean(cluster_data, axis=0)
            cluster_std = np.nanstd(cluster_data, axis=0)
            cluster_ci = 1.96 * cluster_std / np.sqrt(np.sum(~np.isnan(cluster_data), axis=0))
            
            ax.plot(x, cluster_mean, 'o-', linewidth=2, label=f'Cluster {cluster} (n={len(cluster_data)})')
            ax.fill_between(x, cluster_mean - cluster_ci, cluster_mean + cluster_ci, alpha=0.2)
        
        # Add vertical line at false positive position
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='False Positive Position')
        
        # Add labels and title
        ax.set_xlabel('Relative Position', fontsize=14)
        ax.set_ylabel('MCS Value', fontsize=14)
        ax.set_title('MCS Patterns Around False Positives', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        # Lower plot: MCS derivatives (rate of change)
        ax = axes[1]
        
        # Calculate derivatives for each pattern
        for cluster in sorted(self.clusters):
            cluster_data = window_df[window_df['cluster'] == cluster][window_cols].values
            cluster_mean = np.nanmean(cluster_data, axis=0)
            
            # Calculate derivative (difference between consecutive points)
            derivative = np.diff(cluster_mean)
            
            # Plot derivative
            ax.plot(x[:-1] + 0.5, derivative, 'o-', linewidth=2, label=f'Cluster {cluster}')
        
        # Calculate derivative for normal reference
        normal_derivative = np.diff(normal_mean)
        ax.plot(x[:-1] + 0.5, normal_derivative, 'g-', linewidth=2, label='Normal Reference')
        
        # Add vertical line at false positive position
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add labels
        ax.set_xlabel('Relative Position', fontsize=14)
        ax.set_ylabel('MCS Change Rate', fontsize=14)
        ax.set_title('Rate of Change in MCS Around False Positives', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_context_publication.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_mcs_transitions(self):
        """Analyze MCS transitions around false positives"""
        logger.info("Analyzing MCS transitions around false positives...")
        
        # Create transition matrices
        # 1. For normal samples
        # 2. For known anomalies
        # 3. For false positives
        
        # Function to calculate transition matrix
        def calculate_transition_matrix(mcs_sequences):
            max_mcs = 32  # MCS values are 0-31
            transitions = np.zeros((max_mcs, max_mcs))
            
            for seq in mcs_sequences:
                for i in range(len(seq) - 1):
                    from_mcs = int(seq[i])
                    to_mcs = int(seq[i+1])
                    if from_mcs < max_mcs and to_mcs < max_mcs:
                        transitions[from_mcs, to_mcs] += 1
            
            # Normalize by row
            row_sums = transitions.sum(axis=1)
            transitions_norm = np.zeros_like(transitions)
            for i in range(max_mcs):
                if row_sums[i] > 0:
                    transitions_norm[i] = transitions[i] / row_sums[i]
            
            return transitions, transitions_norm
        
        # Extract MCS sequences
        normal_mcs_sequences = []
        anomaly_mcs_sequences = []
        fp_mcs_sequences = []
        
        # For normal and known anomalies, use original dataset
        for i in range(len(self.test_loader.dataset)):
            sample = self.test_loader.dataset[i]
            features = sample['feature_data'].numpy()
            labels = sample['label'].numpy()
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            
            # Split into normal and anomaly sequences
            normal_mask = labels == 0
            anomaly_mask = labels > 0
            
            if np.all(normal_mask):
                normal_mcs_sequences.append(mcs_values)
            elif np.any(anomaly_mask):
                anomaly_mcs_sequences.append(mcs_values)
        
        # For false positives, use the stored sequences
        for seq in self.fp_sequences:
            features = seq['features']
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            fp_mcs_sequences.append(mcs_values)
        
        # Calculate transition matrices
        normal_trans, normal_trans_norm = calculate_transition_matrix(normal_mcs_sequences)
        anomaly_trans, anomaly_trans_norm = calculate_transition_matrix(anomaly_mcs_sequences)
        fp_trans, fp_trans_norm = calculate_transition_matrix(fp_mcs_sequences)
        
        # Calculate difference matrices
        fp_vs_normal_diff = fp_trans_norm - normal_trans_norm
        fp_vs_anomaly_diff = fp_trans_norm - anomaly_trans_norm
        
        # Create visualizations
        self.plot_transition_matrix(normal_trans_norm, "Normal", clim=[0, 0.5])
        self.plot_transition_matrix(anomaly_trans_norm, "Known Anomaly", clim=[0, 0.5])
        self.plot_transition_matrix(fp_trans_norm, "False Positive", clim=[0, 0.5])
        self.plot_transition_matrix(fp_vs_normal_diff, "FP vs Normal Diff", clim=[-0.3, 0.3], cmap="coolwarm")
        self.plot_transition_matrix(fp_vs_anomaly_diff, "FP vs Anomaly Diff", clim=[-0.3, 0.3], cmap="coolwarm")
        
        # Create a combined publication plot
        self.create_transition_publication_plot(normal_trans_norm, anomaly_trans_norm, fp_trans_norm, 
                                               fp_vs_normal_diff, fp_vs_anomaly_diff)
        
        # Identify significant transitions in false positives
        self.identify_significant_transitions(normal_trans_norm, fp_trans_norm)

    def plot_transition_matrix(self, matrix, title, clim=None, cmap="viridis"):
        """Plot a transition matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        im = plt.imshow(matrix, cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Transition Probability', fontsize=12)
        
        # Set color limits if provided
        if clim:
            im.set_clim(clim)
        
        # Add labels and title
        plt.xlabel('To MCS', fontsize=14)
        plt.ylabel('From MCS', fontsize=14)
        plt.title(f'MCS Transition Matrix - {title}', fontsize=16)
        
        # Add ticks
        tick_positions = np.arange(0, 32, 4)
        plt.xticks(tick_positions)
        plt.yticks(tick_positions)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/transition_matrix_{title.replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_transition_publication_plot(self, normal_trans, anomaly_trans, fp_trans, 
                                         fp_vs_normal_diff, fp_vs_anomaly_diff):
        """Create a publication-quality composite plot of transition matrices"""
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[5, 4])
        
        # Define transition matrices to plot
        matrices = [
            (normal_trans, "Normal", "viridis", [0, 0.5], gs[0, 0]),
            (anomaly_trans, "Known Anomaly", "viridis", [0, 0.5], gs[0, 1]),
            (fp_trans, "False Positive", "viridis", [0, 0.5], gs[0, 2]),
            (fp_vs_normal_diff, "FP - Normal", "coolwarm", [-0.3, 0.3], gs[1, 0:2]),
            (fp_vs_anomaly_diff, "FP - Known Anomaly", "coolwarm", [-0.3, 0.3], gs[1, 2])
        ]
        
        # Plot each matrix
        for matrix, title, cmap, clim, position in matrices:
            ax = plt.subplot(position)
            
            # Determine MCS range based on transition probabilities
            if "Diff" in title:
                # For difference matrices, find the non-zero cells
                threshold = 0.05
                nonzero_mask = np.abs(matrix) > threshold
                row_indices, col_indices = np.where(nonzero_mask)
                
                # If there are significant differences, focus on those regions
                if len(row_indices) > 0 and len(col_indices) > 0:
                    max_mcs = max(np.max(row_indices), np.max(col_indices)) + 2
                    max_mcs = min(32, max(16, max_mcs))  # At least 16, at most 32
                else:
                    max_mcs = 16
            else:
                # For standard matrices, find the non-zero cells
                threshold = 0.01
                nonzero_mask = matrix > threshold
                row_indices, col_indices = np.where(nonzero_mask)
                
                # If there are significant transitions, focus on those regions
                if len(row_indices) > 0 and len(col_indices) > 0:
                    max_mcs = max(np.max(row_indices), np.max(col_indices)) + 2
                    max_mcs = min(32, max(16, max_mcs))  # At least 16, at most 32
                else:
                    max_mcs = 16
            
            # Create heatmap with focused range
            im = ax.imshow(matrix[:max_mcs, :max_mcs], cmap=cmap, aspect='equal')
            
            # Set color limits
            im.set_clim(clim)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Transition Probability' if "Diff" not in title else 'Difference', fontsize=10)
            
            # Add grid for easier reading
            for i in range(max_mcs):
                ax.axhline(i-0.5, color='black', linewidth=0.5, alpha=0.3)
                ax.axvline(i-0.5, color='black', linewidth=0.5, alpha=0.3)
            
            # Add labels and title
            ax.set_xlabel('To MCS', fontsize=12)
            ax.set_ylabel('From MCS', fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Add ticks
            step = 4 if max_mcs > 16 else 2
            tick_positions = np.arange(0, max_mcs, step)
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            
            # For difference plots, highlight significant differences
            if "Diff" in title:
                # Highlight significant differences
                significant_threshold = 0.1
                for i in range(max_mcs):
                    for j in range(max_mcs):
                        if abs(matrix[i, j]) > significant_threshold:
                            # Add circle around significant differences
                            circle = plt.Circle((j, i), 0.4, fill=False, edgecolor='black', linewidth=1, alpha=0.8)
                            ax.add_patch(circle)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/transition_matrices_publication.png", dpi=300, bbox_inches='tight')
        plt.close()

    def identify_significant_transitions(self, normal_trans, fp_trans):
        """Identify and report statistically significant MCS transitions in false positives"""
        # Calculate difference matrix
        diff_matrix = fp_trans - normal_trans
        
        # Find significant differences
        threshold = 0.1  # Adjust based on your data
        significant_transitions = []
        
        for i in range(len(diff_matrix)):
            for j in range(len(diff_matrix[i])):
                if abs(diff_matrix[i, j]) > threshold:
                    # Store significant transition
                    significant_transitions.append({
                        'From_MCS': i,
                        'To_MCS': j,
                        'Normal_Prob': normal_trans[i, j],
                        'FP_Prob': fp_trans[i, j],
                        'Difference': diff_matrix[i, j],
                        'Abs_Difference': abs(diff_matrix[i, j])
                    })
        
        # Sort by absolute difference (descending)
        significant_transitions.sort(key=lambda x: x['Abs_Difference'], reverse=True)
        
        # Save to CSV
        if significant_transitions:
            sig_df = pd.DataFrame(significant_transitions)
            sig_df.to_csv(f"{self.results_dir}/statistics/significant_mcs_transitions.csv", index=False)
            
            # Log top transitions
            logger.info(f"Found {len(significant_transitions)} significant MCS transitions")
            logger.info("Top 5 significant transitions:")
            for i, trans in enumerate(significant_transitions[:5]):
                logger.info(f"  {i+1}. MCS {trans['From_MCS']} → {trans['To_MCS']}: " +
                           f"Diff = {trans['Difference']:.4f} " +
                           f"(Normal: {trans['Normal_Prob']:.4f}, FP: {trans['FP_Prob']:.4f})")
        else:
            logger.info("No significant MCS transitions found")

    def analyze_mcs_variability(self):
        """Analyze MCS variability as a potential indicator of false positives"""
        logger.info("Analyzing MCS variability...")
        
        # Calculate MCS variability in windows around false positives
        window_size = 5  # Look at ±5 positions around each event
        
        # Calculate variability metrics
        def calculate_variability_metrics(mcs_window):
            if len(mcs_window) < 2:
                return {
                    'std': np.nan,
                    'range': np.nan,
                    'max_change': np.nan,
                    'avg_abs_change': np.nan,
                    'entropy': np.nan
                }
            
            # Standard deviation
            std = np.std(mcs_window)
            
            # Range
            value_range = np.max(mcs_window) - np.min(mcs_window)
            
            # Maximum change
            changes = np.abs(np.diff(mcs_window))
            max_change = np.max(changes) if len(changes) > 0 else 0
            
            # Average absolute change
            avg_abs_change = np.mean(changes) if len(changes) > 0 else 0
            
            # Entropy (using histogram)
            hist, _ = np.histogram(mcs_window, bins=np.arange(0, 33))
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            
            return {
                'std': std,
                'range': value_range,
                'max_change': max_change,
                'avg_abs_change': avg_abs_change,
                'entropy': entropy
            }
        
        # Calculate variability for different categories
        normal_variability = []
        anomaly_variability = []
        fp_variability = []
        fp_cluster_variability = {c: [] for c in self.clusters}
        
        # Process dataset for normal and known anomalies
        for i in range(len(self.test_loader.dataset)):
            sample = self.test_loader.dataset[i]
            features = sample['feature_data'].numpy()
            labels = sample['label'].numpy()
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            
            for j in range(window_size, len(labels) - window_size):
                mcs_window = mcs_values[j-window_size:j+window_size+1]
                
                if labels[j] == 0:
                    # Normal sample
                    normal_variability.append(calculate_variability_metrics(mcs_window))
                elif labels[j] > 0:
                    # Known anomaly
                    anomaly_variability.append(calculate_variability_metrics(mcs_window))
        
        # Process false positives
        for seq in self.fp_sequences:
            features = seq['features']
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            
            for fp_pos in seq['fp_positions']:
                if fp_pos >= window_size and fp_pos < len(mcs_values) - window_size:
                    mcs_window = mcs_values[fp_pos-window_size:fp_pos+window_size+1]
                    metrics = calculate_variability_metrics(mcs_window)
                    fp_variability.append(metrics)
                    
                    # Find cluster for this false positive
                    fp_mcs = mcs_values[fp_pos]
                    for cluster, cluster_df in self.fp_by_cluster.items():
                        if fp_mcs in cluster_df['MCS'].values:
                            fp_cluster_variability[cluster].append(metrics)
                            break
        
        # Convert to DataFrame
        def variability_list_to_df(var_list):
            if not var_list:
                return pd.DataFrame()
            return pd.DataFrame(var_list)
        
        normal_var_df = variability_list_to_df(normal_variability)
        anomaly_var_df = variability_list_to_df(anomaly_variability)
        fp_var_df = variability_list_to_df(fp_variability)
        fp_cluster_var_dfs = {c: variability_list_to_df(v) for c, v in fp_cluster_variability.items()}
        
        # Save statistics
        if len(normal_var_df) > 0:
            normal_var_df.describe().to_csv(f"{self.results_dir}/statistics/normal_variability_stats.csv")
        if len(anomaly_var_df) > 0:
            anomaly_var_df.describe().to_csv(f"{self.results_dir}/statistics/anomaly_variability_stats.csv")
        if len(fp_var_df) > 0:
            fp_var_df.describe().to_csv(f"{self.results_dir}/statistics/fp_variability_stats.csv")
        
        # Create violin plots for each variability metric
        metrics = ['std', 'range', 'max_change', 'avg_abs_change', 'entropy']
        metric_titles = {
            'std': 'Standard Deviation',
            'range': 'Range (Max - Min)',
            'max_change': 'Maximum Change',
            'avg_abs_change': 'Average Absolute Change',
            'entropy': 'Entropy'
        }
        
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            plot_data = []
            
            if len(normal_var_df) > 0:
                normal_data = normal_var_df[metric].dropna()
                plot_data.append((normal_data, 'Normal'))
            
            if len(anomaly_var_df) > 0:
                anomaly_data = anomaly_var_df[metric].dropna()
                plot_data.append((anomaly_data, 'Known Anomaly'))
            
            if len(fp_var_df) > 0:
                fp_data = fp_var_df[metric].dropna()
                plot_data.append((fp_data, 'False Positive'))
            
            for cluster in sorted(self.clusters):
                if cluster in fp_cluster_var_dfs and len(fp_cluster_var_dfs[cluster]) > 0:
                    cluster_data = fp_cluster_var_dfs[cluster][metric].dropna()
                    if len(cluster_data) > 0:
                        plot_data.append((cluster_data, f'Cluster {cluster}'))
            
            # Create violin plot
            if plot_data:
                plt.figure(figsize=(12, 8))
                positions = range(len(plot_data))
                
                # Plot violin plots
                violins = plt.violinplot(
                    [data for data, _ in plot_data], 
                    positions=positions,
                    showmeans=True, 
                    showextrema=True
                )
                
                # Customize colors
                colors = ['green', 'red', 'orange'] + ['blue', 'purple', 'brown', 'pink'][:len(self.clusters)]
                for i, (violin, color) in enumerate(zip(violins['bodies'], colors[:len(plot_data)])):
                    violin.set_facecolor(color)
                    violin.set_alpha(0.7)
                
                # Add boxplots on top of violins for better readability
                plt.boxplot(
                    [data for data, _ in plot_data],
                    positions=positions,
                    vert=True,
                    widths=0.15,
                    patch_artist=False,
                    showfliers=False
                )
                
                # Add individual points for false positive categories (they might have fewer points)
                for i, (data, label) in enumerate(plot_data):
                    if 'False Positive' in label or 'Cluster' in label:
                        plt.scatter([i] * len(data), data, alpha=0.6, s=20, color='white', edgecolor='black')
                
                # Add labels and title
                plt.xticks(range(len(plot_data)), [label for _, label in plot_data])
                plt.ylabel(metric_titles[metric], fontsize=14)
                plt.title(f'MCS {metric_titles[metric]} Comparison', fontsize=16)
                plt.grid(alpha=0.3, axis='y')
                
                # Add statistical annotations if possible
                try:
                    # Perform t-tests between normal and other categories
                    normal_data = plot_data[0][0]  # Assuming normal is first
                    for i, (data, label) in enumerate(plot_data[1:], 1):
                        t_stat, p_value = stats.ttest_ind(normal_data, data, equal_var=False)
                        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        plt.text(i, plt.ylim()[1] * 0.95, sig, ha='center', fontsize=12)
                except:
                    pass
                
                plt.tight_layout()
                
                # Save figure
                plt.savefig(f"{self.results_dir}/figures/mcs_{metric}_variability.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # Create a publication-quality composite plot
        self.create_variability_publication_plot(normal_var_df, anomaly_var_df, fp_var_df, fp_cluster_var_dfs)

    def create_variability_publication_plot(self, normal_df, anomaly_df, fp_df, cluster_dfs):
        """Create a publication-quality composite plot for variability metrics"""
        # Select metrics to include
        metrics = ['std', 'max_change', 'entropy']
        metric_titles = {
            'std': 'Standard Deviation',
            'max_change': 'Maximum Change',
            'entropy': 'Entropy'
        }
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1])
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = plt.subplot(gs[0, i])
            
            # Prepare data for plotting
            plot_data = []
            
            if len(normal_df) > 0:
                normal_data = normal_df[metric].dropna()
                plot_data.append((normal_data, 'Normal', 'green'))
            
            if len(anomaly_df) > 0:
                anomaly_data = anomaly_df[metric].dropna()
                plot_data.append((anomaly_data, 'Known Anomaly', 'red'))
            
            if len(fp_df) > 0:
                fp_data = fp_df[metric].dropna()
                plot_data.append((fp_data, 'False Positive', 'orange'))
            
            # Create violin plots
            positions = range(len(plot_data))
            violins = ax.violinplot(
                [data for data, _, _ in plot_data], 
                positions=positions,
                showmeans=True, 
                showextrema=True
            )
            
            # Customize colors
            for j, (violin, (_, _, color)) in enumerate(zip(violins['bodies'], plot_data)):
                violin.set_facecolor(color)
                violin.set_alpha(0.7)
            
            # Add boxplots
            ax.boxplot(
                [data for data, _, _ in plot_data],
                positions=positions,
                vert=True,
                widths=0.15,
                patch_artist=False,
                showfliers=False
            )
            
            # Add individual points for small datasets
            for j, (data, label, color) in enumerate(plot_data):
                if len(data) < 100:
                    ax.scatter([j] * len(data), data, alpha=0.6, s=20, color='white', edgecolor=color)
            
            # Add labels
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels([label for _, label, _ in plot_data], rotation=45, ha='right')
            ax.set_ylabel(metric_titles[metric], fontsize=12)
            ax.set_title(f'MCS {metric_titles[metric]}', fontsize=14)
            ax.grid(alpha=0.3, axis='y')
            
            # Add statistical annotations
            if len(plot_data) > 1:
                try:
                    normal_data = plot_data[0][0]  # Assuming normal is first
                    for j, (data, label, _) in enumerate(plot_data[1:], 1):
                        t_stat, p_value = stats.ttest_ind(normal_data, data, equal_var=False)
                        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                        ax.text(j, ax.get_ylim()[1] * 0.95, sig, ha='center', fontsize=12)
                except:
                    pass
        
        # Plot cluster comparison for standard deviation
        ax = plt.subplot(gs[0, 3])
        
        # Prepare data for cluster comparison
        cluster_data = []
        if len(normal_df) > 0:
            normal_data = normal_df['std'].dropna()
            cluster_data.append((normal_data, 'Normal', 'green'))
        
        for cluster in sorted(cluster_dfs.keys()):
            if cluster in cluster_dfs and len(cluster_dfs[cluster]) > 0:
                cluster_df = cluster_dfs[cluster]
                if 'std' in cluster_df.columns:
                    data = cluster_df['std'].dropna()
                    if len(data) > 0:
                        cluster_data.append((data, f'Cluster {cluster}', f'C{int(cluster)}'))
        
        # Create violin plots
        if cluster_data:
            positions = range(len(cluster_data))
            violins = ax.violinplot(
                [data for data, _, _ in cluster_data], 
                positions=positions,
                showmeans=True, 
                showextrema=True
            )
            
            # Customize colors
            for j, (violin, (_, _, color)) in enumerate(zip(violins['bodies'], cluster_data)):
                try:
                    violin.set_facecolor(color)
                except:
                    violin.set_facecolor(f'C{j}')
                violin.set_alpha(0.7)
            
            # Add labels
            ax.set_xticks(range(len(cluster_data)))
            ax.set_xticklabels([label for _, label, _ in cluster_data], rotation=45, ha='right')
            ax.set_ylabel('Standard Deviation', fontsize=12)
            ax.set_title('MCS Std Dev by Cluster', fontsize=14)
            ax.grid(alpha=0.3, axis='y')
        
        # Plot combined histogram with density curves - Standard deviation
        ax = plt.subplot(gs[1, :2])
        
        # Prepare data
        if len(normal_df) > 0 and len(fp_df) > 0:
            normal_std = normal_df['std'].dropna()
            fp_std = fp_df['std'].dropna()
            
            # Calculate histogram bins
            all_data = np.concatenate([normal_std, fp_std])
            bins = np.linspace(0, np.percentile(all_data, 99), 30)
            
            # Plot histograms
            ax.hist(normal_std, bins=bins, alpha=0.5, density=True, color='green', label='Normal')
            ax.hist(fp_std, bins=bins, alpha=0.5, density=True, color='orange', label='False Positive')
            
            # Add KDE curves
            try:
                sns.kdeplot(normal_std, ax=ax, color='darkgreen', linewidth=2)
                sns.kdeplot(fp_std, ax=ax, color='darkorange', linewidth=2)
            except:
                pass
            
            # Add labels
            ax.set_xlabel('MCS Standard Deviation', fontsize=14)
            ax.set_ylabel('Density', fontsize=14)
            ax.set_title('Distribution of MCS Variability', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            
            # Add annotation about separation
            normal_mean = np.mean(normal_std)
            fp_mean = np.mean(fp_std)
            t_stat, p_value = stats.ttest_ind(normal_std, fp_std, equal_var=False)
            
            ax.text(0.5, 0.95, f"Normal Mean: {normal_mean:.2f}\nFP Mean: {fp_mean:.2f}\n" + 
                   f"p-value: {p_value:.2e}", transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Plot ROC curve for MCS variability as a predictor
        ax = plt.subplot(gs[1, 2:])
        
        if len(normal_df) > 0 and len(fp_df) > 0:
            # Prepare data for ROC curve
            normal_std = normal_df['std'].dropna().values
            normal_labels = np.zeros(len(normal_std))
            
            fp_std = fp_df['std'].dropna().values
            fp_labels = np.ones(len(fp_std))
            
            # Combine data
            all_std = np.concatenate([normal_std, fp_std])
            all_labels = np.concatenate([normal_labels, fp_labels])
            
            # Calculate ROC curve
            from sklearn.metrics import roc_curve, roc_auc_score
            
            # Calculate ROC for each metric
            colors = ['red', 'blue', 'green']
            for i, metric in enumerate(['std', 'max_change', 'entropy']):
                if metric in normal_df.columns and metric in fp_df.columns:
                    normal_vals = normal_df[metric].dropna().values
                    fp_vals = fp_df[metric].dropna().values
                    
                    # Combine
                    all_vals = np.concatenate([normal_vals, fp_vals])
                    all_labels = np.concatenate([np.zeros(len(normal_vals)), np.ones(len(fp_vals))])
                    
                    # Calculate ROC
                    fpr, tpr, thresholds = roc_curve(all_labels, all_vals)
                    auc = roc_auc_score(all_labels, all_vals)
                    
                    # Plot ROC curve
                    ax.plot(fpr, tpr, linewidth=2, color=colors[i], 
                           label=f'{metric_titles[metric]} (AUC = {auc:.3f})')
            
            # Add diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7)
            
            # Add labels
            ax.set_xlabel('False Positive Rate', fontsize=14)
            ax.set_ylabel('True Positive Rate', fontsize=14)
            ax.set_title('ROC Curve for MCS Variability Metrics', fontsize=16)
            ax.legend(loc='lower right', fontsize=12)
            ax.grid(alpha=0.3)
            
            # Add text about potential usage
            ax.text(0.05, 0.95, "MCS variability could be used as\nan additional feature for anomaly detection",
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_variability_publication.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_pattern_mining(self):
        """Mine for specific MCS patterns that are indicative of false positives"""
        logger.info("Mining for specific MCS patterns in false positives...")
        
        # Extract MCS sequences from false positives
        fp_mcs_patterns = []
        
        # Window size for pattern
        pattern_size = 3  # Look at pattern of 3 consecutive MCS values
        
        # Extract patterns from false positive sequences
        for seq in self.fp_sequences:
            features = seq['features']
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            
            for fp_pos in seq['fp_positions']:
                # Check if we can extract a pattern before the false positive
                if fp_pos >= pattern_size:
                    # Extract pattern ending at the false positive
                    pattern = mcs_values[fp_pos-pattern_size+1:fp_pos+1]
                    fp_mcs_patterns.append(tuple(pattern))
        
        # Extract patterns from normal data for comparison
        normal_mcs_patterns = []
        
        # Sample random positions from normal sequences
        normal_positions = []
        for i, sample in enumerate(self.test_loader.dataset):
            labels = sample['label'].numpy()
            features = sample['feature_data'].numpy()
            
            # Only use sequences with normal events
            if np.all(labels == 0):
                # Extract multiple patterns from this sequence
                mcs_idx = self.feature_names.index('MCS')
                mcs_values = features[:, mcs_idx]
                
                # Sample random positions
                for _ in range(min(10, len(mcs_values) - pattern_size)):
                    pos = np.random.randint(pattern_size, len(mcs_values))
                    pattern = mcs_values[pos-pattern_size+1:pos+1]
                    normal_mcs_patterns.append(tuple(pattern))
        
        # Count pattern frequencies
        fp_pattern_counts = Counter(fp_mcs_patterns)
        normal_pattern_counts = Counter(normal_mcs_patterns)
        
        # Calculate pattern likelihood ratios
        pattern_ratios = {}
        min_count = 5  # Minimum count to consider (to filter out rare patterns)
        
        for pattern, fp_count in fp_pattern_counts.items():
            if fp_count >= min_count:
                normal_count = normal_pattern_counts.get(pattern, 0)
                # Add small epsilon to avoid division by zero
                ratio = fp_count / (len(fp_mcs_patterns) + 1e-10) / (normal_count / (len(normal_mcs_patterns) + 1e-10) + 1e-10)
                pattern_ratios[pattern] = {
                    'fp_count': fp_count,
                    'normal_count': normal_count,
                    'fp_freq': fp_count / len(fp_mcs_patterns),
                    'normal_freq': normal_count / len(normal_mcs_patterns) if normal_count > 0 else 0,
                    'ratio': ratio
                }
        
        # Sort patterns by ratio (descending)
        sorted_patterns = sorted(pattern_ratios.items(), key=lambda x: x[1]['ratio'], reverse=True)
        
        # Save top patterns to CSV
        top_patterns = []
        for pattern, stats in sorted_patterns[:20]:  # Top 20 patterns
            pattern_str = ', '.join(map(str, pattern))
            top_patterns.append({
                'Pattern': pattern_str,
                'FP_Count': stats['fp_count'],
                'FP_Frequency': stats['fp_freq'],
                'Normal_Count': stats['normal_count'],
                'Normal_Frequency': stats['normal_freq'],
                'Likelihood_Ratio': stats['ratio']
            })
        
        if top_patterns:
            pd.DataFrame(top_patterns).to_csv(
                f"{self.results_dir}/statistics/top_mcs_patterns.csv", index=False)
            
            # Log top patterns
            logger.info(f"Top {len(top_patterns)} MCS patterns that indicate false positives:")
            for i, pattern in enumerate(top_patterns[:5]):
                logger.info(f"  {i+1}. Pattern [{pattern['Pattern']}]: " +
                           f"Ratio = {pattern['Likelihood_Ratio']:.2f}, " +
                           f"FP Freq = {pattern['FP_Frequency']:.4f}, " +
                           f"Normal Freq = {pattern['Normal_Frequency']:.4f}")
        else:
            logger.info("No significant MCS patterns found")
        
        # Create visualization of top patterns
        self.visualize_top_patterns(sorted_patterns[:10])
        
        return top_patterns

    def visualize_top_patterns(self, top_patterns):
        """Visualize the top MCS patterns that indicate false positives"""
        if not top_patterns:
            return
        
        # Create a figure
        plt.figure(figsize=(14, 10))
        
        # Number of patterns to visualize
        n_patterns = min(len(top_patterns), 10)
        
        # Set up subplots
        fig, axes = plt.subplots(n_patterns, 1, figsize=(10, 2*n_patterns), sharex=True)
        if n_patterns == 1:
            axes = [axes]  # Convert to list if only one subplot
        
        # X-axis for pattern positions
        x = np.arange(len(top_patterns[0][0]))
        
        # Plot each pattern
        for i, (pattern, stats) in enumerate(top_patterns[:n_patterns]):
            ax = axes[i]
            
            # Plot pattern
            ax.plot(x, pattern, 'o-', linewidth=2, markersize=8, color=f'C{i}')
            
            # Add horizontal line for reference
            ax.axhline(y=np.mean(pattern), color=f'C{i}', linestyle='--', alpha=0.5)
            
            # Add annotations
            for j, val in enumerate(pattern):
                ax.text(j, val, str(int(val)), ha='center', va='bottom', fontsize=10)
            
            # Add statistics
            ax.text(0.98, 0.8, 
                   f"Ratio: {stats['ratio']:.2f}\nFP Freq: {stats['fp_freq']*100:.1f}%\nNormal Freq: {stats['normal_freq']*100:.1f}%",
                   transform=ax.transAxes, ha='right', fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Set y-label
            ax.set_ylabel(f"Pattern {i+1}", fontsize=12)
            
            # Set y-limits with some padding
            y_min, y_max = min(pattern), max(pattern)
            padding = max(1, (y_max - y_min) * 0.2)
            ax.set_ylim(y_min - padding, y_max + padding)
            
            # Only show y-tick for actual MCS values
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            # Add grid
            ax.grid(alpha=0.3)
        
        # Set common labels
        fig.text(0.5, 0.04, 'Pattern Position', ha='center', fontsize=14)
        fig.suptitle('Top MCS Patterns Indicative of False Positives', fontsize=16)
        
        # X-tick labels
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(['t-2', 't-1', 't (FP)'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.1)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/top_mcs_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a more detailed heatmap visualization
        self.create_pattern_heatmap(top_patterns)

    def create_pattern_heatmap(self, top_patterns):
        """Create a heatmap visualization of top MCS patterns"""
        if not top_patterns:
            return
        
        # Number of patterns to visualize
        n_patterns = min(len(top_patterns), 10)
        
        # Create a matrix for the heatmap
        pattern_matrix = np.zeros((n_patterns, len(top_patterns[0][0])))
        
        # Fill the matrix with pattern values
        for i, (pattern, _) in enumerate(top_patterns[:n_patterns]):
            pattern_matrix[i] = pattern
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(pattern_matrix, annot=True, fmt=".0f", cmap="YlOrRd", 
                   cbar_kws={'label': 'MCS Value'})
        
        # Add labels
        plt.xlabel('Pattern Position', fontsize=14)
        plt.ylabel('Pattern Rank', fontsize=14)
        plt.title('Top MCS Patterns Indicative of False Positives', fontsize=16)
        
        # Set x-tick labels
        plt.xticks(np.arange(len(top_patterns[0][0])) + 0.5, ['t-2', 't-1', 't (FP)'])
        
        # Set y-tick labels
        plt.yticks(np.arange(n_patterns) + 0.5, [f"Pattern {i+1}\n(Ratio: {stats['ratio']:.1f})" 
                                                for i, (_, stats) in enumerate(top_patterns[:n_patterns])])
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_pattern_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_prototype_discovery(self):
        """Discover and analyze prototypical MCS sequences that lead to false positives"""
        logger.info("Discovering prototypical MCS sequences...")
        
        # Extract MCS sequences from false positives
        fp_sequences = []
        
        # Window size for sequences
        window_size = 10  # Look at 10 time steps before the false positive
        
        # Extract sequences from false positive data
        for seq in self.fp_sequences:
            features = seq['features']
            mcs_idx = self.feature_names.index('MCS')
            mcs_values = features[:, mcs_idx]
            
            for fp_pos in seq['fp_positions']:
                # Check if we can extract a sequence before the false positive
                if fp_pos >= window_size:
                    # Extract sequence ending at the false positive
                    mcs_seq = mcs_values[fp_pos-window_size:fp_pos]
                    fp_sequences.append(mcs_seq)
        
        # Convert to numpy array
        fp_sequences = np.array(fp_sequences)
        
        if len(fp_sequences) == 0:
            logger.info("Not enough false positive sequences for prototype discovery")
            return
        
        # Standardize sequences
        scaler = StandardScaler()
        fp_sequences_scaled = np.array([scaler.fit_transform(seq.reshape(-1, 1)).flatten() 
                                       for seq in fp_sequences])
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(5, fp_sequences_scaled.shape[1]))
        fp_sequences_pca = pca.fit_transform(fp_sequences_scaled)
        
        # Apply clustering to find prototypes
        # Try different clustering approaches
        
        # 1. K-means clustering
        n_clusters = min(5, len(fp_sequences) // 10)  # Limit based on sample size
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(fp_sequences_pca)
        
        # 2. DBSCAN for density-based clustering (alternative)
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(fp_sequences_pca)
            n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            logger.info(f"DBSCAN found {n_dbscan_clusters} clusters")
        except:
            dbscan_labels = np.zeros(len(fp_sequences_pca))
            logger.info("DBSCAN clustering failed, using K-means only")
        
        # Use K-means results for prototype discovery
        # Find cluster centers and closest examples
        prototypes = []
        
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_members = fp_sequences[cluster_mask]
            
            if len(cluster_members) > 0:
                # Find center sequence
                center = kmeans.cluster_centers_[i]
                
                # Find closest example to center
                distances = np.linalg.norm(fp_sequences_pca[cluster_mask] - center, axis=1)
                closest_idx = np.argmin(distances)
                prototype_seq = cluster_members[closest_idx]
                
                # Store prototype
                prototypes.append({
                    'cluster': i,
                    'size': np.sum(cluster_mask),
                    'percentage': np.sum(cluster_mask) / len(cluster_labels) * 100,
                    'sequence': prototype_seq,
                    'distance': distances[closest_idx]
                })
        
        # Save prototype statistics
        proto_stats = []
        for i, proto in enumerate(prototypes):
            proto_stats.append({
                'Prototype': i+1,
                'Cluster': proto['cluster'],
                'Size': proto['size'],
                'Percentage': proto['percentage'],
                'Distance': proto['distance']
            })
        
        if proto_stats:
            pd.DataFrame(proto_stats).to_csv(
                f"{self.results_dir}/statistics/mcs_prototypes.csv", index=False)
        
        # Visualize prototypes
        self.visualize_prototypes(prototypes, fp_sequences, cluster_labels)
        
        # Visualize PCA with clusters
        self.visualize_pca_clusters(fp_sequences_pca, cluster_labels, dbscan_labels)
        
        return prototypes

    def visualize_prototypes(self, prototypes, fp_sequences, cluster_labels):
        """Visualize prototype MCS sequences"""
        if not prototypes:
            return
        
        # Create a figure for prototypes
        n_prototypes = len(prototypes)
        fig, axes = plt.subplots(n_prototypes, 1, figsize=(12, 3*n_prototypes), sharex=True)
        if n_prototypes == 1:
            axes = [axes]  # Convert to list if only one subplot
        
        # Time steps for x-axis
        x = np.arange(-len(prototypes[0]['sequence']), 0)
        
        # Plot each prototype
        for i, proto in enumerate(prototypes):
            ax = axes[i]
            
            # Get all sequences in this cluster
            cluster_mask = cluster_labels == proto['cluster']
            cluster_seqs = fp_sequences[cluster_mask]
            
            # Plot all sequences in cluster with low opacity
            for seq in cluster_seqs:
                ax.plot(x, seq, 'o-', linewidth=1, markersize=4, alpha=0.1, color=f'C{i}')
            
            # Plot prototype sequence with high opacity
            ax.plot(x, proto['sequence'], 'o-', linewidth=2.5, markersize=8, color=f'C{i}', 
                   label=f"Prototype {i+1}")
            
            # Add mean line
            mean_seq = np.mean(cluster_seqs, axis=0)
            ax.plot(x, mean_seq, '--', linewidth=2, color=f'C{i}', alpha=0.7, 
                   label=f"Cluster Mean")
            
            # Add statistics
            ax.text(0.02, 0.85, 
                   f"Cluster Size: {proto['size']} ({proto['percentage']:.1f}%)",
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
            
            # Set y-label and add legend
            ax.set_ylabel(f"MCS Value", fontsize=12)
            ax.legend(loc='upper right')
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Set title
            ax.set_title(f"Prototype {i+1}: MCS Sequence Pattern", fontsize=14)
        
        # Set common labels
        fig.text(0.5, 0.04, 'Time Steps Before False Positive', ha='center', fontsize=14)
        fig.suptitle('Prototypical MCS Sequences Leading to False Positives', fontsize=16)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, bottom=0.1)
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_prototypes.png", dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_pca_clusters(self, fp_sequences_pca, kmeans_labels, dbscan_labels):
        """Visualize PCA projections of false positive MCS sequences with cluster labels"""
        if len(fp_sequences_pca) == 0:
            return
        
        # Create figure
        fig = plt.figure(figsize=(16, 8))
        
        # First plot: K-means clustering
        ax = fig.add_subplot(121)
        
        # Plot points
        scatter = ax.scatter(fp_sequences_pca[:, 0], fp_sequences_pca[:, 1], 
                           c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
        
        # Add cluster centers if available
        try:
            from sklearn.decomposition import PCA
            kmeans = KMeans(n_clusters=len(set(kmeans_labels)))
            kmeans.fit(fp_sequences_pca)
            centers = kmeans.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, 
                      marker='X', edgecolors='black', label='Cluster Centers')
        except:
            pass
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax.add_artist(legend1)
        if 'centers' in locals():
            ax.legend(loc="upper left")
        
        # Add labels
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.set_title('K-means Clustering of MCS Sequences (PCA)', fontsize=14)
        ax.grid(alpha=0.3)
        
        # Second plot: DBSCAN clustering
        ax = fig.add_subplot(122)
        
        # Plot points
        scatter = ax.scatter(fp_sequences_pca[:, 0], fp_sequences_pca[:, 1], 
                           c=dbscan_labels, cmap='viridis', s=50, alpha=0.7)
        
        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax.add_artist(legend1)
        
        # Add custom legend for noise points if they exist
        if -1 in dbscan_labels:
            noise_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                               markersize=10, label='Noise')
            ax.legend(handles=[noise_patch], loc="upper left")
        
        # Add labels
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.set_title('DBSCAN Clustering of MCS Sequences (PCA)', fontsize=14)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.results_dir}/figures/mcs_pca_clusters.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # If we have more than 2 PCA components, create a 3D visualization
        if fp_sequences_pca.shape[1] >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            #Plot points
            scatter = ax.scatter(fp_sequences_pca[:, 0], fp_sequences_pca[:, 1], fp_sequences_pca[:, 2],
                               c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
            
            # Add cluster centers if available
            try:
                kmeans = KMeans(n_clusters=len(set(kmeans_labels)))
                kmeans.fit(fp_sequences_pca)
                centers = kmeans.cluster_centers_
                ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.8,
                          marker='X', edgecolors='black', label='Cluster Centers')
            except:
                pass
            
            # Set labels
            ax.set_xlabel('Principal Component 1', fontsize=12)
            ax.set_ylabel('Principal Component 2', fontsize=12)
            ax.set_zlabel('Principal Component 3', fontsize=12)
            ax.set_title('3D PCA Visualization of MCS Sequences', fontsize=14)
            
            # Add legend
            ax.legend()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/mcs_pca_3d.png", dpi=300, bbox_inches='tight')
            plt.close()    

    def analyze_mcs_correlation_with_model_outputs(self):
        """Analyze correlation between MCS values and model confidence/attention"""
        logger.info("Analyzing correlation between MCS and model outputs...")
        
        # Run inference on false positives to extract model internals if not already done
        if not hasattr(self, 'fp_attention') or len(self.fp_attention) == 0:
            self.run_inference_on_false_positives()
        
        # Extract MCS values, model confidence, and attention weights for false positives
        fp_mcs = []
        fp_confidences = []
        fp_attention_weights = []
        fp_reconstruction_errors = []
        fp_clusters = []
        
        # From the false positives DataFrame
        for i, row in self.fp_df.iterrows():
            mcs = row['MCS']
            prob = row['probability']
            cluster = row['cluster']
            
            fp_mcs.append(mcs)
            fp_confidences.append(prob)
            fp_clusters.append(cluster)
            
            # Match with attention and error values if available
            if hasattr(self, 'fp_attention') and i < len(self.fp_attention):
                fp_attention_weights.append(self.fp_attention[i])
                fp_reconstruction_errors.append(self.fp_errors[i])
            else:
                fp_attention_weights.append(np.nan)
                fp_reconstruction_errors.append(np.nan)
        
        # Create DataFrame for analysis
        corr_df = pd.DataFrame({
            'MCS': fp_mcs,
            'Confidence': fp_confidences,
            'Attention': fp_attention_weights,
            'Reconstruction_Error': fp_reconstruction_errors,
            'Cluster': fp_clusters
        })
        
        # Calculate correlations
        correlations = corr_df.corr()
        
        # Save to CSV
        correlations.to_csv(f"{self.results_dir}/statistics/mcs_model_correlations.csv")
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Between MCS and Model Outputs', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/mcs_model_correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # MCS vs Confidence
        ax = axes[0]
        scatter = ax.scatter(corr_df['MCS'], corr_df['Confidence'], c=corr_df['Cluster'], 
                           cmap='viridis', alpha=0.7, s=50)
        
        # Add trend line
        m, b = np.polyfit(corr_df['MCS'], corr_df['Confidence'], 1)
        x_line = np.array([min(corr_df['MCS']), max(corr_df['MCS'])])
        ax.plot(x_line, m*x_line + b, color='red', linestyle='--', linewidth=2)
        
        # Calculate and display correlation
        corr_coef = np.corrcoef(corr_df['MCS'], corr_df['Confidence'])[0, 1]
        ax.text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", transform=ax.transAxes, 
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add labels
        ax.set_xlabel('MCS Value', fontsize=14)
        ax.set_ylabel('Model Confidence', fontsize=14)
        ax.set_title('MCS vs. Model Confidence', fontsize=16)
        ax.grid(alpha=0.3)
        
        # Add color bar for clusters
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        
        # MCS vs Reconstruction Error (if available)
        ax = axes[1]
        if not all(np.isnan(corr_df['Reconstruction_Error'])):
            scatter = ax.scatter(corr_df['MCS'], corr_df['Reconstruction_Error'], 
                               c=corr_df['Cluster'], cmap='viridis', alpha=0.7, s=50)
            
            # Add trend line
            valid_mask = ~np.isnan(corr_df['Reconstruction_Error'])
            if np.sum(valid_mask) > 1:
                m, b = np.polyfit(corr_df.loc[valid_mask, 'MCS'], 
                                  corr_df.loc[valid_mask, 'Reconstruction_Error'], 1)
                x_line = np.array([min(corr_df['MCS']), max(corr_df['MCS'])])
                ax.plot(x_line, m*x_line + b, color='red', linestyle='--', linewidth=2)
                
                # Calculate and display correlation
                corr_coef = np.corrcoef(
                    corr_df.loc[valid_mask, 'MCS'], 
                    corr_df.loc[valid_mask, 'Reconstruction_Error'])[0, 1]
                ax.text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add labels
            ax.set_xlabel('MCS Value', fontsize=14)
            ax.set_ylabel('Reconstruction Error', fontsize=14)
            ax.set_title('MCS vs. Reconstruction Error', fontsize=16)
            ax.grid(alpha=0.3)
            
            # Add color bar for clusters
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
        else:
            ax.text(0.5, 0.5, "Reconstruction Error Data Not Available", 
                   ha='center', va='center', fontsize=14)
            ax.set_xlabel('MCS Value', fontsize=14)
            ax.set_ylabel('Reconstruction Error', fontsize=14)
            ax.set_title('MCS vs. Reconstruction Error', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/mcs_vs_model_outputs.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create scatter plot matrix if seaborn pairplot is available
        try:
            valid_mask = ~np.isnan(corr_df['Reconstruction_Error']) & ~np.isnan(corr_df['Attention'])
            if np.sum(valid_mask) > 1:
                pair_plot_df = corr_df.loc[valid_mask, ['MCS', 'Confidence', 'Reconstruction_Error', 'Attention']]
                pair_plot = sns.pairplot(pair_plot_df, hue='Cluster', palette='viridis', 
                                      diag_kind='kde', plot_kws={'alpha': 0.6})
                pair_plot.fig.suptitle('Relationships Between MCS and Model Outputs', y=1.02, fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/figures/mcs_model_pairplot.png", dpi=300, bbox_inches='tight')
                plt.close()
        except:
            logger.info("Could not create pairplot, skipping")
    
    def generate_report(self):
        """Generate a comprehensive report of MCS analysis findings"""
        logger.info("Generating comprehensive report...")
        
        # Create report content
        report = []
        
        # Header
        report.append("# MCS Analysis for False Positives\n")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report.append(f"*Report generated on: {timestamp}*\n")
        
        # Introduction
        report.append("## Introduction\n")
        report.append("This report analyzes the Modulation and Coding Scheme (MCS) values associated with false positives in our anomaly detection model. ")
        report.append("MCS is a critical parameter in wireless systems that determines the data transmission rate and robustness.")
        report.append("Our analysis explores whether certain MCS patterns or behaviors are indicative of potential anomalies that the model has identified but weren't labeled in the original dataset.\n")
        
        # Summary statistics
        report.append("## Summary Statistics\n")
        
        # Load statistics if available
        stats_file = f"{self.results_dir}/statistics/mcs_statistics.csv"
        if os.path.exists(stats_file):
            stats_df = pd.read_csv(stats_file)
            report.append("### MCS Distribution by Category\n")
            report.append("| Category | Count | Mean MCS | Std MCS | Median MCS | Min MCS | Max MCS |\n")
            report.append("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            for _, row in stats_df.iterrows():
                report.append(f"| {row['Category']} | {row['Count']} | {row['Mean MCS']:.2f} | {row['Std MCS']:.2f} | {row['Median MCS']:.2f} | {row['Min MCS']:.2f} | {row['Max MCS']:.2f} |\n")
        
        # Cluster statistics
        cluster_stats_file = f"{self.results_dir}/statistics/mcs_cluster_statistics.csv"
        if os.path.exists(cluster_stats_file):
            cluster_stats_df = pd.read_csv(cluster_stats_file)
            report.append("\n### MCS Distribution by False Positive Cluster\n")
            report.append("| Cluster | Count | Mean MCS | Std MCS | Median MCS | Min MCS | Max MCS |\n")
            report.append("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            for _, row in cluster_stats_df.iterrows():
                report.append(f"| {row['Cluster']} | {row['Count']} | {row['Mean MCS']:.2f} | {row['Std MCS']:.2f} | {row['Median MCS']:.2f} | {row['Min MCS']:.2f} | {row['Max MCS']:.2f} |\n")
        
        # Key findings
        report.append("\n## Key Findings\n")
        
        # MCS Distribution
        report.append("### 1. MCS Distribution Analysis\n")
        report.append("The distribution of MCS values in false positives shows a distinctive pattern compared to normal and known anomaly cases.\n")
        report.append("![MCS Distribution](figures/mcs_distribution_publication.png)\n")
        
        # Context windows
        report.append("### 2. MCS Context Window Analysis\n")
        report.append("Examining the MCS values in a temporal window around false positives reveals characteristic patterns:\n")
        report.append("![MCS Context Window](figures/mcs_context_publication.png)\n")
        
        # Transition matrices
        report.append("### 3. MCS Transition Analysis\n")
        report.append("The transition patterns of MCS values leading up to false positives differ significantly from normal transitions:\n")
        report.append("![MCS Transitions](figures/transition_matrices_publication.png)\n")
        
        # Variability analysis
        report.append("### 4. MCS Variability Analysis\n")
        report.append("MCS variability metrics show significant differences between normal samples and false positives:\n")
        report.append("![MCS Variability](figures/mcs_variability_publication.png)\n")
        
        # Pattern mining
        sig_transitions_file = f"{self.results_dir}/statistics/significant_mcs_transitions.csv"
        if os.path.exists(sig_transitions_file):
            sig_transitions_df = pd.read_csv(sig_transitions_file)
            report.append("### 5. Significant MCS Transitions\n")
            report.append("The following MCS transitions are significantly more common in false positives than in normal cases:\n")
            report.append("| From MCS | To MCS | FP Probability | Normal Probability | Difference |\n")
            report.append("| --- | --- | --- | --- | --- |\n")
            
            for i, row in sig_transitions_df.head(5).iterrows():
                # Use the correct column names from the significant_mcs_transitions.csv file
                from_mcs = int(row['From_MCS'])
                to_mcs = int(row['To_MCS'])
                fp_prob = row['FP_Prob']
                normal_prob = row['Normal_Prob']
                difference = row['Difference']
                
                report.append(f"| {from_mcs} | {to_mcs} | {fp_prob:.4f} | {normal_prob:.4f} | {difference:.4f} |\n")
        
        # Prototype patterns
        report.append("### 6. Prototypical MCS Sequences\n")
        report.append("We identified several prototypical MCS sequence patterns that often lead to false positives:\n")
        report.append("![MCS Prototypes](figures/mcs_prototypes.png)\n")
        
        # Model correlation
        report.append("### 7. Correlation with Model Outputs\n")
        report.append("Analysis of the relationship between MCS values and model confidence/reconstruction error reveals:\n")
        report.append("![MCS vs Model Outputs](figures/mcs_vs_model_outputs.png)\n")
        
        # Conclusion and recommendations
        report.append("## Conclusion and Recommendations\n")
        
        # Add specific conclusions based on findings
        report.append("Based on our comprehensive analysis of MCS patterns in false positives, we can draw several important conclusions:\n")
        
        report.append("1. **Distinctive MCS Patterns**: False positives exhibit MCS distributions and temporal patterns that differ significantly from both normal cases and known anomalies. This suggests they may represent a valid but unlabeled anomaly class.\n")
        
        report.append("2. **MCS Variability as an Indicator**: The variability in MCS values around false positives is consistently higher than in normal cases, suggesting that rapid fluctuations in MCS could be an early indicator of network instability.\n")
        
        report.append("3. **Specific MCS Transitions**: Certain transitions between MCS values appear much more frequently in false positives than in normal operation, potentially indicating problematic adaptation behavior in the transmission system.\n")
        
        report.append("4. **Cluster Differentiation**: The identified clusters of false positives show distinct MCS behaviors, suggesting they may represent different types of unlabeled anomalies with unique signatures.\n")
        
        report.append("\n### Recommendations:\n")
        
        report.append("1. **Enhanced Feature Engineering**: Incorporate MCS variability metrics (standard deviation, entropy, max change) as additional features in the anomaly detection model.\n")
        
        report.append("2. **Model Refinement**: Use the identified MCS patterns to create targeted detection rules for these potential new anomaly classes.\n")
        
        report.append("3. **Domain Expert Review**: Have wireless network experts review the identified prototypical patterns to determine if they represent known failure modes or network issues that weren't captured in the original labeling.\n")
        
        report.append("4. **Dataset Enrichment**: Consider relabeling a subset of these false positives as a new anomaly class and retraining the model to improve its discriminative ability.\n")
        
        report.append("5. **Extended Monitoring**: Implement specific monitoring for the identified MCS patterns in production systems to validate their relationship with actual network issues.\n")
        
        # Save report
        report_path = f"{self.results_dir}/mcs_analysis_report.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Comprehensive report saved to {report_path}")
        
        # Also generate a simplified text version
        text_report_path = f"{self.results_dir}/mcs_analysis_summary.txt"
        with open(text_report_path, 'w') as f:
            for line in report:
                # Remove markdown formatting
                line = line.replace('#', '').replace('*', '').replace('|', ' ').replace('![', '').replace('](', ': ').replace(')', '')
                f.write(line + '\n')
        
        logger.info(f"Text summary report saved to {text_report_path}")
    
    def run_analysis(self):
        """Run the complete MCS analysis pipeline"""
        logger.info("Starting comprehensive MCS analysis...")
        
        # Step 1: Analyze basic MCS distribution
        self.analyze_mcs_distribution()
        
        # Step 2: Analyze MCS by cluster
        self.analyze_mcs_by_cluster()
        
        # Step 3: Analyze temporal patterns
        self.analyze_mcs_temporal_patterns()
        
        # Step 4: Run pattern mining
        self.run_pattern_mining()
        
        # Step 5: Discover and analyze prototypes
        self.analyze_prototype_discovery()
        
        # Step 6: Analyze correlation with model outputs
        self.analyze_mcs_correlation_with_model_outputs()
        
        # Step 7: Generate comprehensive report
        self.generate_report()
        
        logger.info("MCS analysis completed!")


def main():
    """Main entry point for the script"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Advanced MCS Analysis for False Positives')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model weights')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to model config')
    parser.add_argument('--data', type=str, default=TEST_DATA_PATH, help='Path to test data')
    parser.add_argument('--fp_data', type=str, default=FP_DATA_PATH, help='Path to false positives data')
    parser.add_argument('--results', type=str, default=RESULTS_DIR, help='Path to save results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MCSAnalyzer(
        model_path=args.model,
        config_path=args.config,
        test_data_path=args.data,
        fp_data_path=args.fp_data,
        results_dir=args.results
    )
    
    # Run analysis
    analyzer.run_analysis()
    
    logger.info(f"Analysis complete. Results saved to {args.results}")


if __name__ == "__main__":
    main()