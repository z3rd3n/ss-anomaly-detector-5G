import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Import your existing modules - modify paths as needed
from dataset import create_data_loaders
from trainer import BiGRUAnomalyDetector, setup_logger

# Set up paths
MODEL_PATH = "best_model.pth"
CONFIG_PATH = MODEL_PATH  # Use the same file for both
TEST_DATA_PATH = "unscaled_pdsch_val_min.parquet"
RESULTS_DIR = "feature_analysis_results"
FP_DATA_PATH = "analysis_results/false_positives/false_positives_clustered.csv"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

# Configure logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

class FeatureAnalyzer:
    def __init__(self, model_path, config_path, test_data_path, fp_data_path, results_dir):
        self.model_path = model_path
        self.config_path = config_path
        self.test_data_path = test_data_path
        self.fp_data_path = fp_data_path
        self.results_dir = results_dir
        self.device = device
        self.feature_names = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        self.class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        
        # Load model and data
        self.load_config()
        self.load_model()
        self.load_test_data()
        self.load_false_positives()  # Load FP data after test data
        self.extract_fp_sequences()  # Extract FP sequences after loading both datasets
        
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
        
        # Extract raw data for feature analysis
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
        
        # Create mask for normal vs anomaly samples
        self.normal_mask = (self.all_labels == 0)
        self.anomaly_mask = (self.all_labels > 0)
        
        logger.info(f"Extracted {len(self.all_labels)} timesteps of feature data")

    def load_false_positives(self):
        """Load false positives data from CSV"""
        logger.info(f"Loading false positives from: {self.fp_data_path}")
        
        self.fp_df = pd.read_csv(self.fp_data_path)
        logger.info(f"Loaded {len(self.fp_df)} false positives")
        
        # Split false positives by cluster
        self.clusters = self.fp_df['cluster'].unique()
        self.fp_by_cluster = {c: self.fp_df[self.fp_df['cluster'] == c] for c in self.clusters}
        
        logger.info(f"Found {len(self.clusters)} clusters of false positives")
    
    def extract_fp_sequences(self):
        """Extract sequences containing false positives for context window analysis"""
        logger.info("Extracting sequences containing false positives...")
        
        # Store timestamps from false positives to identify them in sequences
        fp_timestamps = set(self.fp_df['timestamp'].values)
        
        # Store sequences containing false positives
        self.fp_sequences = []
        
        # Process each sequence in test dataset
        for i in range(len(self.test_loader.dataset)):
            sample = self.test_loader.dataset[i]
            features = sample['feature_data'].numpy()
            labels = sample['label'].numpy()
            timestamps = sample['timestamp']
            
            # Check if this sequence contains any false positives
            fp_positions = []
            for j, ts in enumerate(timestamps):
                if ts in fp_timestamps:
                    fp_positions.append(j)
            
            if fp_positions:
                # Store sequence data
                seq_data = {
                    'features': features,
                    'labels': labels,
                    'timestamps': timestamps,
                    'fp_positions': fp_positions
                }
                self.fp_sequences.append(seq_data)
        
        logger.info(f"Found {len(self.fp_sequences)} sequences containing false positives")

    def create_violin_plots(self):
        """Create violin plots for all features by category"""
        logger.info("Creating violin plots for all features...")
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            logger.info(f"Creating violin plot for {feature_name}...")
            
            # Extract feature values
            normal_data = self.all_features[self.normal_mask, feature_idx]
            anomaly_data = self.all_features[self.anomaly_mask, feature_idx]
            
            # Extract feature values for each cluster
            cluster_data = {}
            for cluster in sorted(self.clusters):
                cluster_df = self.fp_by_cluster[cluster]
                cluster_data[cluster] = cluster_df[feature_name].values
            
            # Create violin plot
            plt.figure(figsize=(16, 10))
            
            # Prepare data for plotting
            categories = ['Normal', 'Known Anomaly'] + [f'Cluster {c}' for c in sorted(cluster_data.keys())]
            data_list = [normal_data, anomaly_data] + [cluster_data[c] for c in sorted(cluster_data.keys())]
            
            # Create violin plot
            parts = plt.violinplot(data_list, showmeans=True, showextrema=True)
            
            # Customize violin plots
            colors = ['blue', 'darkcyan', 'green', 'lightgreen']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            # Add box plots for clearer visualization
            plt.boxplot(data_list, positions=range(1, len(categories) + 1), 
                      widths=0.15, patch_artist=False, showfliers=True)
            
            # Add individual points for small datasets
            for i, data in enumerate(data_list):
                if len(data) < 50 or feature_name in ['CRC', 'NDI']:  # For binary features or small datasets
                    plt.scatter([i + 1] * len(data), data, 
                               alpha=0.3, color='black', s=10)
            
            # Add normal mean and median reference lines
            normal_mean = np.mean(normal_data)
            normal_median = np.median(normal_data)
            plt.axhline(normal_mean, color='green', linestyle='--', 
                       label=f'Normal Mean: {normal_mean:.2f}')
            plt.axhline(normal_median, color='green', linestyle=':', 
                       label=f'Normal Median: {normal_median:.2f}')
            
            # Add statistical test results
            for i, category in enumerate(categories[1:], 1):
                # Perform statistical test
                stat, p_value = stats.mannwhitneyu(normal_data, data_list[i])
                
            # Set labels and title
            plt.xlabel('Category', fontsize=14)
            plt.ylabel(f'{feature_name} Value', fontsize=14)
            plt.title(f'{feature_name} Distribution by Category', fontsize=16)
            plt.xticks(range(1, len(categories) + 1), categories)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/figures/{feature_name}_violin_comparison.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_context_window_plots(self):
        """Create context window plots for all features around false positives"""
        logger.info("Creating context window plots for all features...")
        
        # Skip if no false positive sequences are available
        if not hasattr(self, 'fp_sequences') or not self.fp_sequences:
            logger.warning("No false positive sequences available for context window analysis.")
            return
        
        # Context window size (before and after the false positive)
        window_size = 20
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            logger.info(f"Creating context window plot for {feature_name}...")
            
            # Collect feature values around false positives
            fp_context_windows = []
            cluster_labels = []
            
            for seq in self.fp_sequences:
                features = seq['features']
                feature_values = features[:, feature_idx]
                
                for fp_pos in seq['fp_positions']:
                    # Get window around the false positive
                    start = max(0, fp_pos - window_size)
                    end = min(len(feature_values), fp_pos + window_size + 1)
                    
                    # Create context window
                    window = np.full(2*window_size + 1, np.nan)  # Initialize with NaN
                    window_slice = slice(window_size - (fp_pos - start), window_size + (end - fp_pos))
                    window[window_slice] = feature_values[start:end]
                    
                    # Find the cluster for this false positive
                    timestamp = seq['timestamps'][fp_pos]
                    cluster = None
                    
                    # Find the cluster for this timestamp
                    for c in self.clusters:
                        if timestamp in self.fp_by_cluster[c]['timestamp'].values:
                            cluster = c
                            break
                    
                    # Store window and cluster
                    fp_context_windows.append(window)
                    cluster_labels.append(cluster)
            
            # Convert to numpy arrays
            fp_context_windows = np.array(fp_context_windows)
            cluster_labels = np.array(cluster_labels)
            
            # Create windows DataFrame for easier analysis
            window_columns = [f'{feature_name}_t{i-window_size}' for i in range(2*window_size + 1)]
            window_df = pd.DataFrame(fp_context_windows, columns=window_columns)
            window_df['cluster'] = cluster_labels
            
            # Calculate average window by cluster
            cluster_avg_windows = {}
            for cluster in self.clusters:
                cluster_mask = window_df['cluster'] == cluster
                if np.sum(cluster_mask) > 0:
                    cluster_windows = window_df.loc[cluster_mask, window_columns].values
                    avg_window = np.nanmean(cluster_windows, axis=0)
                    std_window = np.nanstd(cluster_windows, axis=0)
                    count = np.sum(cluster_mask)
                    cluster_avg_windows[cluster] = {
                        'mean': avg_window,
                        'std': std_window,
                        'count': count
                    }
            
            # Calculate normal reference pattern
            # Sample random positions from normal sequences
            normal_windows = []
            
            # Extract normal sequences
            normal_sequences = []
            for i in range(len(self.test_loader.dataset)):
                sample = self.test_loader.dataset[i]
                labels = sample['label'].numpy()
                features = sample['feature_data'].numpy()
                
                # Only use sequences with only normal events
                if np.all(labels == 0):
                    normal_sequences.append(features)
            
            # Sample windows from normal sequences
            for seq_features in normal_sequences:
                feature_values = seq_features[:, feature_idx]
                
                # Sample multiple windows from this sequence
                if len(feature_values) > 2*window_size:
                    # Sample up to 5 windows
                    for _ in range(min(5, len(feature_values) - 2*window_size)):
                        center_pos = np.random.randint(window_size, len(feature_values) - window_size)
                        window = feature_values[center_pos - window_size : center_pos + window_size + 1]
                        if len(window) == 2*window_size + 1:
                            normal_windows.append(window)
            
            # Convert to numpy array
            normal_windows = np.array(normal_windows)
            
            # Calculate average normal window
            normal_avg_window = np.mean(normal_windows, axis=0)
            normal_std_window = np.std(normal_windows, axis=0)
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1]})
            
            # Upper plot: Feature patterns around false positives
            ax = axes[0]
            
            # Plot normal reference
            x = np.arange(-window_size, window_size+1)
            ax.plot(x, normal_avg_window, 'g-', linewidth=2, label='Normal Reference')
            ax.fill_between(x, normal_avg_window - normal_std_window, 
                           normal_avg_window + normal_std_window, color='green', alpha=0.2)
            
            # Plot cluster patterns
            colors = ['blue', 'orange']
            for i, cluster in enumerate(sorted(cluster_avg_windows.keys())):
                stats = cluster_avg_windows[cluster]
                color = colors[i % len(colors)]
                mean = stats['mean']
                std = stats['std']
                count = stats['count']
                
                ax.plot(x, mean, 'o-', color=color, linewidth=2, 
                       label=f'Cluster {cluster} (n={count})')
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
            
            # Add vertical line at false positive position
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                      label='False Positive Position')
            
            # Set labels and title
            ax.set_xlabel('Relative Position', fontsize=14)
            ax.set_ylabel(f'{feature_name} Value', fontsize=14)
            ax.set_title(f'{feature_name} Patterns Around False Positives', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            
            # Lower plot: Feature change rate (derivative)
            ax = axes[1]
            
            # Calculate derivative for normal reference
            normal_derivative = np.diff(normal_avg_window)
            ax.plot(x[:-1] + 0.5, normal_derivative, 'g-', linewidth=2, 
                   label='Normal Reference')
            
            # Calculate derivatives for each cluster
            for i, cluster in enumerate(sorted(cluster_avg_windows.keys())):
                stats = cluster_avg_windows[cluster]
                color = colors[i % len(colors)]
                mean = stats['mean']
                
                # Calculate derivative
                derivative = np.diff(mean)
                
                # Plot derivative
                ax.plot(x[:-1] + 0.5, derivative, 'o-', color=color, linewidth=2, 
                       label=f'Cluster {cluster}')
            
            # Add vertical line at false positive position
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            
            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Set labels and title
            ax.set_xlabel('Relative Position', fontsize=14)
            ax.set_ylabel(f'{feature_name} Change Rate', fontsize=14)
            ax.set_title(f'Rate of Change in {feature_name} Around False Positives', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f"{self.results_dir}/figures/{feature_name}_context_publication.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_analysis(self):
        """Run the feature analysis"""
        logger.info("Starting feature analysis...")
        
        # Create violin plots for all features
        self.create_violin_plots()
        
        # Create context window plots for all features
        #self.create_context_window_plots()
        
        logger.info(f"Analysis complete. Results saved to {self.results_dir}")


def main():
    """Main entry point for the script"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Feature Analysis by Cluster')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to model weights')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to model config')
    parser.add_argument('--data', type=str, default=TEST_DATA_PATH, help='Path to test data')
    parser.add_argument('--fp_data', type=str, default=FP_DATA_PATH, help='Path to false positives data')
    parser.add_argument('--results', type=str, default=RESULTS_DIR, help='Path to save results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = FeatureAnalyzer(
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