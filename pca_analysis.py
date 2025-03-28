import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

class PCAClusterAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, 'pca_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Feature names from the original analysis
        self.feature_names = ['SFN', 'Slot', 'HARQ', 'MCS', 'CRC', 'ReTx', 'NDI']
        self.class_names = ['Normal', 'Type 1', 'Type 2', 'Type 3', 'Type 4']
        
        # Set up plot style
        try:
            # Try the newer style name format
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            try:
                # Try the older style name
                plt.style.use('seaborn-whitegrid')
            except:
                # Fallback to a basic style that's available in all versions
                plt.style.use('ggplot')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def load_data(self):
        """Load the false positives data and PCA components"""
        print("Loading data...")
        
        # Load false positives data with features
        fp_path = os.path.join(self.results_dir, 'false_positives', 'false_positives_clustered.csv')
        if not os.path.exists(fp_path):
            print(f"Error: Could not find {fp_path}")
            return False
            
        self.fp_df = pd.read_csv(fp_path)
        
        # Load PCA components
        pca_path = os.path.join(self.results_dir, 'false_positives', 'pca_components.csv')
        if not os.path.exists(pca_path):
            print(f"Error: Could not find {pca_path}")
            return False
            
        self.pca_df = pd.read_csv(pca_path)
        
        # Load test data instances if available (for comparison)
        test_instances_path = os.path.join(self.results_dir, 'statistics', 'dataset_statistics.csv')
        if os.path.exists(test_instances_path):
            self.test_stats_df = pd.read_csv(test_instances_path)
        else:
            self.test_stats_df = None
            
        # Check if we have all necessary feature columns
        missing_features = [f for f in self.feature_names if f not in self.fp_df.columns]
        if missing_features:
            print(f"Warning: Missing features in data: {missing_features}")
            # Filter feature_names to only include available features
            self.feature_names = [f for f in self.feature_names if f in self.fp_df.columns]
            
        print(f"Loaded {len(self.fp_df)} false positive samples with {len(self.feature_names)} features")
        print(f"Feature names: {self.feature_names}")
        return True
        
    def perform_pca_analysis(self):
        """Perform PCA analysis on the false positives data"""
        print("Performing PCA analysis...")
        
        # Extract features
        X = self.fp_df[self.feature_names].values
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca = PCA()
        self.pca_result = self.pca.fit_transform(X_scaled)
        
        # Get explained variance
        self.explained_variance = self.pca.explained_variance_ratio_ * 100
        self.cumulative_variance = np.cumsum(self.explained_variance)
        
        # Get component loadings (correlation between original features and principal components)
        self.loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        
        # Get feature contributions to each PC (squared loadings)
        self.feature_contributions = self.loadings**2
        for i in range(len(self.feature_contributions)):
            self.feature_contributions[i] /= np.sum(self.feature_contributions[i])
            
        # Get clusters from loaded data
        self.clusters = self.fp_df['cluster'].values
        self.n_clusters = len(np.unique(self.clusters))
        
        print(f"PCA completed with {self.n_clusters} clusters")
        print(f"Explained variance by first 3 PCs: {self.explained_variance[:3].sum():.2f}%")
        return True
        
    def plot_explained_variance(self):
        """Plot explained variance by principal components"""
        plt.figure(figsize=(12, 6))
        
        # Bar plot for individual explained variance
        plt.bar(range(1, len(self.explained_variance) + 1), 
                self.explained_variance, 
                alpha=0.7, 
                label='Individual')
        
        # Line plot for cumulative explained variance
        plt.step(range(1, len(self.cumulative_variance) + 1), 
                 self.cumulative_variance, 
                 where='mid', 
                 label='Cumulative',
                 color='red')
        
        plt.axhline(y=95, color='k', linestyle='--', alpha=0.7, label='95% Threshold')
        
        # Find how many components are needed for 95% variance
        n_components_95 = np.argmax(self.cumulative_variance >= 95) + 1
        plt.axvline(x=n_components_95, color='g', linestyle='--', alpha=0.7, 
                   label=f'{n_components_95} PCs for 95% variance')
        
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, len(self.explained_variance) + 1))
        plt.legend()
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'explained_variance.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved explained variance plot to {output_path}")
        
        # Save explained variance data to CSV
        variance_df = pd.DataFrame({
            'Principal Component': range(1, len(self.explained_variance) + 1),
            'Individual Variance (%)': self.explained_variance,
            'Cumulative Variance (%)': self.cumulative_variance
        })
        csv_path = os.path.join(self.output_dir, 'explained_variance.csv')
        variance_df.to_csv(csv_path, index=False)
        print(f"Saved explained variance data to {csv_path}")
        
    def plot_feature_contributions(self):
        """Plot feature contributions to principal components"""
        # For the first 3 components
        n_components = min(3, len(self.feature_names))
        
        plt.figure(figsize=(14, 8))
        contribution_data = pd.DataFrame(
            self.feature_contributions[:, :n_components], 
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Plot as a heatmap
        sns.heatmap(contribution_data, annot=True, cmap='YlGnBu', fmt='.2f')
        plt.title('Feature Contributions to Principal Components')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'feature_contributions_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved feature contributions heatmap to {output_path}")
        
        # Plot as a bar chart for each PC
        for i in range(n_components):
            plt.figure(figsize=(12, 6))
            contrib = self.feature_contributions[:, i]
            y_pos = np.arange(len(self.feature_names))
            
            # Sort by contribution
            sorted_idx = np.argsort(contrib)
            
            plt.barh(y_pos, contrib[sorted_idx], align='center')
            plt.yticks(y_pos, [self.feature_names[idx] for idx in sorted_idx])
            plt.xlabel('Contribution')
            plt.title(f'Feature Contributions to PC{i+1} ({self.explained_variance[i]:.2f}% variance)')
            plt.tight_layout()
            
            output_path = os.path.join(self.output_dir, f'feature_contributions_pc{i+1}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved feature contributions for PC{i+1} to {output_path}")
            
        # Save feature contributions to CSV
        contrib_df = pd.DataFrame(
            self.feature_contributions,
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        csv_path = os.path.join(self.output_dir, 'feature_contributions.csv')
        contrib_df.to_csv(csv_path)
        print(f"Saved feature contributions data to {csv_path}")
        
    def plot_loadings(self):
        """Plot PCA loadings to show relationship between original features and PCs"""
        # For the first 3 components
        n_components = min(3, len(self.feature_names))
        
        # Create loading plot
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
        
        if n_components == 1:
            axes = [axes]
            
        for i in range(n_components):
            # Sort loadings by absolute value
            loadings = self.loadings[:, i]
            sorted_idx = np.argsort(np.abs(loadings))[::-1]
            
            # Create bar chart
            bars = axes[i].barh(np.arange(len(self.feature_names)), 
                               loadings[sorted_idx], 
                               color=['b' if x > 0 else 'r' for x in loadings[sorted_idx]])
            
            axes[i].set_yticks(np.arange(len(self.feature_names)))
            axes[i].set_yticklabels([self.feature_names[j] for j in sorted_idx])
            axes[i].set_xlabel('Loading value')
            axes[i].set_title(f'Feature Loadings on PC{i+1} ({self.explained_variance[i]:.2f}% variance)')
            
            # Add value labels to bars
            for bar in bars:
                width = bar.get_width()
                label_pos = width + 0.01 if width > 0 else width - 0.01
                alignment = 'left' if width > 0 else 'right'
                axes[i].text(label_pos, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}', va='center', ha=alignment)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'pca_loadings.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved PCA loadings plot to {output_path}")
        
        # Save loadings to CSV
        loadings_df = pd.DataFrame(
            self.loadings,
            index=self.feature_names,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
        )
        csv_path = os.path.join(self.output_dir, 'pca_loadings.csv')
        loadings_df.to_csv(csv_path)
        print(f"Saved PCA loadings data to {csv_path}")
        
    def plot_biplot(self):
        """Create a biplot to show both data points and feature loadings using existing PCA components"""
        # Check if we have PCA components already loaded
        if not hasattr(self, 'pca_df') or self.pca_df is None:
            print("No PCA components found in loaded data")
            return

        # Use the PCA components directly from the loaded file
        pc_pairs = [(0, 1), (0, 2), (1, 2)]
        
        for pc_i, pc_j in pc_pairs:
            # Skip if we don't have enough components
            if pc_j >= self.pca_df.shape[1] - 1:  # -1 to account for cluster column
                continue
                
            fig, ax = plt.subplots(figsize=(14, 10))
            
            # Plot the data points colored by cluster using the loaded values
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.pca_df['cluster'] == cluster_id
                ax.scatter(
                    self.pca_df.loc[cluster_mask, f'PC{pc_i+1}'],
                    self.pca_df.loc[cluster_mask, f'PC{pc_j+1}'],
                    alpha=0.7,
                    label=f'Cluster {cluster_id}'
                )
            
            # Calculate cluster centers
            centers = []
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.pca_df['cluster'] == cluster_id
                center_x = self.pca_df.loc[cluster_mask, f'PC{pc_i+1}'].mean()
                center_y = self.pca_df.loc[cluster_mask, f'PC{pc_j+1}'].mean()
                centers.append((center_x, center_y))
                
                # Plot cluster center
                ax.scatter(
                    center_x, center_y,
                    marker='X',
                    s=200,
                    c='red',
                    edgecolors='black',
                    label=f'Cluster {cluster_id} Center' if cluster_id == 0 else None
                )
                
                # Add annotation for cluster
                ax.annotate(
                    f"Cluster {cluster_id}\nUnknown Anomaly Pattern",
                    xy=(center_x, center_y),
                    xytext=(10, -20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Plot feature vectors (from original loadings)
            # We need to scale them appropriately for visibility
            if hasattr(self, 'loadings'):
                multiplier = 5  # Scale factor to make arrows visible
                for i, feature in enumerate(self.feature_names):
                    ax.arrow(0, 0, 
                            self.loadings[i, pc_i] * multiplier, 
                            self.loadings[i, pc_j] * multiplier, 
                            head_width=0.1, head_length=0.1, fc='k', ec='k')
                    ax.text(self.loadings[i, pc_i] * multiplier * 1.15, 
                            self.loadings[i, pc_j] * multiplier * 1.15, 
                            feature, color='k', ha='center', va='center')
            
            # Set labels and title
            pc1_variance = self.explained_variance[pc_i] if hasattr(self, 'explained_variance') else 28.31  # Default from your image
            pc2_variance = self.explained_variance[pc_j] if hasattr(self, 'explained_variance') else 15.05  # Default from your image
            
            ax.set_xlabel(f'PC{pc_i+1} ({pc1_variance:.2f}% variance)')
            ax.set_ylabel(f'PC{pc_j+1} ({pc2_variance:.2f}% variance)')
            ax.set_title(f'PCA Visualization (PC{pc_i+1} vs PC{pc_j+1})')
            
            # Add grid and legend
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Save figure
            output_path = os.path.join(self.output_dir, f'pca_viz_pc{pc_i+1}_pc{pc_j+1}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved PCA visualization of PC{pc_i+1} vs PC{pc_j+1} to {output_path}")
            
    def compare_cluster_features(self):
        """Compare feature distributions across clusters"""
        # Add known anomalies and normal events data from main dataset if available
        # to compare with false positive clusters
        
        # Create violin plots for each feature
        for feature in self.feature_names:
            plt.figure(figsize=(14, 8))
            
            # Create DataFrame for plotting
            plot_data = []
            
            # Add data for each cluster
            for cluster_id in range(self.n_clusters):
                cluster_data = self.fp_df[self.fp_df['cluster'] == cluster_id][feature]
                cluster_df = pd.DataFrame({
                    'Feature Value': cluster_data.values,
                    'Category': [f'Cluster {cluster_id}'] * len(cluster_data)
                })
                plot_data.append(cluster_df)
            
            # Combine all data
            plot_df = pd.concat(plot_data, ignore_index=True)
            
            # Create violin plot
            ax = sns.violinplot(x='Category', y='Feature Value', data=plot_df)
            
            # Add individual observations with jittered points
            if len(plot_df) < 1000:  # Only for smaller datasets to avoid overcrowding
                sns.stripplot(x='Category', y='Feature Value', data=plot_df, 
                            color='black', size=2, alpha=0.3)
            
            # Calculate and add mean and median labels
            for i, category in enumerate(plot_df['Category'].unique()):
                category_data = plot_df[plot_df['Category'] == category]['Feature Value']
                mean_val = category_data.mean()
                median_val = category_data.median()
                plt.text(i, plot_df['Feature Value'].max(), 
                        f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}', 
                        ha='center', va='bottom')
            
            # Add statistical test results
            if self.n_clusters >= 2:
                # Perform statistical test between clusters
                cluster_0_data = self.fp_df[self.fp_df['cluster'] == 0][feature].values
                cluster_1_data = self.fp_df[self.fp_df['cluster'] == 1][feature].values
                
                try:
                    # Try t-test first
                    t_stat, p_value = stats.ttest_ind(cluster_0_data, cluster_1_data, equal_var=False)
                    test_name = "Welch's t-test"
                except:
                    # If t-test fails, use Mann-Whitney U test
                    t_stat, p_value = stats.mannwhitneyu(cluster_0_data, cluster_1_data)
                    test_name = "Mann-Whitney U test"
                
                # Add p-value annotation
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                plt.annotate(f'{test_name}: {significance} (p={p_value:.4f})', 
                           xy=(0.5, 0.01), xycoords='figure fraction',
                           ha='center', va='bottom', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            # Set labels and title
            plt.xlabel('Category', fontsize=12)
            plt.ylabel(f'{feature} Value', fontsize=12)
            plt.title(f'Distribution of {feature} Values by Cluster', fontsize=14)
            
            # Save figure
            output_path = os.path.join(self.output_dir, f'feature_distribution_{feature}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved distribution plot for {feature} to {output_path}")
            
        # Create a summary table of feature statistics by cluster
        stats_rows = []
        
        for feature in self.feature_names:
            for cluster_id in range(self.n_clusters):
                cluster_data = self.fp_df[self.fp_df['cluster'] == cluster_id][feature]
                
                stats_rows.append({
                    'Feature': feature,
                    'Cluster': f'Cluster {cluster_id}',
                    'Mean': cluster_data.mean(),
                    'Median': cluster_data.median(),
                    'Std Dev': cluster_data.std(),
                    'Min': cluster_data.min(),
                    'Max': cluster_data.max(),
                    'Count': len(cluster_data)
                })
        
        # Create DataFrame from statistics
        stats_df = pd.DataFrame(stats_rows)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'cluster_feature_statistics.csv')
        stats_df.to_csv(csv_path, index=False)
        print(f"Saved cluster feature statistics to {csv_path}")
        
    def calculate_cluster_discriminative_features(self):
        """Calculate which features are most discriminative between clusters"""
        if self.n_clusters < 2:
            print("Need at least 2 clusters to calculate discriminative features")
            return
            
        # Calculate effect size (Cohen's d) for each feature between clusters
        effect_sizes = []
        
        for feature in self.feature_names:
            # Get data for each cluster
            cluster_0_data = self.fp_df[self.fp_df['cluster'] == 0][feature].values
            cluster_1_data = self.fp_df[self.fp_df['cluster'] == 1][feature].values
            
            # Calculate mean and standard deviation
            mean_0 = np.mean(cluster_0_data)
            mean_1 = np.mean(cluster_1_data)
            
            # Pooled standard deviation
            n_0 = len(cluster_0_data)
            n_1 = len(cluster_1_data)
            
            # Handle case of zero variance
            std_0 = np.std(cluster_0_data) if np.std(cluster_0_data) > 0 else 1e-10
            std_1 = np.std(cluster_1_data) if np.std(cluster_1_data) > 0 else 1e-10
            
            pooled_std = np.sqrt(((n_0 - 1) * std_0**2 + (n_1 - 1) * std_1**2) / (n_0 + n_1 - 2))
            
            # Calculate Cohen's d
            cohen_d = abs(mean_0 - mean_1) / pooled_std if pooled_std > 0 else 0
            
            # Calculate t-statistic and p-value
            try:
                t_stat, p_value = stats.ttest_ind(cluster_0_data, cluster_1_data, equal_var=False)
            except:
                # If t-test fails, use non-parametric test
                try:
                    t_stat, p_value = stats.mannwhitneyu(cluster_0_data, cluster_1_data)
                except:
                    t_stat, p_value = 0, 1
            
            effect_sizes.append({
                'Feature': feature,
                'Effect Size (Cohen\'s d)': cohen_d,
                'Mean Cluster 0': mean_0,
                'Mean Cluster 1': mean_1,
                'Std Cluster 0': std_0,
                'Std Cluster 1': std_1,
                't-statistic': t_stat,
                'p-value': p_value,
                'Significant': p_value < 0.05
            })
        
        # Create DataFrame
        effect_df = pd.DataFrame(effect_sizes)
        
        # Sort by effect size
        effect_df = effect_df.sort_values('Effect Size (Cohen\'s d)', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, 'discriminative_features.csv')
        effect_df.to_csv(csv_path, index=False)
        print(f"Saved discriminative features analysis to {csv_path}")
        
        # Plot effect sizes
        plt.figure(figsize=(12, 8))
        
        # Sort by effect size
        effect_df = effect_df.sort_values('Effect Size (Cohen\'s d)')
        
        # Create bar chart
        bars = plt.barh(effect_df['Feature'], effect_df['Effect Size (Cohen\'s d)'])
        
        # Color bars by significance
        for i, bar in enumerate(bars):
            if effect_df.iloc[i]['Significant']:
                bar.set_color('darkblue')
            else:
                bar.set_color('lightblue')
        
        # Add labels
        for i, v in enumerate(effect_df['Effect Size (Cohen\'s d)']):
            plt.text(v + 0.05, i, f'{v:.2f}', va='center')
        
        # Add a legend for significance
        plt.scatter([], [], color='darkblue', label='Significant (p < 0.05)')
        plt.scatter([], [], color='lightblue', label='Not Significant')
        plt.legend(loc='lower right')
        
        plt.xlabel('Effect Size (Cohen\'s d)')
        plt.title('Discriminative Power of Features Between Clusters')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, 'discriminative_power.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved discriminative power plot to {output_path}")
        
        return effect_df
    
    def plot_3d_pca(self):
        """Re-create the 3D PCA visualization with additional annotations"""
        # Check if we have at least 3 principal components
        if self.pca.n_components_ < 3:
            print("Need at least 3 principal components for 3D visualization")
            return
            
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each cluster with different color
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.clusters == cluster_id
            ax.scatter(
                self.pca_result[cluster_mask, 0],
                self.pca_result[cluster_mask, 1],
                self.pca_result[cluster_mask, 2],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )
        
        # Calculate cluster centers and add them to the plot
        cluster_centers = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.clusters == cluster_id
            center = np.mean(self.pca_result[cluster_mask, :3], axis=0)
            cluster_centers.append(center)
            
            # Plot cluster center
            ax.scatter(
                center[0], center[1], center[2],
                marker='X',
                s=200,
                c='red',
                edgecolors='black',
                label=f'Cluster {cluster_id} Center' if cluster_id == 0 else None
            )
        
        # Add annotations for top discriminative features (if we calculated them)
        try:
            effect_df = self.calculate_cluster_discriminative_features()
            top_features = effect_df.head(3)['Feature'].values
            
            feature_text = "Top discriminative features:\n"
            for i, feature in enumerate(top_features):
                feature_text += f"{i+1}. {feature} (d={effect_df[effect_df['Feature'] == feature]['Effect Size (Cohen\'s d)'].values[0]:.2f})\n"
                
            # Add as text in the corner
            ax.text2D(0.05, 0.95, feature_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except:
            pass
        
        # Set axis labels with explained variance
        ax.set_xlabel(f'PC1 ({self.explained_variance[0]:.2f}% variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.explained_variance[1]:.2f}% variance)', fontsize=12)
        ax.set_zlabel(f'PC3 ({self.explained_variance[2]:.2f}% variance)', fontsize=12)
        
        # Add legend (without duplicate entries)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')
        
        ax.set_title('Enhanced 3D PCA Visualization of False Positive Clusters', fontsize=14)
        
        # Save the figure
        output_path = os.path.join(self.output_dir, 'enhanced_3d_pca.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved enhanced 3D PCA visualization to {output_path}")
        
    def generate_summary_report(self):
        """Generate a summary report of the analysis"""
        # Create a markdown summary
        report = [
            "# False Positive Cluster Analysis Summary",
            "\n## PCA Analysis",
            f"- Number of clusters identified: {self.n_clusters}",
            f"- Explained variance by first 3 PCs: {self.explained_variance[:3].sum():.2f}%",
            f"- PC1 explains {self.explained_variance[0]:.2f}% of variance",
            f"- PC2 explains {self.explained_variance[1]:.2f}% of variance",
            f"- PC3 explains {self.explained_variance[2]:.2f}% of variance"
        ]
        
        # Add information about cluster sizes
        report.append("\n## Cluster Sizes")
        for cluster_id in range(self.n_clusters):
            cluster_count = np.sum(self.clusters == cluster_id)
            cluster_percent = 100 * cluster_count / len(self.clusters)
            report.append(f"- Cluster {cluster_id}: {cluster_count} samples ({cluster_percent:.2f}%)")
        
        # Add feature contribution information
        report.append("\n## Feature Contributions to Principal Components")
        
        # Top 3 contributing features to each PC
        for i in range(min(3, self.pca.n_components_)):
            contributions = self.feature_contributions[:, i]
            top_indices = np.argsort(contributions)[-3:][::-1]
            
            report.append(f"\n### PC{i+1} ({self.explained_variance[i]:.2f}% variance)")
            report.append("Top contributing features:")
            
            for j, idx in enumerate(top_indices):
                feature = self.feature_names[idx]
                contrib = contributions[idx] * 100
                report.append(f"- {j+1}. {feature}: {contrib:.2f}%")
        
        # Add discriminative feature information if we have at least 2 clusters
        if self.n_clusters >= 2:
            try:
                effect_df = pd.read_csv(os.path.join(self.output_dir, 'discriminative_features.csv'))
                
                report.append("\n## Discriminative Features Between Clusters")
                report.append("Features that best separate the clusters (ordered by effect size):")
                
                for i, row in effect_df.head(5).iterrows():
                    feature = row['Feature']
                    effect = row['Effect Size (Cohen\'s d)']
                    p_value = row['p-value']
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    
                    mean_0 = row['Mean Cluster 0']
                    mean_1 = row['Mean Cluster 1']
                    
                    report.append(f"- {feature}: effect size = {effect:.2f} {significance}")
                    report.append(f"  - Mean in Cluster 0: {mean_0:.2f}")
                    report.append(f"  - Mean in Cluster 1: {mean_1:.2f}")
            except:
                pass
                
        # Add cluster interpretation
        report.append("\n## Cluster Interpretation")
        report.append("Based on the analysis, the following interpretations can be made:")
        
        try:
            # Try to interpret clusters based on their feature values
            effect_df = pd.read_csv(os.path.join(self.output_dir, 'discriminative_features.csv'))
            stats_df = pd.read_csv(os.path.join(self.output_dir, 'cluster_feature_statistics.csv'))
            
            # For each cluster, find distinctive characteristics
            for cluster_id in range(self.n_clusters):
                report.append(f"\n### Cluster {cluster_id}")
                
                # Filter statistics for this cluster
                cluster_stats = stats_df[stats_df['Cluster'] == f'Cluster {cluster_id}']
                
                # Get top 3 discriminative features
                top_features = effect_df.head(3)['Feature'].values
                
                characteristics = []
                
                for feature in top_features:
                    # Get statistics for this feature in this cluster
                    feature_stats = cluster_stats[cluster_stats['Feature'] == feature]
                    if len(feature_stats) == 0:
                        continue
                        
                    # Get mean value
                    mean_value = feature_stats['Mean'].values[0]
                    
                    # Get mean value in other cluster for comparison
                    other_cluster = f'Cluster {1-cluster_id}' if self.n_clusters == 2 else None
                    other_stats = stats_df[(stats_df['Cluster'] == other_cluster) & 
                                         (stats_df['Feature'] == feature)]
                    
                    if len(other_stats) > 0:
                        other_mean = other_stats['Mean'].values[0]
                        compare_text = f"higher than" if mean_value > other_mean else f"lower than"
                        characteristics.append(f"{feature} tends to be {compare_text} in other cluster(s) ({mean_value:.2f} vs {other_mean:.2f})")
                    else:
                        characteristics.append(f"{feature} has mean value of {mean_value:.2f}")
                
                # Add characteristics to report
                if characteristics:
                    report.append("Distinctive characteristics:")
                    for char in characteristics:
                        report.append(f"- {char}")
                        
                # Try to interpret in terms of telecommunications domain
                if 'ReTx' in top_features and 'CRC' in top_features:
                    # Interpret based on ReTx and CRC patterns
                    retx_stats = cluster_stats[cluster_stats['Feature'] == 'ReTx']
                    crc_stats = cluster_stats[cluster_stats['Feature'] == 'CRC']
                    
                    if len(retx_stats) > 0 and len(crc_stats) > 0:
                        retx_mean = retx_stats['Mean'].values[0]
                        crc_mean = crc_stats['Mean'].values[0]
                        
                        # Interpret the pattern
                        if retx_mean > 2 and crc_mean < 0.5:
                            report.append("\nPossible interpretation: This cluster may represent unnecessary retransmissions where retransmission count is high despite successful CRC checks.")
                        elif crc_mean > 0.5 and retx_mean < 1:
                            report.append("\nPossible interpretation: This cluster may represent missing retransmissions where CRC checks failed but retransmissions weren't triggered.")
                        elif 'NDI' in top_features:
                            ndi_stats = cluster_stats[cluster_stats['Feature'] == 'NDI']
                            if len(ndi_stats) > 0:
                                ndi_mean = ndi_stats['Mean'].values[0]
                                if ndi_mean > 0.5 and crc_mean > 0.5 and retx_mean < 1:
                                    report.append("\nPossible interpretation: This cluster may represent cases where new data was sent while a retransmission was pending.")
                
        except Exception as e:
            report.append(f"Could not generate detailed interpretation: {str(e)}")
            
        # Add conclusion
        report.append("\n## Conclusion")
        report.append("The analysis of false positive clusters reveals distinct patterns that the model identified as anomalous despite being labeled as normal in the dataset. These patterns likely represent edge cases that exhibit characteristics similar to known anomalies but weren't explicitly labeled as such in the original data.")
        report.append("\nThis suggests that the model may be identifying genuine anomalies that were missed during the labeling process, rather than simply making classification errors. Further investigation with domain experts could help determine if these false positives should be relabeled as true anomalies in future iterations of the model.")
        
        # Save the report
        report_path = os.path.join(self.output_dir, 'analysis_summary.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"Saved summary report to {report_path}")
        
        # Also save as text
        txt_path = os.path.join(self.output_dir, 'analysis_summary.txt')
        with open(txt_path, 'w') as f:
            f.write('\n'.join(report))
            
        return report
        
    def run_analysis(self):
        """Run the complete PCA cluster analysis pipeline"""
        if not self.load_data():
            print("Failed to load data. Aborting analysis.")
            return False
            
        self.perform_pca_analysis()
        self.plot_explained_variance()
        self.plot_feature_contributions()
        self.plot_loadings()
        self.plot_biplot()
        self.compare_cluster_features()
        self.calculate_cluster_discriminative_features()
        self.plot_3d_pca()
        self.generate_summary_report()
        
        print(f"Analysis complete. Results saved to {self.output_dir}")
        return True


def main():
    """Main entry point for the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PCA separation in false positive clusters')
    parser.add_argument('--results_dir', type=str, default='analysis_results', 
                      help='Directory containing analysis results from inference.py')
    
    args = parser.parse_args()
    
    analyzer = PCAClusterAnalyzer(args.results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()