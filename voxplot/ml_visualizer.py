#!/usr/bin/env python3
"""
ML Visualization Module for VoxPlot
===================================
Advanced visualizations for machine learning analysis of 3D forest structure data.
Creates publication-quality figures for clustering, dimensionality reduction,
spatial patterns, and comparative model analysis.

This module provides:
- 3D clustering visualizations
- Dimensionality reduction plots (PCA, t-SNE, UMAP)
- Spatial pattern heatmaps and distribution plots
- Occlusion analysis visualizations
- Comprehensive ML dashboards
- Physical accuracy assessment plots

Author: Joshua B-L & Claude Code
Date: 2025-09-10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Additional visualization libraries
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore', category=UserWarning)

def safe_scalar_extract(value, default=0):
    """
    Safely extract a scalar value from potentially array-like inputs.
    
    Args:
        value: Input value (scalar, array, or array-like)
        default: Default value to return if extraction fails
        
    Returns:
        Scalar numeric value
    """
    try:
        if hasattr(value, '__len__') and len(value) == 1:
            # Single-element array
            return float(value[0]) if hasattr(value[0], '__float__') else float(value)
        elif hasattr(value, '__len__') and len(value) > 1:
            # Multi-element array - return first element or mean
            return float(value[0]) if hasattr(value[0], '__float__') else default
        elif hasattr(value, '__float__'):
            # Scalar numeric value
            return float(value)
        else:
            # Non-numeric value
            return default
    except (ValueError, TypeError, IndexError):
        return default


class MLVisualizer:
    """
    Advanced ML visualizations for forest structure analysis.
    
    Creates publication-quality figures for:
    - Clustering analysis results
    - Dimensionality reduction visualizations
    - Spatial pattern analysis
    - Model comparison dashboards
    - Physical accuracy assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ML visualizer with Nature journal styling."""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self._setup_publication_style()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default visualization configuration."""
        return {
            'figure_settings': {
                'dpi': 600,
                'figsize_large': [20, 24],
                'figsize_medium': [16, 12],
                'figsize_small': [12, 8],
                'figsize_square': [12, 12]
            },
            'typography': {
                'title_fontsize': 24,
                'subtitle_fontsize': 18,
                'label_fontsize': 14,
                'tick_fontsize': 11,
                'annotation_fontsize': 12,
                'legend_fontsize': 12
            },
            'colors': {
                'model_palette': [
                    "#2E86AB",  # Professional blue (AmapVox)
                    "#A23B72",  # Vibrant magenta (VoxLAD)
                    "#F18F01",  # Warm orange (VoxPy)
                    "#C73E1D",  # Deep red
                    "#592E83",  # Purple
                    "#1B998B"   # Teal
                ],
                'density_colors': {
                    'lad': '#2ECC71',  # Green for leaf
                    'wad': '#8B4513',  # Brown for wood  
                    'pad': '#4A90E2'   # Blue for plant
                },
                'cluster_colors': [
                    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
                    '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
                ],
                'diverging_cmap': 'RdBu_r',
                'sequential_cmap': 'viridis'
            }
        }
    
    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality figures."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        plt.rcParams.update({
            'figure.dpi': self.config['figure_settings']['dpi'],
            'savefig.dpi': self.config['figure_settings']['dpi'],
            'font.size': self.config['typography']['label_fontsize'],
            'axes.labelsize': self.config['typography']['label_fontsize'],
            'axes.titlesize': self.config['typography']['subtitle_fontsize'],
            'xtick.labelsize': self.config['typography']['tick_fontsize'],
            'ytick.labelsize': self.config['typography']['tick_fontsize'],
            'legend.fontsize': self.config['typography']['legend_fontsize'],
            'figure.titlesize': self.config['typography']['title_fontsize'],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 1.5,
            'grid.linewidth': 0.8,
            'lines.linewidth': 2,
            'patch.linewidth': 1,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'svg.fonttype': 'none',
            'pdf.fonttype': 42,
            'ps.fonttype': 42
        })
    
    def create_comprehensive_ml_dashboard(self, 
                                        ml_results: Dict[str, Any],
                                        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
                                        output_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive ML analysis dashboard.
        
        Args:
            ml_results: Results from ML analysis
            data_dict: Original voxel data (optional, for enhanced visualizations)
            output_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info("Creating comprehensive ML dashboard")
        
        # Create large figure with complex grid
        fig = plt.figure(figsize=self.config['figsize']['dashboard_large'])
        gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.25)
        
        # Main title
        fig.suptitle('Machine Learning Analysis Dashboard: Forest Structure Patterns', 
                    fontsize=self.config['typography']['title_fontsize'],
                    fontweight='bold', y=0.98)
        
        # Row 1: Clustering Analysis (spans 2 columns each)
        try:
            self.logger.info("DEBUG: Creating clustering comparison plot")
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_clustering_comparison(ax1, ml_results)
            self.logger.info("DEBUG: Clustering comparison plot completed")
        except Exception as e:
            self.logger.error(f"DEBUG: Error in clustering comparison: {e}")
            import traceback
            self.logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        try:
            self.logger.info("DEBUG: Creating cluster characteristics plot")
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_cluster_characteristics(ax2, ml_results)
            self.logger.info("DEBUG: Cluster characteristics plot completed")
        except Exception as e:
            self.logger.error(f"DEBUG: Error in cluster characteristics: {e}")
            import traceback
            self.logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        # Row 2: Dimensionality Reduction
        plot_functions = [
            (gs[1, 0], self._plot_pca_analysis, "PCA analysis"),
            (gs[1, 1], self._plot_tsne_visualization, "t-SNE visualization"), 
            (gs[1, 2], self._plot_feature_importance, "feature importance"),
            (gs[1, 3], self._plot_explained_variance, "explained variance")
        ]
        
        for grid_spec, plot_func, name in plot_functions:
            try:
                self.logger.info(f"DEBUG: Creating {name} plot")
                ax = fig.add_subplot(grid_spec)
                if name == "t-SNE visualization":
                    plot_func(ax, ml_results, data_dict)
                else:
                    plot_func(ax, ml_results)
                self.logger.info(f"DEBUG: {name} plot completed")
            except Exception as e:
                self.logger.error(f"DEBUG: Error in {name}: {e}")
                import traceback
                self.logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        # Row 3-5: Remaining plots with comprehensive debugging
        remaining_plots = [
            (gs[2, :2], self._plot_density_hotspots, "density hotspots", True),
            (gs[2, 2:], self._plot_height_stratified_patterns, "height stratified patterns", False),
            (gs[3, :2], self._plot_physical_accuracy_assessment, "physical accuracy assessment", False),
            (gs[3, 2:], self._plot_occlusion_analysis, "occlusion analysis", False),
            (gs[4, :2], self._plot_model_similarity_matrix, "model similarity matrix", False),
            (gs[4, 2:], self._plot_performance_rankings, "performance rankings", False)
        ]
        
        for grid_spec, plot_func, name, needs_data_dict in remaining_plots:
            try:
                self.logger.info(f"DEBUG: Creating {name} plot")
                ax = fig.add_subplot(grid_spec)
                if needs_data_dict:
                    plot_func(ax, ml_results, data_dict)
                else:
                    plot_func(ax, ml_results)
                self.logger.info(f"DEBUG: {name} plot completed")
            except Exception as e:
                self.logger.error(f"DEBUG: Error in {name}: {e}")
                import traceback
                self.logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
        
        # Save figure if path provided
        if output_path:
            self._save_figure(fig, output_path, 'ml_comprehensive_dashboard')
        
        return fig
    
    def _plot_clustering_comparison(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot clustering analysis comparison."""
        clustering_results = ml_results.get('clustering_analysis', {})
        
        if not clustering_results:
            ax.text(0.5, 0.5, 'No clustering results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Clustering Analysis Comparison')
            return
        
        # Extract silhouette scores for different models and cluster numbers
        models = list(clustering_results.keys())
        cluster_numbers = []
        silhouette_scores = []
        model_labels = []
        
        for model_name, model_clustering in clustering_results.items():
            kmeans_results = model_clustering.get('kmeans', {})
            for n_clusters, cluster_data in kmeans_results.items():
                if isinstance(n_clusters, int) and 'silhouette_score' in cluster_data:
                    cluster_numbers.append(n_clusters)
                    silhouette_scores.append(cluster_data['silhouette_score'])
                    model_labels.append(model_name)
        
        if not silhouette_scores:
            ax.text(0.5, 0.5, 'No silhouette scores available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Clustering Quality Comparison')
            return
        
        # Create grouped bar plot
        df = pd.DataFrame({
            'Model': model_labels,
            'Clusters': cluster_numbers,
            'Silhouette_Score': silhouette_scores
        })
        
        # Group by model and plot
        colors = self.config['colors']['model_palette']
        x_pos = 0
        
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            if len(model_data) > 0:
                x_positions = np.arange(x_pos, x_pos + len(model_data))
                ax.bar(x_positions, model_data['Silhouette_Score'].values,
                      color=colors[i % len(colors)], alpha=0.8, 
                      label=model, width=0.8)
                
                # Add cluster number labels
                for j, (pos, score, clusters) in enumerate(zip(x_positions, 
                                                              model_data['Silhouette_Score'].values,
                                                              model_data['Clusters'].values)):
                    ax.text(pos, score + 0.01, f'{clusters}c', 
                           ha='center', va='bottom', fontsize=9)
                
                x_pos += len(model_data) + 1
        
        ax.set_xlabel('Models and Cluster Configurations')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality Comparison\n(Higher is Better)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_cluster_characteristics(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot cluster characteristics analysis."""
        clustering_results = ml_results.get('clustering_analysis', {})
        
        # Find the best performing model for detailed analysis
        best_model = None
        best_score = -1
        
        for model_name, model_clustering in clustering_results.items():
            best_kmeans = model_clustering.get('best_kmeans', {})
            score = best_kmeans.get('score', 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if best_model and best_model in clustering_results:
            characteristics = clustering_results[best_model].get('cluster_characteristics', {})
            
            if characteristics:
                # Plot cluster sizes
                cluster_ids = []
                cluster_sizes = []
                cluster_densities = []
                
                for cluster_id, cluster_data in characteristics.items():
                    cluster_ids.append(cluster_id.replace('cluster_', 'C'))
                    cluster_sizes.append(cluster_data.get('percentage', 0))
                    density_stats = cluster_data.get('density_stats', {})
                    cluster_densities.append(density_stats.get('mean', 0))
                
                # Create scatter plot of cluster size vs mean density
                colors = self.config['colors']['cluster_colors'][:len(cluster_ids)]
                scatter = ax.scatter(cluster_sizes, cluster_densities, 
                                   c=colors, s=200, alpha=0.7, edgecolors='black')
                
                # Add cluster labels
                for i, (size, density, cluster_id) in enumerate(zip(cluster_sizes, cluster_densities, cluster_ids)):
                    ax.annotate(cluster_id, (size, density), xytext=(5, 5), 
                              textcoords='offset points', fontsize=10, fontweight='bold')
                
                ax.set_xlabel('Cluster Size (%)')
                ax.set_ylabel('Mean Density')
                ax.set_title(f'Cluster Characteristics - {best_model}\n(Size vs Mean Density)')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No cluster characteristics for {best_model}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Cluster Characteristics')
        else:
            ax.text(0.5, 0.5, 'No clustering characteristics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cluster Characteristics')
    
    def _plot_pca_analysis(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot PCA analysis results."""
        dr_results = ml_results.get('dimensionality_reduction', {})
        
        if not dr_results:
            ax.text(0.5, 0.5, 'No PCA results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA Analysis')
            return
        
        # Find a model with PCA results
        model_with_pca = None
        for model_name, model_dr in dr_results.items():
            if 'pca' in model_dr and model_dr['pca']:
                model_with_pca = model_name
                break
        
        if model_with_pca:
            pca_results = dr_results[model_with_pca]['pca']
            
            # Plot explained variance for different component numbers
            components = []
            cumulative_variance = []
            
            for n_comp, pca_data in pca_results.items():
                if isinstance(n_comp, int):
                    components.append(n_comp)
                    cum_var = pca_data.get('cumulative_variance', [])
                    if len(cum_var) > 0:  # Safe check for array length
                        cumulative_variance.append(cum_var[-1])  # Last value is total
            
            if components and cumulative_variance:
                ax.plot(components, cumulative_variance, 'o-', 
                       color=self.config['colors']['model_palette'][0], linewidth=2, markersize=8)
                ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
                ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
                
                ax.set_xlabel('Number of Components')
                ax.set_ylabel('Cumulative Explained Variance')
                ax.set_title(f'PCA Explained Variance - {model_with_pca}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            else:
                ax.text(0.5, 0.5, 'No valid PCA variance data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('PCA Analysis')
        else:
            ax.text(0.5, 0.5, 'No PCA results found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('PCA Analysis')
    
    def _plot_tsne_visualization(self, ax: plt.Axes, ml_results: Dict[str, Any], 
                                data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """Plot t-SNE visualization."""
        dr_results = ml_results.get('dimensionality_reduction', {})
        
        # Find a model with t-SNE results
        model_with_tsne = None
        tsne_data = None
        
        for model_name, model_dr in dr_results.items():
            tsne_results = model_dr.get('tsne', {})
            if tsne_results:
                # Get the first available t-SNE result
                for perplexity, tsne_info in tsne_results.items():
                    if 'transformed_data' in tsne_info:
                        model_with_tsne = model_name
                        tsne_data = tsne_info['transformed_data']
                        break
                if tsne_data is not None:
                    break
        
        if model_with_tsne and tsne_data is not None:
            # Get clustering information for coloring if available
            clustering_results = ml_results.get('clustering_analysis', {}).get(model_with_tsne, {})
            best_kmeans = clustering_results.get('best_kmeans', {})
            
            if best_kmeans and 'n_clusters' in best_kmeans:
                n_clusters = best_kmeans['n_clusters']
                kmeans_results = clustering_results.get('kmeans', {})
                cluster_labels = kmeans_results.get(n_clusters, {}).get('cluster_labels')
                
                if cluster_labels is not None and len(cluster_labels) == len(tsne_data):
                    # Color by clusters
                    colors = self.config['colors']['cluster_colors']
                    unique_labels = np.unique(cluster_labels)
                    
                    for i, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                                 c=colors[i % len(colors)], label=f'Cluster {label}',
                                 alpha=0.6, s=20)
                    
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    # Default coloring
                    ax.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                             c=self.config['colors']['model_palette'][0], alpha=0.6, s=20)
            else:
                ax.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                         c=self.config['colors']['model_palette'][0], alpha=0.6, s=20)
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f't-SNE Visualization - {model_with_tsne}')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No t-SNE results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('t-SNE Visualization')
    
    def _plot_feature_importance(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot feature importance from Random Forest analysis."""
        regression_results = ml_results.get('regression_analysis', {})
        
        feature_importance_data = []
        
        # Extract feature importance from pairwise regressions
        for pair_name, pair_results in regression_results.get('pairwise_regressions', {}).items():
            rf_results = pair_results.get('random_forest', {})
            if 'feature_importance' in rf_results:
                importance = rf_results['feature_importance']
                for i, imp in enumerate(importance):
                    feature_importance_data.append({
                        'pair': pair_name.replace('_vs_', ' vs '),
                        'feature': f'Feature_{i+1}',
                        'importance': imp
                    })
        
        if feature_importance_data:
            df = pd.DataFrame(feature_importance_data)
            
            # Create grouped bar plot
            pairs = df['pair'].unique()
            features = df['feature'].unique()
            
            x = np.arange(len(features))
            width = 0.8 / len(pairs)
            
            colors = self.config['colors']['model_palette']
            
            for i, pair in enumerate(pairs):
                pair_data = df[df['pair'] == pair]
                pair_importance = [pair_data[pair_data['feature'] == f]['importance'].iloc[0] 
                                 if len(pair_data[pair_data['feature'] == f]) > 0 else 0 
                                 for f in features]
                
                ax.bar(x + i * width, pair_importance, width, 
                      label=pair, color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Random Forest Feature Importance')
            ax.set_xticks(x + width * (len(pairs) - 1) / 2)
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No feature importance data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
    
    def _plot_explained_variance(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot explained variance ratios for PCA components."""
        dr_results = ml_results.get('dimensionality_reduction', {})
        
        # Find best PCA result to display
        best_model = None
        best_variance = None
        
        for model_name, model_dr in dr_results.items():
            pca_results = model_dr.get('pca', {})
            for n_comp, pca_data in pca_results.items():
                if 'explained_variance_ratio' in pca_data:
                    variance_ratio = pca_data['explained_variance_ratio']
                    if len(variance_ratio) >= 3:  # At least 3 components
                        best_model = model_name
                        best_variance = variance_ratio
                        break
            if best_variance is not None:
                break
        
        if best_variance is not None:
            components = range(1, len(best_variance) + 1)
            
            # Individual variance explained
            ax.bar(components, best_variance, alpha=0.7, 
                  color=self.config['colors']['model_palette'][0], label='Individual')
            
            # Cumulative variance (line plot)
            ax2 = ax.twinx()
            cumulative = np.cumsum(best_variance)
            ax2.plot(components, cumulative, 'ro-', linewidth=2, markersize=6, 
                    color='red', label='Cumulative')
            ax2.set_ylabel('Cumulative Explained Variance', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title(f'PCA Explained Variance - {best_model}')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax.text(0.5, 0.5, 'No explained variance data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Explained Variance')
    
    def _plot_density_hotspots(self, ax: plt.Axes, ml_results: Dict[str, Any], 
                              data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """Plot density hotspot analysis."""
        spatial_results = ml_results.get('spatial_patterns', {})
        
        if not spatial_results:
            ax.text(0.5, 0.5, 'No spatial pattern results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Density Hotspot Analysis')
            return
        
        # Collect hotspot data
        models = []
        high_density_percentages = []
        low_density_percentages = []
        
        for model_name, spatial_data in spatial_results.items():
            hotspots = spatial_data.get('density_hotspots', {})
            
            high_regions = hotspots.get('high_density_regions', {})
            low_regions = hotspots.get('low_density_regions', {})
            
            if high_regions and low_regions:
                models.append(model_name)
                high_density_percentages.append(high_regions.get('percentage', 0))
                low_density_percentages.append(low_regions.get('percentage', 0))
        
        if models:
            x = np.arange(len(models))
            width = 0.35
            
            # Create grouped bar chart
            bars1 = ax.bar(x - width/2, high_density_percentages, width, 
                          label='High Density Regions', 
                          color=self.config['colors']['density_colors']['pad'], alpha=0.8)
            bars2 = ax.bar(x + width/2, low_density_percentages, width, 
                          label='Low Density Regions', 
                          color=self.config['colors']['density_colors']['lad'], alpha=0.8)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.1f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Percentage of Total Voxels (%)')
            ax.set_title('Density Hotspot Distribution Across Models')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hotspot data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Density Hotspot Analysis')
    
    def _plot_height_stratified_patterns(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot height-stratified density patterns."""
        spatial_results = ml_results.get('spatial_patterns', {})
        
        if not spatial_results:
            ax.text(0.5, 0.5, 'No spatial pattern results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Height-Stratified Patterns')
            return
        
        # Collect height layer data
        models = list(spatial_results.keys())
        height_layers = ['lower', 'middle', 'upper']
        
        # Create data matrix for heatmap
        data_matrix = np.zeros((len(models), len(height_layers)))
        
        for i, model_name in enumerate(models):
            height_patterns = spatial_results[model_name].get('height_stratified_patterns', {})
            
            for j, layer in enumerate(height_layers):
                layer_data = height_patterns.get(layer, {})
                density_stats = layer_data.get('density_statistics', {})
                mean_density = density_stats.get('mean', 0)
                data_matrix[i, j] = mean_density
        
        # Create heatmap
        if np.any(data_matrix > 0):
            im = ax.imshow(data_matrix, cmap=self.config['colors']['sequential_cmap'], 
                          aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Mean Density')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(height_layers)))
            ax.set_yticks(np.arange(len(models)))
            ax.set_xticklabels([layer.capitalize() for layer in height_layers])
            ax.set_yticklabels(models)
            
            # Add text annotations
            for i in range(len(models)):
                for j in range(len(height_layers)):
                    # Safe scalar extraction to avoid array ambiguity
                    cell_value = safe_scalar_extract(data_matrix[i, j])
                    threshold = safe_scalar_extract(data_matrix.max() / 2)
                    text_color = "white" if cell_value > threshold else "black"
                    
                    text = ax.text(j, i, f'{cell_value:.3f}',
                                 ha="center", va="center", color=text_color,
                                 fontsize=10, fontweight='bold')
            
            ax.set_title('Mean Density by Height Layer and Model')
            ax.set_xlabel('Crown Layer')
            ax.set_ylabel('Model')
        else:
            ax.text(0.5, 0.5, 'No height-stratified density data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Height-Stratified Patterns')
    
    def _plot_physical_accuracy_assessment(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot physical accuracy assessment."""
        accuracy_results = ml_results.get('physical_accuracy', {})
        
        if not accuracy_results:
            ax.text(0.5, 0.5, 'No physical accuracy results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Physical Accuracy Assessment')
            return
        
        # Collect accuracy metrics
        models = []
        outlier_percentages = []
        zero_density_percentages = []
        negative_densities = []
        
        for model_name, model_accuracy in accuracy_results.items():
            distribution_analysis = model_accuracy.get('density_distribution_analysis', {})
            plausibility = model_accuracy.get('physical_plausibility', {})
            
            if distribution_analysis:
                models.append(model_name)
                outlier_percentages.append(distribution_analysis.get('outlier_percentage', 0))
                zero_density_percentages.append(distribution_analysis.get('zero_density_percentage', 0))
                negative_densities.append(plausibility.get('negative_densities', 0))
        
        if models:
            x = np.arange(len(models))
            width = 0.25
            
            # Create grouped bar chart
            bars1 = ax.bar(x - width, outlier_percentages, width, 
                          label='Outliers (%)', alpha=0.8, color='orange')
            bars2 = ax.bar(x, zero_density_percentages, width, 
                          label='Zero Density (%)', alpha=0.8, color='lightblue')
            bars3 = ax.bar(x + width, negative_densities, width, 
                          label='Negative Values', alpha=0.8, color='red')
            
            # Add value labels on bars
            for bars, values in zip([bars1, bars2, bars3], [outlier_percentages, zero_density_percentages, negative_densities]):
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    # Safe scalar extraction to avoid array ambiguity
                    val_scalar = safe_scalar_extract(val)
                    
                    if val_scalar > 0:
                        ax.annotate(f'{val_scalar:.1f}' if val_scalar < 100 else f'{int(val_scalar)}',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3), textcoords="offset points",
                                  ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Count / Percentage')
            ax.set_title('Physical Plausibility Assessment')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis to log scale if values vary greatly - with safe array handling
            try:
                max_outliers = safe_scalar_extract(np.max(outlier_percentages)) if len(outlier_percentages) > 0 else 0
                max_zeros = safe_scalar_extract(np.max(zero_density_percentages)) if len(zero_density_percentages) > 0 else 0
                min_negatives = safe_scalar_extract(np.min(negative_densities)) if len(negative_densities) > 0 else 0
                
                if max(max_outliers, max_zeros) > 10 * min_negatives and min_negatives > 0:
                    ax.set_yscale('log')
            except (ValueError, TypeError, AttributeError):
                pass  # Skip log scale if data is problematic
        else:
            ax.text(0.5, 0.5, 'No accuracy metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Physical Accuracy Assessment')
    
    def _plot_occlusion_analysis(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot occlusion analysis results."""
        occlusion_results = ml_results.get('occlusion_analysis', {})
        
        if not occlusion_results:
            ax.text(0.5, 0.5, 'No occlusion analysis results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Occlusion Analysis')
            return
        
        # Collect occlusion metrics
        models = []
        shadow_percentages = []
        correction_scores = []
        
        for model_name, model_occlusion in occlusion_results.items():
            shadow_analysis = model_occlusion.get('shadow_analysis', {})
            correction_assessment = model_occlusion.get('occlusion_correction_assessment', {})
            
            models.append(model_name)
            shadow_percentages.append(shadow_analysis.get('shadow_percentage', 0))
            
            # Convert correction assessment to numeric score
            empirical = correction_assessment.get('empirical_assessment', 'unknown')
            score_map = {'good': 3, 'moderate': 2, 'poor': 1, 'unknown': 1.5}
            correction_scores.append(score_map.get(empirical, 1.5))
        
        if models:
            # Create double y-axis plot
            ax2 = ax.twinx()
            
            x = np.arange(len(models))
            
            # Shadow percentage (bars)
            bars = ax.bar(x, shadow_percentages, alpha=0.6, color='gray', label='Shadow Regions (%)')
            
            # Correction score (line plot)
            line = ax2.plot(x, correction_scores, 'ro-', linewidth=3, markersize=8, 
                           color='red', label='Correction Quality')
            
            # Customize axes
            ax.set_xlabel('Models')
            ax.set_ylabel('Shadow Regions (%)', color='gray')
            ax2.set_ylabel('Occlusion Correction Quality', color='red')
            ax2.set_ylim(0.5, 3.5)
            ax2.set_yticks([1, 2, 3])
            ax2.set_yticklabels(['Poor', 'Moderate', 'Good'])
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, val in zip(bars, shadow_percentages):
                height = bar.get_height()
                # Safe scalar extraction to avoid array ambiguity
                val_scalar = safe_scalar_extract(val)
                
                if val_scalar > 0:
                    ax.annotate(f'{val_scalar:.1f}%',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3), textcoords="offset points",
                              ha='center', va='bottom', fontsize=9)
            
            ax.set_title('Occlusion Analysis: Shadow Detection & Correction Assessment')
            ax.grid(True, alpha=0.3)
            
            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax.text(0.5, 0.5, 'No occlusion metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Occlusion Analysis')
    
    def _plot_model_similarity_matrix(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot model similarity matrix."""
        comparative_results = ml_results.get('comparative_analysis', {})
        similarity_analysis = comparative_results.get('model_similarity_analysis', {})
        
        if not similarity_analysis or 'similarity_matrix' not in similarity_analysis:
            ax.text(0.5, 0.5, 'No model similarity data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Similarity Matrix')
            return
        
        similarity_matrix = np.array(similarity_analysis['similarity_matrix'])
        model_names = similarity_analysis.get('model_names', [])
        
        if len(similarity_matrix) > 0 and len(model_names) > 0:
            # Create heatmap
            im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Similarity Score')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(model_names)))
            ax.set_yticks(np.arange(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_yticklabels(model_names)
            
            # Add text annotations
            for i in range(len(model_names)):
                for j in range(len(model_names)):
                    if i != j:  # Don't annotate diagonal
                        # Safe scalar extraction to avoid array ambiguity
                        cell_value = safe_scalar_extract(similarity_matrix[i, j])
                        text_color = "white" if cell_value < 0.5 else "black"
                        
                        text = ax.text(j, i, f'{cell_value:.3f}',
                                     ha="center", va="center", color=text_color,
                                     fontsize=10, fontweight='bold')
            
            ax.set_title('Model Similarity Matrix\n(Based on Spatial Pattern Features)')
        else:
            ax.text(0.5, 0.5, 'Invalid similarity matrix data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Similarity Matrix')
    
    def _plot_performance_rankings(self, ax: plt.Axes, ml_results: Dict[str, Any]):
        """Plot performance rankings of models."""
        comparative_results = ml_results.get('comparative_analysis', {})
        performance_ranking = comparative_results.get('performance_ranking', {})
        
        if not performance_ranking or 'ranked_models' not in performance_ranking:
            ax.text(0.5, 0.5, 'No performance ranking data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance Rankings')
            return
        
        ranked_models = performance_ranking['ranked_models']
        criteria_scores = performance_ranking.get('criteria_scores', {})
        
        if ranked_models:
            models = [model for model, score in ranked_models]
            overall_scores = [score for model, score in ranked_models]
            
            # Create horizontal bar chart for overall ranking
            y_pos = np.arange(len(models))
            colors = self.config['colors']['model_palette'][:len(models)]
            
            bars = ax.barh(y_pos, overall_scores, color=colors, alpha=0.8)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, overall_scores)):
                ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', va='center', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(models)
            ax.set_xlabel('Overall Performance Score')
            ax.set_title('Model Performance Rankings\n(Higher is Better)')
            ax.grid(True, alpha=0.3, axis='x')
            ax.set_xlim(0, max(overall_scores) * 1.15)
            
            # Add ranking positions
            for i, (model, score) in enumerate(ranked_models):
                ax.text(-0.05, i, f'#{i+1}', ha='right', va='center', 
                       fontweight='bold', fontsize=12, 
                       transform=ax.get_yaxis_transform())
        else:
            ax.text(0.5, 0.5, 'No ranking data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Performance Rankings')
    
    def create_3d_clustering_visualization(self, 
                                         data_dict: Dict[str, pd.DataFrame],
                                         ml_results: Dict[str, Any],
                                         model_name: str,
                                         output_path: Optional[Path] = None) -> plt.Figure:
        """
        Create 3D clustering visualization for a specific model.
        
        Args:
            data_dict: Original voxel data
            ml_results: ML analysis results
            model_name: Name of the model to visualize
            output_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        self.logger.info(f"Creating 3D clustering visualization for {model_name}")
        
        if model_name not in data_dict:
            self.logger.error(f"Model {model_name} not found in data_dict")
            return None
        
        df = data_dict[model_name]
        if not all(col in df.columns for col in ['x', 'y', 'z', 'density_value']):
            self.logger.error(f"Required columns missing for {model_name}")
            return None
        
        fig = plt.figure(figsize=self.config['figsize']['square_plot'])
        ax = fig.add_subplot(111, projection='3d')
        
        # Get clustering results for this model
        clustering_results = ml_results.get('clustering_analysis', {}).get(model_name, {})
        best_kmeans = clustering_results.get('best_kmeans', {})
        
        if best_kmeans and 'n_clusters' in best_kmeans:
            n_clusters = best_kmeans['n_clusters']
            kmeans_results = clustering_results.get('kmeans', {})
            cluster_info = kmeans_results.get(n_clusters, {})
            cluster_labels = cluster_info.get('cluster_labels')
            
            if cluster_labels is not None:
                # Sample data for performance (3D plots can be slow)
                if len(df) > 5000:
                    # Ensure cluster_labels matches df size before sampling
                    if len(cluster_labels) == len(df):
                        sample_idx = np.random.choice(len(df), 5000, replace=False)
                        df_sample = df.iloc[sample_idx]
                        cluster_labels_sample = cluster_labels[sample_idx]
                    else:
                        # Mismatch in sizes - use original data up to cluster_labels size
                        min_size = min(len(df), len(cluster_labels))
                        if min_size > 5000:
                            sample_idx = np.random.choice(min_size, 5000, replace=False)
                            df_sample = df.iloc[sample_idx]
                            cluster_labels_sample = cluster_labels[sample_idx]
                        else:
                            df_sample = df.iloc[:min_size]
                            cluster_labels_sample = cluster_labels[:min_size]
                else:
                    # Ensure sizes match for smaller datasets
                    min_size = min(len(df), len(cluster_labels))
                    df_sample = df.iloc[:min_size]
                    cluster_labels_sample = cluster_labels[:min_size]
                
                # Plot each cluster with different color
                colors = self.config['colors']['cluster_colors']
                unique_labels = np.unique(cluster_labels_sample)
                
                for i, label in enumerate(unique_labels):
                    mask = cluster_labels_sample == label
                    cluster_data = df_sample[mask]
                    
                    scatter = ax.scatter(cluster_data['x'], cluster_data['y'], cluster_data['z'],
                                       c=colors[i % len(colors)], label=f'Cluster {label}',
                                       alpha=0.6, s=20)
                
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # No clustering labels, color by density
                scatter = ax.scatter(df['x'], df['y'], df['z'], 
                                   c=df['density_value'], 
                                   cmap=self.config['colors']['sequential_cmap'],
                                   alpha=0.6, s=20)
                plt.colorbar(scatter, ax=ax, shrink=0.8, label='Density')
        else:
            # No clustering results, color by density
            scatter = ax.scatter(df['x'], df['y'], df['z'], 
                               c=df['density_value'], 
                               cmap=self.config['colors']['sequential_cmap'],
                               alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, shrink=0.8, label='Density')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Height (Z)')
        ax.set_title(f'3D Clustering Visualization - {model_name}')
        
        # Save figure if path provided
        if output_path:
            self._save_figure(fig, output_path, f'3d_clustering_{model_name.lower()}')
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, output_path: Path, name: str):
        """Save figure in multiple formats."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"{name}_{timestamp}"
        
        # Save in multiple formats
        formats = ['png', 'pdf', 'svg']
        for fmt in formats:
            filepath = output_path / f"{base_name}.{fmt}"
            try:
                fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                           dpi=self.config['figure_settings']['dpi'])
                self.logger.info(f"Saved ML visualization to {filepath}")
            except Exception as e:
                self.logger.error(f"Failed to save {filepath}: {e}")


if __name__ == "__main__":
    # Example usage
    visualizer = MLVisualizer()
    
    print("ML Visualizer initialized successfully!")
    print("Available visualization methods:")
    print("- Comprehensive ML dashboard")
    print("- 3D clustering visualizations")
    print("- Dimensionality reduction plots")
    print("- Spatial pattern analysis")
    print("- Physical accuracy assessment")
    print("- Model comparison dashboards")