#!/usr/bin/env python3
"""
Statistical Visualization Module for VoxPlot
============================================
Publication-quality visualizations for statistical analysis results,
following Nature Plant journal standards.

This module provides:
- Statistical comparison plots (box plots, violin plots)
- Error metric visualizations (bar charts, heatmaps)
- Correlation and agreement plots (scatter plots, Bland-Altman)
- Model performance dashboards
- Publication-ready figure export
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


class StatisticalVisualizer:
    """
    Create publication-quality statistical visualizations for model comparisons.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the statistical visualizer with Nature journal styling.
        
        Args:
            config: Visualization configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Ensure we have a complete configuration
        default_config = self._get_default_config()
        if config is None:
            self.config = default_config
        else:
            # Merge with defaults to ensure all required keys exist
            self.config = default_config.copy()
            if 'visualization' in config:
                self.config['visualization'].update(config['visualization'])
        
        # Apply Nature journal styling
        self._setup_publication_style()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default visualization configuration."""
        return {
            'visualization': {
                'dpi': 600,
                'figsize_large': [20, 24],
                'figsize_medium': [16, 12],
                'figsize_small': [12, 9],
                'typography': {
                    'title_fontsize': 24,
                    'subtitle_fontsize': 18,
                    'label_fontsize': 14,
                    'tick_fontsize': 11,
                    'annotation_fontsize': 12
                },
                'colors': {
                    'model_palette': [
                        "#4A90E2",  # Professional blue (AmapVox)
                        "#E74C3C",  # Vibrant coral red (VoxLAD)
                        "#2ECC71",  # Fresh green (VoxPy)
                        "#9B59B6",  # Rich purple
                        "#F39C12",  # Warm orange
                    ],
                    'density_colors': {
                        'lad': '#2ECC71',  # Green for leaf
                        'wad': '#8B4513',  # Brown for wood
                        'pad': '#4A90E2'   # Blue for plant
                    }
                }
            }
        }
    
    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality figures."""
        # Use seaborn for better defaults
        sns.set_style("whitegrid")
        
        # Update matplotlib parameters
        plt.rcParams.update({
            'figure.dpi': self.config['visualization']['dpi'],
            'savefig.dpi': self.config['visualization']['dpi'],
            'font.size': self.config['visualization']['typography']['label_fontsize'],
            'axes.labelsize': self.config['visualization']['typography']['label_fontsize'],
            'axes.titlesize': self.config['visualization']['typography']['subtitle_fontsize'],
            'xtick.labelsize': self.config['visualization']['typography']['tick_fontsize'],
            'ytick.labelsize': self.config['visualization']['typography']['tick_fontsize'],
            'legend.fontsize': self.config['visualization']['typography']['label_fontsize'],
            'figure.titlesize': self.config['visualization']['typography']['title_fontsize'],
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
            'svg.fonttype': 'none',  # Ensure text remains editable in SVG
            'pdf.fonttype': 42,  # TrueType fonts for PDF
            'ps.fonttype': 42
        })
    
    def create_comprehensive_dashboard(self, 
                                      comparison_results: Dict[str, Any],
                                      output_path: Optional[Path] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard showing all statistical comparisons.
        
        Args:
            comparison_results: Results from ModelComparison.compare_all_models()
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Create large figure with subplots
        fig = plt.figure(figsize=self.config['visualization']['figsize_large'])
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.25)
        
        # Title
        fig.suptitle('Statistical Model Comparison Dashboard', 
                    fontsize=self.config['visualization']['typography']['title_fontsize'],
                    fontweight='bold', y=0.98)
        
        # 1. Error metrics comparison (top row)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_error_metrics_comparison(ax1, comparison_results)
        
        # 2. Model ranking heatmap (second row, left)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_ranking_heatmap(ax2, comparison_results)
        
        # 3. Statistical test results (second row, middle)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_statistical_tests(ax3, comparison_results)
        
        # 4. Bootstrap confidence intervals (second row, right)
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_bootstrap_ci(ax4, comparison_results)
        
        # 5. Correlation matrix (third row, left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_correlation_matrix(ax5, comparison_results)
        
        # 6. Spatial error distribution (third row, middle)
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_spatial_errors(ax6, comparison_results)
        
        # 7. Cross-validation results (third row, right)
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_cross_validation(ax7, comparison_results)
        
        # 8. Summary table (bottom row)
        ax8 = fig.add_subplot(gs[3, :])
        self._plot_summary_table(ax8, comparison_results)
        
        # Save if path provided
        if output_path:
            self._save_figure(fig, output_path, 'statistical_dashboard')
        
        return fig
    
    def _plot_error_metrics_comparison(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot comparison of error metrics across models and density types."""
        # Extract error metrics
        metrics_data = []
        for density_type, density_results in results.get('by_density_type', {}).items():
            if 'error_metrics' not in density_results:
                continue
            
            for model_name, metrics in density_results['error_metrics'].items():
                metrics_data.append({
                    'Model': model_name,
                    'Density': density_type.upper(),
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'Bias': metrics['bias']
                })
        
        if not metrics_data:
            ax.text(0.5, 0.5, 'No error metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Metrics Comparison')
            return
        
        df = pd.DataFrame(metrics_data)
        
        # Create grouped bar plot
        x = np.arange(len(df['Model'].unique()))
        width = 0.25
        
        models = df['Model'].unique()
        density_types = df['Density'].unique()
        
        colors = self.config['visualization']['colors']['model_palette']
        
        for i, metric in enumerate(['MAE', 'RMSE', 'Bias']):
            for j, density in enumerate(density_types):
                data = df[df['Density'] == density].set_index('Model')[metric]
                offset = (i - 1) * width + j * width / len(density_types)
                
                ax.bar(x + offset, 
                      [data.get(m, 0) for m in models],
                      width / len(density_types),
                      label=f'{density}-{metric}',
                      color=colors[j % len(colors)],
                      alpha=0.7 + i * 0.1)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Error Value')
        ax.set_title('Error Metrics Comparison Across Models and Density Types')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
    
    def _plot_ranking_heatmap(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot model ranking heatmap."""
        rankings = results.get('overall_ranking', {})
        
        if not rankings:
            ax.text(0.5, 0.5, 'No ranking data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Rankings')
            return
        
        # Create ranking matrix
        models = list(rankings.keys())
        metrics = ['mean_rank', 'mean_mae', 'mean_rmse', 'mean_abs_bias']
        
        matrix = np.zeros((len(models), len(metrics)))
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                matrix[i, j] = rankings[model].get(metric, 0)
        
        # Normalize each column to 0-1 for better visualization
        for j in range(matrix.shape[1]):
            col_min = matrix[:, j].min()
            col_max = matrix[:, j].max()
            if col_max > col_min:
                matrix[:, j] = (matrix[:, j] - col_min) / (col_max - col_min)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha='center', va='center', color='black',
                             fontsize=9)
        
        ax.set_title('Model Performance Ranking\n(Lower is Better)')
    
    def _plot_statistical_tests(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot statistical test results."""
        test_results = []
        
        for density_type, density_results in results.get('by_density_type', {}).items():
            if 'statistical_tests' not in density_results:
                continue
            
            tests = density_results['statistical_tests']
            
            # Extract p-values
            if 'one_sample_ttest' in tests:
                test_results.append({
                    'Density': density_type.upper(),
                    'Test': 'One-sample t-test',
                    'p-value': tests['one_sample_ttest']['p_value'],
                    'Significant': tests['one_sample_ttest']['significant']
                })
            
            if 'anova' in tests:
                test_results.append({
                    'Density': density_type.upper(),
                    'Test': 'ANOVA',
                    'p-value': tests['anova']['p_value'],
                    'Significant': tests['anova']['significant']
                })
        
        if not test_results:
            ax.text(0.5, 0.5, 'No statistical test results available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Statistical Tests')
            return
        
        df = pd.DataFrame(test_results)
        
        # Create bar plot of p-values
        x = np.arange(len(df))
        colors = ['red' if sig else 'gray' for sig in df['Significant']]
        
        bars = ax.bar(x, df['p-value'], color=colors, alpha=0.7)
        
        # Add significance threshold line
        ax.axhline(y=0.05, color='black', linestyle='--', linewidth=1, label='α = 0.05')
        
        # Customize plot
        ax.set_xlabel('Test')
        ax.set_ylabel('p-value')
        ax.set_title('Statistical Significance Tests')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Density']}\n{row['Test']}" for _, row in df.iterrows()],
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, p_val in zip(bars, df['p-value']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{p_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    def _plot_bootstrap_ci(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot bootstrap confidence intervals."""
        ci_data = []
        
        for density_type, density_results in results.get('by_density_type', {}).items():
            if 'bootstrap_ci' not in density_results:
                continue
            
            reference = density_results.get('reference_value', 0)
            
            for model_name, ci_info in density_results['bootstrap_ci'].items():
                ci_data.append({
                    'Model': model_name,
                    'Density': density_type.upper(),
                    'Mean': ci_info['mean_estimate'],
                    'Lower': ci_info['ci_95_lower'],
                    'Upper': ci_info['ci_95_upper'],
                    'Reference': reference
                })
        
        if not ci_data:
            ax.text(0.5, 0.5, 'No bootstrap CI data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bootstrap Confidence Intervals')
            return
        
        df = pd.DataFrame(ci_data)
        
        # Create error bar plot
        x = np.arange(len(df))
        colors = self.config['visualization']['colors']['model_palette']
        
        for i, (_, row) in enumerate(df.iterrows()):
            color = colors[i % len(colors)]
            # Calculate error bars ensuring non-negative values
            lower_err = max(0, row['Mean'] - row['Lower'])
            upper_err = max(0, row['Upper'] - row['Mean'])
            ax.errorbar(i, row['Mean'], 
                       yerr=[[lower_err], [upper_err]],
                       fmt='o', markersize=8, color=color, 
                       capsize=5, capthick=2,
                       label=f"{row['Model']} ({row['Density']})")
            
            # Add reference line
            if i == 0 or row['Reference'] != df.iloc[i-1]['Reference']:
                ax.axhline(y=row['Reference'], color='red', linestyle='--', 
                          alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Model-Density Combination')
        ax.set_ylabel('Area Index Value')
        ax.set_title('95% Bootstrap Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Model'][:4]}-{row['Density']}" for _, row in df.iterrows()],
                          rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_matrix(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot correlation matrix between models."""
        correlations = {}
        
        for density_type, density_results in results.get('by_density_type', {}).items():
            if 'agreement_analysis' not in density_results:
                continue
            
            if 'pairwise' in density_results['agreement_analysis']:
                for comparison, agreement in density_results['agreement_analysis']['pairwise'].items():
                    key = f"{comparison} ({density_type.upper()})"
                    correlations[key] = agreement['pearson_r']
        
        if not correlations:
            ax.text(0.5, 0.5, 'No correlation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Correlations')
            return
        
        # Create correlation matrix visualization
        labels = list(correlations.keys())
        values = list(correlations.values())
        
        # Create bar plot
        x = np.arange(len(labels))
        colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
        
        bars = ax.barh(x, values, color=colors, alpha=0.7)
        
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Pearson Correlation Coefficient')
        ax.set_title('Pairwise Model Correlations')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
    
    def _plot_spatial_errors(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot spatial error analysis."""
        error_data = []
        
        for density_type, density_results in results.get('by_density_type', {}).items():
            if 'spatial_analysis' not in density_results:
                continue
            
            for comparison, spatial in density_results['spatial_analysis'].items():
                if 'error_by_height' in spatial:
                    for height_zone, stats in spatial['error_by_height'].items():
                        error_data.append({
                            'Comparison': comparison.replace('_', ' '),
                            'Density': density_type.upper(),
                            'Height': height_zone.capitalize(),
                            'Mean Error': stats['mean'],
                            'Std Error': stats['std']
                        })
        
        if not error_data:
            ax.text(0.5, 0.5, 'No spatial error data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Spatial Error Distribution')
            return
        
        df = pd.DataFrame(error_data)
        
        # Create grouped violin plot
        if len(df) > 0:
            # Pivot data for visualization
            pivot_df = df.pivot_table(values='Mean Error', 
                                     index='Height', 
                                     columns='Density',
                                     aggfunc='mean')
            
            # Create heatmap
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r',
                       center=0, ax=ax, cbar_kws={'label': 'Mean Error'})
            
            ax.set_title('Spatial Error Distribution by Height Zone')
            ax.set_xlabel('Density Type')
            ax.set_ylabel('Height Zone')
    
    def _plot_cross_validation(self, ax: plt.Axes, results: Dict[str, Any]):
        """Plot cross-validation results."""
        cv_data = []
        
        for density_type, density_results in results.get('by_density_type', {}).items():
            if 'cross_validation' not in density_results:
                continue
            
            for model_name, cv_info in density_results['cross_validation'].items():
                cv_data.append({
                    'Model': model_name,
                    'Density': density_type.upper(),
                    'Mean': cv_info['mean_index'],
                    'Std': cv_info['std_index'],
                    'CV': cv_info['cv_coefficient']
                })
        
        if not cv_data:
            ax.text(0.5, 0.5, 'No cross-validation data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Cross-Validation Results')
            return
        
        df = pd.DataFrame(cv_data)
        
        # Create scatter plot with error bars
        x = np.arange(len(df))
        colors = self.config['visualization']['colors']['model_palette']
        
        for i, (_, row) in enumerate(df.iterrows()):
            color = colors[i % len(colors)]
            ax.errorbar(i, row['Mean'], yerr=row['Std'],
                       fmt='o', markersize=10, color=color,
                       capsize=5, capthick=2,
                       label=f"{row['Model']} ({row['Density']})")
            
            # Add CV coefficient as text
            ax.text(i, row['Mean'] + row['Std'] + 0.05,
                   f"CV: {row['CV']:.2%}", ha='center', fontsize=8)
        
        ax.set_xlabel('Model-Density Combination')
        ax.set_ylabel('Cross-Validated Index')
        ax.set_title('Cross-Validation Stability Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Model'][:4]}-{row['Density']}" for _, row in df.iterrows()],
                          rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
    
    def _plot_summary_table(self, ax: plt.Axes, results: Dict[str, Any]):
        """Create summary table of key findings."""
        ax.axis('tight')
        ax.axis('off')
        
        # Extract summary data
        summary = results.get('summary', {})
        overall_ranking = results.get('overall_ranking', {})
        
        # Create table data
        table_data = []
        
        # Add header
        table_data.append(['Metric', 'Best Model', 'Value', 'Notes'])
        
        # Add overall ranking
        if overall_ranking:
            best_model = min(overall_ranking.items(), key=lambda x: x[1]['mean_rank'])
            table_data.append(['Overall Rank', best_model[0], 
                             f"{best_model[1]['mean_rank']:.1f}", 
                             'Mean rank across all metrics'])
        
        # Add best models by metric
        best_by_metric = results.get('best_model_by_metric', {})
        for metric, info in best_by_metric.items():
            if info.get('model'):
                table_data.append([metric.replace('_', ' ').title(),
                                 info.get('model', 'N/A'),
                                 f"{info.get('value', 0):.3f}",
                                 f"For {info.get('density_type', 'N/A').upper()}"])
        
        # Create table
        table = ax.table(cellText=table_data,
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.15, 0.35])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F8F9FA')
        
        ax.set_title('Summary of Key Findings', fontsize=14, fontweight='bold', pad=20)
    
    def create_model_comparison_plot(self,
                                    comparison_results: Dict[str, Any],
                                    density_type: str,
                                    output_path: Optional[Path] = None) -> plt.Figure:
        """
        Create detailed comparison plot for a specific density type.
        
        Args:
            comparison_results: Results from statistical analysis
            density_type: Specific density type to visualize
            output_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=self.config['visualization']['figsize_medium'])
        fig.suptitle(f'{density_type.upper()} Statistical Comparison',
                    fontsize=self.config['visualization']['typography']['title_fontsize'],
                    fontweight='bold')
        
        # Get density-specific results
        density_results = comparison_results.get('by_density_type', {}).get(density_type, {})
        
        if not density_results:
            fig.text(0.5, 0.5, f'No results available for {density_type}',
                    ha='center', va='center')
            return fig
        
        # 1. Model indices vs ground truth
        ax = axes[0, 0]
        self._plot_indices_comparison(ax, density_results)
        
        # 2. Error metrics
        ax = axes[0, 1]
        self._plot_error_bars(ax, density_results)
        
        # 3. Statistical test results
        ax = axes[0, 2]
        self._plot_test_results(ax, density_results)
        
        # 4. Bland-Altman plot
        ax = axes[1, 0]
        self._plot_bland_altman(ax, density_results)
        
        # 5. Residual distribution
        ax = axes[1, 1]
        self._plot_residual_distribution(ax, density_results)
        
        # 6. Performance metrics table
        ax = axes[1, 2]
        self._plot_metrics_table(ax, density_results)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            self._save_figure(fig, output_path, f'{density_type}_comparison')
        
        return fig
    
    def _plot_indices_comparison(self, ax: plt.Axes, density_results: Dict[str, Any]):
        """Plot model indices against ground truth reference."""
        model_indices = density_results.get('model_indices', {})
        reference = density_results.get('reference_value', 0)
        
        if not model_indices:
            ax.text(0.5, 0.5, 'No index data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Model Indices')
            return
        
        models = list(model_indices.keys())
        values = list(model_indices.values())
        
        x = np.arange(len(models))
        colors = self.config['visualization']['colors']['model_palette']
        
        bars = ax.bar(x, values, color=colors[:len(models)], alpha=0.7)
        ax.axhline(y=reference, color='red', linestyle='--', linewidth=2,
                  label=f'Ground Truth ({reference:.2f})')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Area Index')
        ax.set_title('Model Indices vs Ground Truth')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom')
    
    def _plot_error_bars(self, ax: plt.Axes, density_results: Dict[str, Any]):
        """Plot error metrics with error bars."""
        error_metrics = density_results.get('error_metrics', {})
        
        if not error_metrics:
            ax.text(0.5, 0.5, 'No error metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error Metrics')
            return
        
        models = list(error_metrics.keys())
        mae_values = [error_metrics[m]['mae'] for m in models]
        rmse_values = [error_metrics[m]['rmse'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, mae_values, width, label='MAE', alpha=0.7)
        ax.bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Error')
        ax.set_title('Error Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_test_results(self, ax: plt.Axes, density_results: Dict[str, Any]):
        """Plot statistical test results."""
        tests = density_results.get('statistical_tests', {})
        
        if not tests:
            ax.text(0.5, 0.5, 'No test results available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Statistical Tests')
            return
        
        # Extract individual model tests
        individual = tests.get('individual_tests', {})
        
        if individual:
            models = list(individual.keys())
            percent_diff = [individual[m]['percent_difference'] for m in models]
            
            x = np.arange(len(models))
            colors = ['green' if abs(p) < 10 else 'orange' if abs(p) < 20 else 'red' 
                     for p in percent_diff]
            
            bars = ax.bar(x, percent_diff, color=colors, alpha=0.7)
            
            # Add reference lines
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.axhline(y=10, color='green', linestyle='--', alpha=0.5)
            ax.axhline(y=-10, color='green', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('% Difference from Ground Truth')
            ax.set_title('Model Performance vs Ground Truth')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, percent_diff):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.1f}%', ha='center', 
                       va='bottom' if val > 0 else 'top')
    
    def _plot_bland_altman(self, ax: plt.Axes, density_results: Dict[str, Any]):
        """Create Bland-Altman plot for agreement analysis."""
        model_indices = density_results.get('model_indices', {})
        reference = density_results.get('reference_value', 0)
        
        if not model_indices:
            ax.text(0.5, 0.5, 'No data for Bland-Altman plot',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Bland-Altman Plot')
            return
        
        # Calculate means and differences
        means = []
        differences = []
        labels = []
        
        for model, value in model_indices.items():
            mean_val = (value + reference) / 2
            diff = value - reference
            means.append(mean_val)
            differences.append(diff)
            labels.append(model)
        
        # Plot
        colors = self.config['visualization']['colors']['model_palette']
        for i, (m, d, l) in enumerate(zip(means, differences, labels)):
            ax.scatter(m, d, s=100, color=colors[i % len(colors)],
                      label=l, alpha=0.7, edgecolors='black')
        
        # Add mean difference and limits of agreement
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        ax.axhline(y=mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.2f}')
        ax.axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--',
                  label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.2f}')
        ax.axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--',
                  label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.2f}')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Mean of Model and Reference')
        ax.set_ylabel('Difference (Model - Reference)')
        ax.set_title('Bland-Altman Agreement Plot')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    def _plot_residual_distribution(self, ax: plt.Axes, density_results: Dict[str, Any]):
        """Plot distribution of residuals."""
        model_indices = density_results.get('model_indices', {})
        reference = density_results.get('reference_value', 0)
        
        if not model_indices:
            ax.text(0.5, 0.5, 'No residual data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Residual Distribution')
            return
        
        # Calculate residuals
        residuals = [value - reference for value in model_indices.values()]
        models = list(model_indices.keys())
        
        # Create violin plot
        parts = ax.violinplot([residuals], positions=[0], widths=0.7,
                             showmeans=True, showextrema=True)
        
        # Color the violin plot
        for pc in parts['bodies']:
            pc.set_facecolor('#4A90E2')
            pc.set_alpha(0.7)
        
        # Add individual points
        y_pos = np.random.normal(0, 0.04, size=len(residuals))
        colors = self.config['visualization']['colors']['model_palette']
        
        for i, (r, m) in enumerate(zip(residuals, models)):
            ax.scatter(y_pos[i], r, s=100, color=colors[i % len(colors)],
                      label=m, alpha=0.7, edgecolors='black')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('Residual (Model - Reference)')
        ax.set_title('Distribution of Residuals')
        ax.set_xticks([])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_metrics_table(self, ax: plt.Axes, density_results: Dict[str, Any]):
        """Create metrics summary table."""
        ax.axis('tight')
        ax.axis('off')
        
        error_metrics = density_results.get('error_metrics', {})
        
        if not error_metrics:
            ax.text(0.5, 0.5, 'No metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Create table data
        headers = ['Model', 'MAE', 'RMSE', 'Bias', '% Error', 'Rank']
        table_data = [headers]
        
        for model, metrics in error_metrics.items():
            row = [
                model[:10],  # Truncate long names
                f"{metrics['mae']:.3f}",
                f"{metrics['rmse']:.3f}",
                f"{metrics['bias']:.3f}",
                f"{metrics['percent_error']:.1f}%",
                str(metrics.get('rank', '-'))
            ]
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data,
                        cellLoc='center',
                        loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color cells based on rank
        for i in range(1, len(table_data)):
            rank = int(table_data[i][-1]) if table_data[i][-1] != '-' else 999
            if rank == 1:
                color = '#2ECC71'  # Green for best
            elif rank == 2:
                color = '#F39C12'  # Orange for second
            else:
                color = '#F8F9FA'  # Light gray for others
            
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
        
        ax.set_title('Performance Metrics Summary', fontweight='bold', pad=10)
    
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
            fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                       dpi=self.config['visualization']['dpi'])
            self.logger.info(f"Saved figure to {filepath}")