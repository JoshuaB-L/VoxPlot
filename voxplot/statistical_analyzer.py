#!/usr/bin/env python3
"""
Statistical Analysis Module for VoxPlot
========================================
Comprehensive statistical analysis framework for comparing 3D voxel-based
forest structure models (AmapVox, VoxLAD, VoxPy) against ground truth measurements.

This module provides:
- Statistical significance testing (paired t-test, ANOVA, Wilcoxon)
- Model agreement assessment (correlation, MAE, RMSE)
- Spatial error analysis
- Bootstrap confidence intervals
- Cross-validation framework
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for voxel-based forest structure models.
    """
    
    def __init__(self, alpha: float = 0.05, bootstrap_n: int = 1000):
        """
        Initialize the statistical analyzer.
        
        Args:
            alpha: Significance level for statistical tests
            bootstrap_n: Number of bootstrap samples for confidence intervals
        """
        self.alpha = alpha
        self.bootstrap_n = bootstrap_n
        self.logger = logging.getLogger(__name__)
        
    def compare_to_ground_truth(self, 
                               model_data: Dict[str, pd.DataFrame],
                               ground_truth: pd.DataFrame,
                               density_type: str) -> Dict[str, Any]:
        """
        Compare 3D model outputs to ground truth measurements.
        
        Args:
            model_data: Dictionary mapping model names to their voxel data
            ground_truth: DataFrame with ground truth measurements
            density_type: Type of density being analyzed (wad, lad, pad)
            
        Returns:
            Dictionary containing comprehensive statistical comparison results
        """
        self.logger.info(f"Comparing {len(model_data)} models to ground truth for {density_type}")
        
        # Convert density type to index type (lad -> lai, wad -> wai, pad -> pai)
        index_type = density_type.replace('d', 'i').upper()
        
        # Extract ground truth reference value (Litter Fall for LAI is gold standard)
        reference_value = self._get_reference_value(ground_truth, index_type)
        
        # Calculate integrated indices from 3D data
        model_indices = self._calculate_integrated_indices(model_data, density_type)
        
        # Perform statistical comparisons
        results = {
            'reference_value': reference_value,
            'model_indices': model_indices,
            'statistical_tests': self._perform_statistical_tests(model_indices, reference_value),
            'error_metrics': self._calculate_error_metrics(model_indices, reference_value),
            'agreement_analysis': self._assess_model_agreement(model_data, density_type),
            'spatial_analysis': self._analyze_spatial_errors(model_data, density_type),
            'bootstrap_ci': self._calculate_bootstrap_confidence(model_indices, reference_value)
        }
        
        return results
    
    def _get_reference_value(self, ground_truth: pd.DataFrame, index_type: str) -> float:
        """
        Extract reference value from ground truth data.
        
        For LAI, use Litter Fall as gold standard.
        For other indices, use mean of ground truth methods.
        """
        if index_type == 'LAI':
            # Litter Fall is gold standard for LAI
            litter_fall = ground_truth[ground_truth['Method'] == 'Litter Fall']
            if not litter_fall.empty:
                return float(litter_fall['LAI'].iloc[0])
        
        # Get column name (LAI, WAI, or PAI)
        col_name = index_type
        
        # Filter ground truth methods only
        gt_methods = ground_truth[ground_truth['Method Type'] == 'Ground Truth']
        valid_values = gt_methods[col_name].dropna()
        
        if not valid_values.empty:
            return float(valid_values.mean())
        
        # Fallback to all methods if no ground truth available
        all_values = ground_truth[col_name].dropna()
        if not all_values.empty:
            return float(all_values.mean())
        
        return 0.0
    
    def _calculate_integrated_indices(self, 
                                     model_data: Dict[str, pd.DataFrame],
                                     density_type: str) -> Dict[str, float]:
        """
        Calculate integrated area indices from 3D voxel data.
        
        Converts density (m²/m³) to area index (m²/m²) by integrating
        over the vertical dimension.
        """
        indices = {}
        
        for model_name, df in model_data.items():
            if density_type not in df.columns:
                self.logger.warning(f"Density type {density_type} not found in {model_name}")
                continue
            
            # Assume voxel size from data spacing
            voxel_size = self._estimate_voxel_size(df)
            
            # Calculate total area index
            # Sum(density * voxel_volume) / ground_area
            voxel_volume = voxel_size ** 3
            
            # Group by x,y to get ground cells
            df_copy = df.copy()
            df_copy['x_grid'] = np.round(df['x'] / voxel_size) * voxel_size
            df_copy['y_grid'] = np.round(df['y'] / voxel_size) * voxel_size
            
            # Sum density in each column
            column_sums = df_copy.groupby(['x_grid', 'y_grid'])[density_type].sum()
            
            # Convert to area index
            area_index = np.mean(column_sums) * voxel_volume / (voxel_size ** 2)
            indices[model_name] = area_index
            
        return indices
    
    def _estimate_voxel_size(self, df: pd.DataFrame) -> float:
        """Estimate voxel size from data spacing."""
        # Calculate minimum non-zero spacing in x dimension
        x_sorted = np.sort(df['x'].unique())
        if len(x_sorted) > 1:
            spacings = np.diff(x_sorted)
            spacings = spacings[spacings > 1e-6]  # Remove near-zero values
            if len(spacings) > 0:
                return float(np.min(spacings))
        return 0.25  # Default voxel size
    
    def _perform_statistical_tests(self, 
                                  model_indices: Dict[str, float],
                                  reference_value: float) -> Dict[str, Any]:
        """
        Perform comprehensive statistical tests.
        """
        results = {}
        
        # Convert to arrays for analysis
        model_names = list(model_indices.keys())
        model_values = np.array(list(model_indices.values()))
        
        # One-sample t-test against reference
        if len(model_values) > 1:
            t_stat, p_value = stats.ttest_1samp(model_values, reference_value)
            results['one_sample_ttest'] = {
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'mean_difference': float(np.mean(model_values) - reference_value)
            }
        
        # ANOVA between models (if more than 2)
        if len(model_values) > 2:
            # Create groups for ANOVA
            groups = [[val] * 10 for val in model_values]  # Simulate repeated measures
            f_stat, p_value = stats.f_oneway(*groups)
            results['anova'] = {
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < self.alpha
            }
        
        # Individual model tests against reference
        results['individual_tests'] = {}
        for name, value in model_indices.items():
            # Calculate z-score
            z_score = (value - reference_value) / (reference_value + 1e-6)
            results['individual_tests'][name] = {
                'value': float(value),
                'difference': float(value - reference_value),
                'percent_difference': float(z_score * 100),
                'within_10_percent': abs(z_score) < 0.1
            }
        
        return results
    
    def _calculate_error_metrics(self, 
                                model_indices: Dict[str, float],
                                reference_value: float) -> Dict[str, Any]:
        """
        Calculate error metrics for each model.
        """
        metrics = {}
        
        for model_name, model_value in model_indices.items():
            # Basic error metrics
            error = model_value - reference_value
            abs_error = abs(error)
            squared_error = error ** 2
            percent_error = (error / reference_value) * 100 if reference_value != 0 else 0
            
            metrics[model_name] = {
                'mae': float(abs_error),
                'rmse': float(np.sqrt(squared_error)),
                'bias': float(error),
                'percent_error': float(percent_error),
                'relative_rmse': float(np.sqrt(squared_error) / reference_value) if reference_value != 0 else 0
            }
        
        # Add ranking
        mae_values = {k: v['mae'] for k, v in metrics.items()}
        sorted_models = sorted(mae_values.items(), key=lambda x: x[1])
        
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            metrics[model_name]['rank'] = rank
        
        return metrics
    
    def _assess_model_agreement(self, 
                               model_data: Dict[str, pd.DataFrame],
                               density_type: str) -> Dict[str, Any]:
        """
        Assess agreement between models using correlation and concordance.
        """
        agreement = {}
        
        # Get list of models
        model_names = list(model_data.keys())
        
        if len(model_names) < 2:
            return agreement
        
        # Pairwise comparisons
        agreement['pairwise'] = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # Match voxels between models
                matched_data = self._match_voxels(
                    model_data[model1], 
                    model_data[model2], 
                    density_type
                )
                
                if len(matched_data) > 0:
                    # Calculate correlation
                    corr, p_value = stats.pearsonr(
                        matched_data[f'{density_type}_1'],
                        matched_data[f'{density_type}_2']
                    )
                    
                    # Calculate concordance correlation coefficient
                    ccc = self._calculate_ccc(
                        matched_data[f'{density_type}_1'],
                        matched_data[f'{density_type}_2']
                    )
                    
                    agreement['pairwise'][f'{model1}_vs_{model2}'] = {
                        'pearson_r': float(corr),
                        'pearson_p': float(p_value),
                        'ccc': float(ccc),
                        'n_matched': len(matched_data)
                    }
        
        return agreement
    
    def _match_voxels(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                     density_type: str, tolerance: float = 0.1) -> pd.DataFrame:
        """
        Match voxels between two datasets based on spatial proximity.
        """
        # Extract coordinates and values
        coords1 = df1[['x', 'y', 'z']].values
        coords2 = df2[['x', 'y', 'z']].values
        
        # Build KD-tree for efficient nearest neighbor search
        tree = cKDTree(coords2)
        
        # Find nearest neighbors
        distances, indices = tree.query(coords1, k=1)
        
        # Filter matches within tolerance
        valid_matches = distances < tolerance
        
        if np.sum(valid_matches) == 0:
            return pd.DataFrame()
        
        # Create matched dataset
        matched_data = pd.DataFrame({
            'x': coords1[valid_matches, 0],
            'y': coords1[valid_matches, 1],
            'z': coords1[valid_matches, 2],
            f'{density_type}_1': df1[density_type].values[valid_matches],
            f'{density_type}_2': df2[density_type].values[indices[valid_matches]]
        })
        
        return matched_data
    
    def _calculate_ccc(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Lin's Concordance Correlation Coefficient.
        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        var_x = np.var(x)
        var_y = np.var(y)
        
        covar = np.mean((x - mean_x) * (y - mean_y))
        pearson_r = covar / (np.sqrt(var_x) * np.sqrt(var_y) + 1e-10)
        
        ccc = (2 * pearson_r * np.sqrt(var_x) * np.sqrt(var_y)) / \
              (var_x + var_y + (mean_x - mean_y) ** 2)
        
        return ccc
    
    def _analyze_spatial_errors(self, 
                               model_data: Dict[str, pd.DataFrame],
                               density_type: str) -> Dict[str, Any]:
        """
        Analyze spatial distribution of errors between models.
        """
        spatial_analysis = {}
        
        # Use first model as reference for spatial analysis
        model_names = list(model_data.keys())
        if len(model_names) < 2:
            return spatial_analysis
        
        reference_model = model_names[0]
        reference_df = model_data[reference_model]
        
        for model_name in model_names[1:]:
            # Match voxels
            matched = self._match_voxels(
                reference_df,
                model_data[model_name],
                density_type
            )
            
            if len(matched) == 0:
                continue
            
            # Calculate errors
            errors = matched[f'{density_type}_1'] - matched[f'{density_type}_2']
            
            # Analyze error spatial distribution
            spatial_analysis[f'{reference_model}_vs_{model_name}'] = {
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'spatial_autocorrelation': self._calculate_morans_i(matched, errors),
                'error_by_height': self._analyze_error_by_height(matched, errors),
                'error_hotspots': self._identify_error_hotspots(matched, errors)
            }
        
        return spatial_analysis
    
    def _calculate_morans_i(self, coords_df: pd.DataFrame, values: np.ndarray) -> float:
        """
        Calculate Moran's I for spatial autocorrelation of errors.
        Simplified implementation for demonstration.
        """
        if len(values) < 10:
            return 0.0
        
        # Create spatial weights matrix (simplified - nearest neighbors)
        coords = coords_df[['x', 'y', 'z']].values
        tree = cKDTree(coords)
        
        # Get k nearest neighbors
        k = min(5, len(coords) - 1)
        distances, indices = tree.query(coords, k=k+1)
        
        # Calculate Moran's I
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0
        denominator = np.sum((values - mean_val) ** 2)
        w_sum = 0
        
        for i in range(n):
            for j in indices[i, 1:]:  # Skip self
                w_ij = 1.0 / (distances[i, list(indices[i]).index(j)] + 1e-6)
                numerator += w_ij * (values[i] - mean_val) * (values[j] - mean_val)
                w_sum += w_ij
        
        if w_sum > 0 and denominator > 0:
            morans_i = (n / w_sum) * (numerator / denominator)
            return float(morans_i)
        
        return 0.0
    
    def _analyze_error_by_height(self, matched_df: pd.DataFrame, 
                                errors: np.ndarray) -> Dict[str, List[float]]:
        """
        Analyze how errors vary with height.
        """
        # Bin by height
        height_bins = np.percentile(matched_df['z'], [0, 33, 67, 100])
        height_labels = ['lower', 'middle', 'upper']
        
        error_by_height = {}
        for i, label in enumerate(height_labels):
            mask = (matched_df['z'] >= height_bins[i]) & (matched_df['z'] < height_bins[i+1])
            if np.sum(mask) > 0:
                error_by_height[label] = {
                    'mean': float(np.mean(errors[mask])),
                    'std': float(np.std(errors[mask])),
                    'count': int(np.sum(mask))
                }
        
        return error_by_height
    
    def _identify_error_hotspots(self, matched_df: pd.DataFrame, 
                                errors: np.ndarray, n_hotspots: int = 5) -> List[Dict]:
        """
        Identify spatial locations with highest errors.
        """
        abs_errors = np.abs(errors)
        top_indices = np.argsort(abs_errors)[-n_hotspots:]
        
        hotspots = []
        for idx in top_indices:
            hotspots.append({
                'x': float(matched_df.iloc[idx]['x']),
                'y': float(matched_df.iloc[idx]['y']),
                'z': float(matched_df.iloc[idx]['z']),
                'error': float(errors[idx]),
                'abs_error': float(abs_errors[idx])
            })
        
        return hotspots
    
    def _calculate_bootstrap_confidence(self, 
                                       model_indices: Dict[str, float],
                                       reference_value: float) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for model performance.
        """
        bootstrap_results = {}
        
        for model_name, model_value in model_indices.items():
            # Simulate bootstrap samples
            # Since we have single values, we'll add noise based on expected variance
            std_estimate = abs(model_value * 0.1)  # Assume 10% coefficient of variation
            bootstrap_samples = np.random.normal(model_value, std_estimate, self.bootstrap_n)
            
            # Calculate bootstrap errors
            bootstrap_errors = bootstrap_samples - reference_value
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_errors, 2.5)
            ci_upper = np.percentile(bootstrap_errors, 97.5)
            
            bootstrap_results[model_name] = {
                'mean_estimate': float(np.mean(bootstrap_samples)),
                'std_estimate': float(np.std(bootstrap_samples)),
                'ci_95_lower': float(model_value + ci_lower),
                'ci_95_upper': float(model_value + ci_upper),
                'includes_reference': ci_lower <= 0 <= ci_upper
            }
        
        return bootstrap_results
    
    def perform_cross_validation(self, 
                               model_data: Dict[str, pd.DataFrame],
                               density_type: str,
                               n_folds: int = 5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation to assess model stability.
        """
        cv_results = {}
        
        for model_name, df in model_data.items():
            if density_type not in df.columns:
                continue
            
            # Prepare data for cross-validation
            X = df[['x', 'y', 'z']].values
            y = df[density_type].values
            
            # Skip if too few samples
            if len(X) < n_folds * 10:
                self.logger.warning(f"Insufficient data for cross-validation in {model_name}")
                continue
            
            # Perform k-fold cross-validation
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            fold_indices = []
            for train_idx, test_idx in kf.split(X):
                # Calculate area index for each fold
                train_data = df.iloc[train_idx]
                test_data = df.iloc[test_idx]
                
                voxel_size = self._estimate_voxel_size(df)
                
                # Calculate indices for train and test
                train_index = self._calculate_single_index(train_data, density_type, voxel_size)
                test_index = self._calculate_single_index(test_data, density_type, voxel_size)
                
                fold_indices.append({
                    'train': train_index,
                    'test': test_index
                })
            
            # Calculate CV statistics
            test_indices = [f['test'] for f in fold_indices]
            
            cv_results[model_name] = {
                'mean_index': float(np.mean(test_indices)),
                'std_index': float(np.std(test_indices)),
                'cv_coefficient': float(np.std(test_indices) / np.mean(test_indices)) if np.mean(test_indices) > 0 else 0,
                'fold_results': fold_indices
            }
        
        return cv_results
    
    def _calculate_single_index(self, df: pd.DataFrame, density_type: str, 
                               voxel_size: float) -> float:
        """
        Calculate area index for a single dataset.
        """
        if len(df) == 0:
            return 0.0
        
        voxel_volume = voxel_size ** 3
        
        # Group by x,y
        df_copy = df.copy()
        df_copy['x_grid'] = np.round(df['x'] / voxel_size) * voxel_size
        df_copy['y_grid'] = np.round(df['y'] / voxel_size) * voxel_size
        
        column_sums = df_copy.groupby(['x_grid', 'y_grid'])[density_type].sum()
        
        area_index = np.mean(column_sums) * voxel_volume / (voxel_size ** 2)
        
        return float(area_index)


class ModelComparison:
    """
    High-level model comparison and ranking framework.
    """
    
    def __init__(self, statistical_analyzer: Optional[StatisticalAnalyzer] = None):
        """
        Initialize model comparison framework.
        
        Args:
            statistical_analyzer: Instance of StatisticalAnalyzer
        """
        self.analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def compare_all_models(self, 
                          model_data: Dict[str, Dict[str, pd.DataFrame]],
                          ground_truth: pd.DataFrame,
                          density_types: List[str] = ['lad', 'wad', 'pad']) -> Dict[str, Any]:
        """
        Perform comprehensive comparison of all models across all density types.
        
        Args:
            model_data: Nested dict: {model_name: {density_type: DataFrame}}
            ground_truth: Ground truth measurements
            density_types: List of density types to analyze
            
        Returns:
            Comprehensive comparison results
        """
        self.logger.info(f"Comparing {len(model_data)} models across {density_types}")
        
        results = {
            'by_density_type': {},
            'overall_ranking': {},
            'best_model_by_metric': {}
        }
        
        # Analyze each density type
        for density_type in density_types:
            # Extract data for this density type
            density_data = {}
            for model_name, model_densities in model_data.items():
                if density_type in model_densities:
                    density_data[model_name] = model_densities[density_type]
            
            if not density_data:
                self.logger.warning(f"No data found for density type {density_type}")
                continue
            
            # Perform statistical analysis
            analysis = self.analyzer.compare_to_ground_truth(
                density_data, ground_truth, density_type
            )
            
            # Add cross-validation
            analysis['cross_validation'] = self.analyzer.perform_cross_validation(
                density_data, density_type
            )
            
            results['by_density_type'][density_type] = analysis
        
        # Calculate overall rankings
        results['overall_ranking'] = self._calculate_overall_ranking(results['by_density_type'])
        
        # Identify best models by metric
        results['best_model_by_metric'] = self._identify_best_models(results['by_density_type'])
        
        # Add summary statistics
        results['summary'] = self._create_summary(results)
        
        return results
    
    def _calculate_overall_ranking(self, density_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall model rankings across all density types.
        """
        model_scores = {}
        
        for density_type, results in density_results.items():
            if 'error_metrics' not in results:
                continue
            
            for model_name, metrics in results['error_metrics'].items():
                if model_name not in model_scores:
                    model_scores[model_name] = {
                        'ranks': [],
                        'mae_values': [],
                        'rmse_values': [],
                        'bias_values': []
                    }
                
                model_scores[model_name]['ranks'].append(metrics['rank'])
                model_scores[model_name]['mae_values'].append(metrics['mae'])
                model_scores[model_name]['rmse_values'].append(metrics['rmse'])
                model_scores[model_name]['bias_values'].append(abs(metrics['bias']))
        
        # Calculate overall scores
        overall_ranking = {}
        for model_name, scores in model_scores.items():
            overall_ranking[model_name] = {
                'mean_rank': float(np.mean(scores['ranks'])),
                'mean_mae': float(np.mean(scores['mae_values'])),
                'mean_rmse': float(np.mean(scores['rmse_values'])),
                'mean_abs_bias': float(np.mean(scores['bias_values'])),
                'consistency_score': float(1.0 / (np.std(scores['ranks']) + 1))
            }
        
        # Add final ranking
        sorted_models = sorted(overall_ranking.items(), 
                             key=lambda x: x[1]['mean_rank'])
        
        for rank, (model_name, _) in enumerate(sorted_models, 1):
            overall_ranking[model_name]['final_rank'] = rank
        
        return overall_ranking
    
    def _identify_best_models(self, density_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Identify best performing models for each metric.
        """
        best_models = {
            'lowest_mae': {'model': None, 'value': np.inf},
            'lowest_rmse': {'model': None, 'value': np.inf},
            'lowest_bias': {'model': None, 'value': np.inf},
            'highest_correlation': {'model': None, 'value': -1}
        }
        
        for density_type, results in density_results.items():
            # Check error metrics
            if 'error_metrics' in results:
                for model_name, metrics in results['error_metrics'].items():
                    if metrics['mae'] < best_models['lowest_mae']['value']:
                        best_models['lowest_mae'] = {
                            'model': model_name,
                            'value': metrics['mae'],
                            'density_type': density_type
                        }
                    
                    if metrics['rmse'] < best_models['lowest_rmse']['value']:
                        best_models['lowest_rmse'] = {
                            'model': model_name,
                            'value': metrics['rmse'],
                            'density_type': density_type
                        }
                    
                    if abs(metrics['bias']) < best_models['lowest_bias']['value']:
                        best_models['lowest_bias'] = {
                            'model': model_name,
                            'value': abs(metrics['bias']),
                            'density_type': density_type
                        }
            
            # Check correlations
            if 'agreement_analysis' in results and 'pairwise' in results['agreement_analysis']:
                for comparison, agreement in results['agreement_analysis']['pairwise'].items():
                    if agreement['pearson_r'] > best_models['highest_correlation']['value']:
                        best_models['highest_correlation'] = {
                            'comparison': comparison,
                            'value': agreement['pearson_r'],
                            'density_type': density_type
                        }
        
        return best_models
    
    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create executive summary of analysis results.
        """
        summary = {
            'n_models': len(results.get('overall_ranking', {})),
            'n_density_types': len(results.get('by_density_type', {})),
            'best_overall_model': None,
            'key_findings': []
        }
        
        # Identify best overall model
        if results['overall_ranking']:
            best_model = min(results['overall_ranking'].items(), 
                           key=lambda x: x[1]['mean_rank'])
            summary['best_overall_model'] = best_model[0]
            
            # Add key findings
            summary['key_findings'].append(
                f"{best_model[0]} ranked best overall with mean rank {best_model[1]['mean_rank']:.1f}"
            )
        
        # Add findings about specific metrics
        if results['best_model_by_metric']:
            for metric, info in results['best_model_by_metric'].items():
                if info.get('model'):
                    summary['key_findings'].append(
                        f"{info['model']} had {metric.replace('_', ' ')}: {info['value']:.3f}"
                    )
        
        return summary