#!/usr/bin/env python3
"""
Data analysis module for VoxPlot.

This module provides comprehensive analysis functions for voxel-based forest structure data,
including crown layer analysis, spatial distribution analysis, and statistical comparisons.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from utils import (
    filter_by_density_threshold, filter_by_height_range, calculate_crown_layer_bounds,
    create_2d_raster, aggregate_to_grid, calculate_statistics, safe_divide
)


class ForestStructureAnalyzer:
    """Analyzes forest structure from voxel-based density data."""
    
    def __init__(self, voxel_size: float = 0.25):
        self.voxel_size = voxel_size
        self.logger = logging.getLogger(__name__)
    
    def analyze_crown_layers(self, df: pd.DataFrame, density_type: str,
                           crown_base_height: float = 0.7, 
                           min_density: float = 0.05,
                           num_layers: int = 3) -> Dict[str, Any]:
        """
        Analyze crown layers and calculate density metrics for each layer.
        
        Args:
            df: DataFrame containing voxel data
            density_type: Type of density being analyzed (wad, lad, pad)
            crown_base_height: Minimum height to consider as crown
            min_density: Minimum density threshold
            num_layers: Number of crown layers to create
        
        Returns:
            Dictionary containing layer analysis results
        """
        self.logger.debug(f"Analyzing crown layers for {density_type}")
        
        # Filter data to crown region with valid density
        crown_df = filter_by_height_range(df, min_height=crown_base_height)
        crown_df = filter_by_density_threshold(crown_df, density_type, min_density)
        
        if len(crown_df) == 0:
            return self._empty_layer_analysis()
        
        # Calculate layer boundaries
        crown_height, layer_boundaries = calculate_crown_layer_bounds(
            crown_df, crown_base_height=crown_base_height, num_layers=num_layers
        )
        
        if crown_height <= 0:
            return self._empty_layer_analysis()
        
        # Create layer masks
        z_min = crown_df['z'].min()
        z_max = crown_df['z'].max()
        
        layer_masks = self._create_layer_masks(crown_df, layer_boundaries, z_min, z_max)
        
        # Analyze each layer
        layer_metrics = {}
        layer_rasters = {}
        
        # Analyze whole crown
        whole_metrics, whole_raster = self._analyze_single_layer(crown_df, density_type, "whole")
        layer_metrics['whole'] = whole_metrics
        layer_rasters['whole'] = whole_raster
        
        # Analyze individual layers
        layer_names = ['lower', 'middle', 'upper']
        for i, (layer_name, mask) in enumerate(zip(layer_names, layer_masks)):
            layer_df = crown_df[mask]
            metrics, raster = self._analyze_single_layer(layer_df, density_type, layer_name)
            layer_metrics[layer_name] = metrics
            layer_rasters[layer_name] = raster
        
        # Create vertical profile slice
        zx_slice = self._create_vertical_slice(crown_df, density_type)
        
        return {
            'metrics': layer_metrics,
            'rasters': layer_rasters,
            'zx_slice': zx_slice,
            'layer_boundaries': layer_boundaries,
            'crown_bounds': {'z_min': z_min, 'z_max': z_max},
            'crown_height': crown_height,
            'total_voxels': len(crown_df)
        }
    
    def _create_layer_masks(self, crown_df: pd.DataFrame, layer_boundaries: List[float],
                           z_min: float, z_max: float) -> List[np.ndarray]:
        """Create boolean masks for each crown layer."""
        if len(layer_boundaries) < 2:
            # If we don't have enough boundaries, create equal thirds
            layer_height = (z_max - z_min) / 3
            lower_bound = z_min + layer_height
            middle_bound = z_min + 2 * layer_height
        else:
            lower_bound, middle_bound = layer_boundaries[:2]
        
        lower_mask = crown_df['z'] <= lower_bound
        middle_mask = (crown_df['z'] > lower_bound) & (crown_df['z'] <= middle_bound)
        upper_mask = crown_df['z'] > middle_bound
        
        return [lower_mask, middle_mask, upper_mask]
    
    def _analyze_single_layer(self, layer_df: pd.DataFrame, density_type: str, 
                             layer_name: str) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        """
        Analyze a single crown layer and create density metrics and raster maps.
        
        This method creates proper 2D raster maps where each pixel represents
        the density metric (LAI, WAI, PAI) for that ground area.
        """
        if len(layer_df) == 0:
            return self._empty_layer_metrics(), None
        
        # Calculate voxel volume
        voxel_volume = self.voxel_size ** 3
        cell_area = self.voxel_size ** 2
        
        # Create 2D raster of density values
        # First aggregate density values to grid cells
        aggregated_df = aggregate_to_grid(
            layer_df, self.voxel_size, density_type, 'sum'
        )
        
        if len(aggregated_df) == 0:
            return self._empty_layer_metrics(), None
        
        # Convert density sums to area index values (LAI, WAI, PAI)
        # For each grid cell: (sum of density * voxel_volume) / cell_area
        aggregated_df['area_index'] = (aggregated_df[density_type] * voxel_volume) / cell_area
        
        # Create 2D raster array
        raster_array, x_coords, y_coords = create_2d_raster(
            aggregated_df, self.voxel_size, 'area_index'
        )
        
        # Calculate total metrics
        total_area = np.sum(aggregated_df[density_type]) * voxel_volume  # Total leaf/wood/plant area
        ground_area = len(aggregated_df) * cell_area  # Total ground area covered
        mean_area_index = safe_divide(total_area, ground_area)  # Mean area index across ground area
        
        # Additional statistics
        area_index_stats = calculate_statistics(aggregated_df['area_index'])
        density_stats = calculate_statistics(layer_df[density_type])
        
        metrics = {
            'total_area': total_area,
            'ground_area': ground_area,
            'mean_area_index': mean_area_index,
            'max_area_index': area_index_stats['max'],
            'std_area_index': area_index_stats['std'],
            'voxel_count': len(layer_df),
            'grid_cells': len(aggregated_df),
            'mean_density': density_stats['mean'],
            'max_density': density_stats['max']
        }
        
        raster_data = {
            'array': raster_array,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'extent': (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
            'cell_size': self.voxel_size
        } if len(raster_array) > 0 else None
        
        return metrics, raster_data
    
    def _create_vertical_slice(self, crown_df: pd.DataFrame, density_type: str) -> Optional[Dict[str, Any]]:
        """Create a vertical (ZX) slice through the crown for profile visualization."""
        if len(crown_df) == 0:
            return None
        
        # Use mean Y coordinate for the slice
        y_mean = (crown_df['y'].max() + crown_df['y'].min()) / 2
        y_slice_width = self.voxel_size * 2
        
        # Filter data within slice width
        slice_mask = (crown_df['y'] >= y_mean - y_slice_width) & (crown_df['y'] <= y_mean + y_slice_width)
        slice_data = crown_df[slice_mask]
        
        if len(slice_data) == 0:
            return None
        
        # Aggregate to ZX grid
        slice_data_copy = slice_data.copy()
        slice_data_copy['z_grid'] = np.round(slice_data_copy['z'] / self.voxel_size) * self.voxel_size
        slice_data_copy['x_grid'] = np.round(slice_data_copy['x'] / self.voxel_size) * self.voxel_size
        
        zx_aggregated = slice_data_copy.groupby(['z_grid', 'x_grid'])[density_type].mean().reset_index()
        
        # Create 2D array for ZX slice
        z_coords = np.sort(zx_aggregated['z_grid'].unique())
        x_coords = np.sort(zx_aggregated['x_grid'].unique())
        
        zx_array = np.zeros((len(z_coords), len(x_coords)))
        
        for _, row in zx_aggregated.iterrows():
            z_idx = np.where(z_coords == row['z_grid'])[0][0]
            x_idx = np.where(x_coords == row['x_grid'])[0][0]
            zx_array[z_idx, x_idx] = row[density_type]
        
        return {
            'array': zx_array,
            'z_coords': z_coords,
            'x_coords': x_coords,
            'extent': (x_coords.min(), x_coords.max(), z_coords.min(), z_coords.max()),
            'y_center': y_mean,
            'slice_width': y_slice_width * 2
        }
    
    def _empty_layer_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'metrics': {
                'whole': self._empty_layer_metrics(),
                'upper': self._empty_layer_metrics(),
                'middle': self._empty_layer_metrics(),
                'lower': self._empty_layer_metrics()
            },
            'rasters': {
                'whole': None, 'upper': None, 'middle': None, 'lower': None
            },
            'zx_slice': None,
            'layer_boundaries': [],
            'crown_bounds': {'z_min': 0, 'z_max': 0},
            'crown_height': 0,
            'total_voxels': 0
        }
    
    def _empty_layer_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'total_area': 0.0,
            'ground_area': 0.0,
            'mean_area_index': 0.0,
            'max_area_index': 0.0,
            'std_area_index': 0.0,
            'voxel_count': 0,
            'grid_cells': 0,
            'mean_density': 0.0,
            'max_density': 0.0
        }
    
    def analyze_vertical_profile(self, df: pd.DataFrame, density_type: str,
                               crown_base_height: float = 0.7,
                               bin_size: float = 0.5) -> Dict[str, Any]:
        """
        Analyze vertical density profile.
        
        Args:
            df: DataFrame containing voxel data
            density_type: Type of density being analyzed
            crown_base_height: Minimum height to consider
            bin_size: Size of height bins for analysis
        
        Returns:
            Dictionary containing vertical profile data
        """
        # Filter to crown region with positive density
        crown_df = filter_by_height_range(df, min_height=crown_base_height)
        crown_df = filter_by_density_threshold(crown_df, density_type, 0.0)
        
        if len(crown_df) == 0:
            return {'heights': np.array([]), 'densities': np.array([]), 'counts': np.array([])}
        
        # Create height bins
        z_min = crown_df['z'].min()
        z_max = crown_df['z'].max()
        bins = np.arange(z_min, z_max + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size / 2
        
        # Calculate mean density for each height bin
        binned_densities = []
        binned_counts = []
        
        for i in range(len(bins) - 1):
            mask = (crown_df['z'] >= bins[i]) & (crown_df['z'] < bins[i + 1])
            bin_data = crown_df[mask]
            
            if len(bin_data) > 0:
                binned_densities.append(bin_data[density_type].mean())
                binned_counts.append(len(bin_data))
            else:
                binned_densities.append(0.0)
                binned_counts.append(0)
        
        return {
            'heights': bin_centers,
            'densities': np.array(binned_densities),
            'counts': np.array(binned_counts),
            'bin_size': bin_size,
            'height_range': (z_min, z_max)
        }
    
    def analyze_spatial_distribution(self, df: pd.DataFrame, density_type: str,
                                   crown_base_height: float = 0.7) -> Dict[str, Any]:
        """
        Analyze spatial distribution of density values.
        
        Args:
            df: DataFrame containing voxel data
            density_type: Type of density being analyzed
            crown_base_height: Minimum height to consider
        
        Returns:
            Dictionary containing spatial distribution analysis
        """
        # Filter data
        crown_df = filter_by_height_range(df, min_height=crown_base_height)
        nonzero_df = filter_by_density_threshold(crown_df, density_type, 0.0)
        
        if len(nonzero_df) == 0:
            return self._empty_spatial_analysis()
        
        # Create spatial raster
        raster_array, x_coords, y_coords = create_2d_raster(
            nonzero_df, self.voxel_size, density_type, aggregation_method='mean'
        )
        
        # Calculate spatial statistics
        spatial_stats = {
            'total_voxels': len(crown_df),
            'nonzero_voxels': len(nonzero_df),
            'density_coverage': len(nonzero_df) / len(crown_df) if len(crown_df) > 0 else 0,
            'spatial_extent': {
                'x_range': x_coords.max() - x_coords.min() if len(x_coords) > 0 else 0,
                'y_range': y_coords.max() - y_coords.min() if len(y_coords) > 0 else 0,
                'area': (x_coords.max() - x_coords.min()) * (y_coords.max() - y_coords.min()) if len(x_coords) > 0 and len(y_coords) > 0 else 0
            }
        }
        
        # Density statistics
        density_stats = calculate_statistics(nonzero_df[density_type])
        
        return {
            'raster': {
                'array': raster_array,
                'x_coords': x_coords,
                'y_coords': y_coords,
                'extent': (x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()) if len(x_coords) > 0 else (0, 0, 0, 0)
            },
            'spatial_stats': spatial_stats,
            'density_stats': density_stats
        }
    
    def _empty_spatial_analysis(self) -> Dict[str, Any]:
        """Return empty spatial analysis structure."""
        return {
            'raster': {
                'array': np.array([]),
                'x_coords': np.array([]),
                'y_coords': np.array([]),
                'extent': (0, 0, 0, 0)
            },
            'spatial_stats': {
                'total_voxels': 0,
                'nonzero_voxels': 0,
                'density_coverage': 0,
                'spatial_extent': {'x_range': 0, 'y_range': 0, 'area': 0}
            },
            'density_stats': {
                'count': 0, 'mean': 0, 'median': 0, 'std': 0,
                'min': 0, 'max': 0, 'q25': 0, 'q75': 0
            }
        }


class ComparativeAnalyzer:
    """Performs comparative analysis between multiple datasets."""
    
    def __init__(self, voxel_size: float = 0.25):
        self.voxel_size = voxel_size
        self.logger = logging.getLogger(__name__)
        self.forest_analyzer = ForestStructureAnalyzer(voxel_size)
    
    def compare_crown_metrics(self, datasets: List[pd.DataFrame], density_type: str,
                            crown_base_height: float = 0.7,
                            min_density: float = 0.05) -> Dict[str, Any]:
        """
        Compare crown layer metrics between multiple datasets.
        
        Args:
            datasets: List of DataFrames to compare
            density_type: Type of density being analyzed
            crown_base_height: Minimum height to consider as crown
            min_density: Minimum density threshold
        
        Returns:
            Dictionary containing comparative analysis results
        """
        self.logger.info(f"Comparing crown metrics for {len(datasets)} datasets")
        
        comparison_results = {
            'model_names': [],
            'layer_metrics': [],
            'layer_analyses': []
        }
        
        # Analyze each dataset
        for df in datasets:
            model_name = df['display_name'].iloc[0]
            
            # Perform crown layer analysis
            layer_analysis = self.forest_analyzer.analyze_crown_layers(
                df, density_type, crown_base_height, min_density
            )
            
            comparison_results['model_names'].append(model_name)
            comparison_results['layer_metrics'].append(layer_analysis['metrics'])
            comparison_results['layer_analyses'].append(layer_analysis)
        
        # Create comparison summary
        comparison_results['summary'] = self._create_metrics_summary(
            comparison_results['model_names'], 
            comparison_results['layer_metrics']
        )
        
        return comparison_results
    
    def compare_vertical_profiles(self, datasets: List[pd.DataFrame], density_type: str,
                                crown_base_height: float = 0.7,
                                bin_size: float = 0.5) -> Dict[str, Any]:
        """Compare vertical density profiles between datasets."""
        profiles = {
            'model_names': [],
            'profiles': []
        }
        
        for df in datasets:
            model_name = df['display_name'].iloc[0]
            profile = self.forest_analyzer.analyze_vertical_profile(
                df, density_type, crown_base_height, bin_size
            )
            
            profiles['model_names'].append(model_name)
            profiles['profiles'].append(profile)
        
        return profiles
    
    def compare_spatial_distributions(self, datasets: List[pd.DataFrame], density_type: str,
                                    crown_base_height: float = 0.7) -> Dict[str, Any]:
        """Compare spatial distributions between datasets."""
        distributions = {
            'model_names': [],
            'distributions': []
        }
        
        for df in datasets:
            model_name = df['display_name'].iloc[0]
            distribution = self.forest_analyzer.analyze_spatial_distribution(
                df, density_type, crown_base_height
            )
            
            distributions['model_names'].append(model_name)
            distributions['distributions'].append(distribution)
        
        return distributions
    
    def _create_metrics_summary(self, model_names: List[str], 
                               layer_metrics: List[Dict[str, Dict]]) -> pd.DataFrame:
        """Create summary table of layer metrics for comparison."""
        summary_data = []
        
        layer_names = ['whole', 'upper', 'middle', 'lower']
        metric_names = ['total_area', 'mean_area_index', 'max_area_index', 'voxel_count']
        
        for model_name, metrics in zip(model_names, layer_metrics):
            for layer_name in layer_names:
                for metric_name in metric_names:
                    summary_data.append({
                        'Model': model_name,
                        'Layer': layer_name.capitalize(),
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Value': metrics[layer_name].get(metric_name, 0)
                    })
        
        return pd.DataFrame(summary_data)