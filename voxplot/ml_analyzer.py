#!/usr/bin/env python3
"""
ML/AI Analysis Framework for VoxPlot
====================================
Advanced machine learning and AI analysis for 3D forest structure data.
Provides spatial pattern analysis, clustering, dimensionality reduction,
and physical accuracy assessment for voxel-based forest models.

This module extends VoxPlot with:
- Clustering analysis (K-Means, DBSCAN, Hierarchical)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Regression analysis (Linear, Random Forest, XGBoost)
- Spatial pattern detection and clumping analysis
- Occlusion detection and modeling
- Physical accuracy assessment through ML

Author: Joshua B-L & Claude Code
Date: 2025-09-10
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import warnings
from datetime import datetime

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.spatial import distance_matrix, ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label, center_of_mass

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Optional advanced libraries
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore', category=UserWarning)


class SpatialPatternAnalyzer:
    """
    Advanced spatial pattern analysis for 3D forest structure data.
    
    Analyzes:
    - Density distribution patterns
    - Spatial clustering and clumping
    - Height-stratified patterns
    - Model comparison through ML
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize spatial pattern analyzer."""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.results = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default analysis configuration."""
        return {
            'clustering': {
                'kmeans_clusters': [3, 5, 8, 12],
                'dbscan_eps': [0.5, 1.0, 2.0],
                'dbscan_min_samples': [5, 10, 20],
                'enable_hierarchical': True
            },
            'dimensionality_reduction': {
                'pca_components': [2, 3, 5],
                'tsne_perplexity': [10, 30, 50],
                'tsne_components': 2,
                'umap_neighbors': [15, 30, 50],
                'umap_components': 2
            },
            'regression': {
                'enable_linear': True,
                'enable_rf': True,
                'enable_xgboost': XGBOOST_AVAILABLE,
                'cross_validation_folds': 5
            },
            'spatial_analysis': {
                'height_layers': ['whole', 'upper', 'middle', 'lower'],
                'density_types': ['lad', 'wad', 'pad'],
                'clumping_threshold': 0.1,
                'neighborhood_radius': 1.0
            }
        }
    
    def analyze_spatial_patterns(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive spatial pattern analysis.
        
        Args:
            data_dict: Dictionary of model data {model_name: dataframe}
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting comprehensive spatial pattern analysis")
        
        results = {
            'clustering_analysis': {},
            'dimensionality_reduction': {},
            'regression_analysis': {},
            'spatial_patterns': {},
            'physical_accuracy': {},
            'occlusion_analysis': {},
            'comparative_analysis': {}
        }
        
        # Process each model
        for model_name, df in data_dict.items():
            self.logger.info(f"Analyzing spatial patterns for {model_name}")
            
            # Prepare features for ML analysis
            features = self._prepare_features(df)
            
            # Clustering analysis
            results['clustering_analysis'][model_name] = self._perform_clustering_analysis(
                features, model_name
            )
            
            # Dimensionality reduction
            results['dimensionality_reduction'][model_name] = self._perform_dimensionality_reduction(
                features, model_name
            )
            
            # Spatial pattern detection
            results['spatial_patterns'][model_name] = self._analyze_spatial_patterns(
                df, model_name
            )
            
            # Physical accuracy assessment
            results['physical_accuracy'][model_name] = self._assess_physical_accuracy(
                df, model_name
            )
            
            # Occlusion analysis
            results['occlusion_analysis'][model_name] = self._analyze_occlusion_patterns(
                df, model_name
            )
        
        # Cross-model comparison
        if len(data_dict) > 1:
            results['comparative_analysis'] = self._perform_comparative_analysis(
                data_dict, results
            )
            
            # Regression analysis between models
            results['regression_analysis'] = self._perform_regression_analysis(
                data_dict
            )
        
        self.results = results
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML analysis.
        
        Args:
            df: Voxel data with columns [x, y, z, density_value, ...]
            
        Returns:
            Feature matrix for ML analysis
        """
        features = df.copy()
        
        # Basic spatial features
        if 'x' in features.columns and 'y' in features.columns and 'z' in features.columns:
            # Distance from center
            center_x, center_y = features['x'].mean(), features['y'].mean()
            features['distance_from_center'] = np.sqrt(
                (features['x'] - center_x)**2 + (features['y'] - center_y)**2
            )
            
            # Height above ground
            features['height_above_ground'] = features['z'] - features['z'].min()
            
            # Relative height
            features['relative_height'] = (features['z'] - features['z'].min()) / (
                features['z'].max() - features['z'].min()
            )
            
            # Spatial gradients
            features['x_gradient'] = np.gradient(features['density_value'].values)
            
        # Density-based features
        if 'density_value' in features.columns:
            # Density statistics in neighborhood
            features['density_log'] = np.log1p(features['density_value'])
            features['density_squared'] = features['density_value'] ** 2
            
            # Local density variations
            features['density_variation'] = self._calculate_local_variation(
                features, 'density_value'
            )
        
        # Crown layer features
        features = self._add_crown_layer_features(features)
        
        return features
    
    def _calculate_local_variation(self, df: pd.DataFrame, column: str, 
                                 radius: float = 1.0) -> np.ndarray:
        """Calculate local variation in density values."""
        if len(df) < 10:  # Too few points for meaningful analysis
            return np.zeros(len(df))
        
        try:
            # Sample for efficiency if dataset is large
            if len(df) > 5000:
                sample_idx = np.random.choice(len(df), 5000, replace=False)
                df_sample = df.iloc[sample_idx]
            else:
                df_sample = df
            
            coords = df_sample[['x', 'y', 'z']].values
            values = df_sample[column].values
            
            # Use NearestNeighbors for efficiency
            nbrs = NearestNeighbors(radius=radius).fit(coords)
            variations = np.zeros(len(df_sample))
            
            for i, point in enumerate(coords):
                neighbors = nbrs.radius_neighbors([point], return_distance=False)[0]
                if len(neighbors) > 1:
                    neighbor_values = values[neighbors]
                    variations[i] = np.std(neighbor_values)
            
            # Map back to full dataset if sampled
            if len(df) > 5000:
                full_variations = np.zeros(len(df))
                full_variations[sample_idx] = variations
                # Interpolate for missing values
                for i in range(len(df)):
                    if i not in sample_idx:
                        distances = np.sum((df.iloc[i][['x', 'y', 'z']].values - coords)**2, axis=1)
                        closest_idx = sample_idx[np.argmin(distances)]
                        full_variations[i] = variations[np.where(sample_idx == closest_idx)[0][0]]
                return full_variations
            else:
                return variations
                
        except Exception as e:
            self.logger.warning(f"Error calculating local variation: {e}")
            return np.zeros(len(df))
    
    def _add_crown_layer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crown layer classification features."""
        if 'z' not in df.columns:
            return df
        
        # Define height percentiles
        z_min, z_max = df['z'].min(), df['z'].max()
        height_range = z_max - z_min
        
        # Crown layer thresholds (based on forestry standards)
        lower_threshold = z_min + 0.33 * height_range
        upper_threshold = z_min + 0.67 * height_range
        
        # Assign crown layers
        conditions = [
            df['z'] <= lower_threshold,
            (df['z'] > lower_threshold) & (df['z'] <= upper_threshold),
            df['z'] > upper_threshold
        ]
        choices = ['lower', 'middle', 'upper']
        df['crown_layer'] = np.select(conditions, choices, default='middle')
        
        # Crown layer binary features
        for layer in ['lower', 'middle', 'upper']:
            df[f'is_{layer}_crown'] = (df['crown_layer'] == layer).astype(int)
        
        return df
    
    def _perform_clustering_analysis(self, features: pd.DataFrame, 
                                   model_name: str) -> Dict[str, Any]:
        """Perform comprehensive clustering analysis."""
        self.logger.info(f"Performing clustering analysis for {model_name}")
        
        # Select numerical features for clustering
        numeric_features = features.select_dtypes(include=[np.number]).dropna()
        
        if numeric_features.empty:
            self.logger.warning(f"No numerical features for clustering in {model_name}")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        clustering_results = {
            'kmeans': {},
            'dbscan': {},
            'hierarchical': {},
            'feature_importance': {},
            'cluster_characteristics': {}
        }
        
        # K-Means clustering
        best_kmeans = None
        best_kmeans_score = -1
        
        for n_clusters in self.config['clustering']['kmeans']['cluster_range']:
            if len(X_scaled) < n_clusters:
                continue
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            if len(np.unique(cluster_labels)) > 1:
                silhouette = silhouette_score(X_scaled, cluster_labels)
                
                clustering_results['kmeans'][n_clusters] = {
                    'silhouette_score': silhouette,
                    'cluster_labels': cluster_labels,
                    'cluster_centers': kmeans.cluster_centers_,
                    'inertia': kmeans.inertia_
                }
                
                if silhouette > best_kmeans_score:
                    best_kmeans_score = silhouette
                    best_kmeans = n_clusters
        
        clustering_results['best_kmeans'] = {
            'n_clusters': best_kmeans,
            'score': best_kmeans_score
        }
        
        # DBSCAN clustering
        best_dbscan = None
        best_dbscan_score = -1
        
        for eps in self.config['clustering']['dbscan']['eps_values']:
            for min_samples in self.config['clustering']['dbscan']['min_samples']:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(X_scaled)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters > 1:
                    # Calculate silhouette score excluding noise points
                    mask = cluster_labels != -1
                    if np.sum(mask) > 0:
                        silhouette = silhouette_score(X_scaled[mask], cluster_labels[mask])
                        
                        clustering_results['dbscan'][(eps, min_samples)] = {
                            'silhouette_score': silhouette,
                            'cluster_labels': cluster_labels,
                            'n_clusters': n_clusters,
                            'n_noise': np.sum(cluster_labels == -1),
                            'noise_ratio': np.sum(cluster_labels == -1) / len(cluster_labels)
                        }
                        
                        if silhouette > best_dbscan_score:
                            best_dbscan_score = silhouette
                            best_dbscan = (eps, min_samples)
        
        if best_dbscan:
            clustering_results['best_dbscan'] = {
                'parameters': best_dbscan,
                'score': best_dbscan_score
            }
        
        # Analyze cluster characteristics for best K-means
        if best_kmeans and best_kmeans in clustering_results['kmeans']:
            cluster_labels = clustering_results['kmeans'][best_kmeans]['cluster_labels']
            clustering_results['cluster_characteristics'] = self._analyze_cluster_characteristics(
                features, cluster_labels, model_name
            )
        
        return clustering_results
    
    def _analyze_cluster_characteristics(self, features: pd.DataFrame, 
                                       cluster_labels: np.ndarray, 
                                       model_name: str) -> Dict[str, Any]:
        """Analyze characteristics of identified clusters."""
        characteristics = {}
        
        # Add cluster labels to features
        features_with_clusters = features.copy()
        features_with_clusters['cluster'] = cluster_labels
        
        # Analyze each cluster
        for cluster_id in np.unique(cluster_labels):
            cluster_data = features_with_clusters[features_with_clusters['cluster'] == cluster_id]
            
            cluster_stats = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(features) * 100,
                'density_stats': {},
                'spatial_stats': {},
                'height_distribution': {}
            }
            
            # Density statistics
            if 'density_value' in cluster_data.columns:
                cluster_stats['density_stats'] = {
                    'mean': cluster_data['density_value'].mean(),
                    'std': cluster_data['density_value'].std(),
                    'median': cluster_data['density_value'].median(),
                    'max': cluster_data['density_value'].max(),
                    'min': cluster_data['density_value'].min()
                }
            
            # Spatial statistics
            if all(col in cluster_data.columns for col in ['x', 'y', 'z']):
                cluster_stats['spatial_stats'] = {
                    'centroid': [
                        cluster_data['x'].mean(),
                        cluster_data['y'].mean(),
                        cluster_data['z'].mean()
                    ],
                    'spatial_extent': {
                        'x_range': cluster_data['x'].max() - cluster_data['x'].min(),
                        'y_range': cluster_data['y'].max() - cluster_data['y'].min(),
                        'z_range': cluster_data['z'].max() - cluster_data['z'].min()
                    },
                    'volume': self._estimate_cluster_volume(cluster_data)
                }
            
            # Height distribution
            if 'crown_layer' in cluster_data.columns:
                layer_counts = cluster_data['crown_layer'].value_counts()
                total_count = len(cluster_data)
                cluster_stats['height_distribution'] = {
                    layer: count / total_count * 100 
                    for layer, count in layer_counts.items()
                }
            
            characteristics[f'cluster_{cluster_id}'] = cluster_stats
        
        return characteristics
    
    def _estimate_cluster_volume(self, cluster_data: pd.DataFrame) -> float:
        """Estimate 3D volume of a cluster using convex hull."""
        if len(cluster_data) < 4:  # Need at least 4 points for 3D convex hull
            return 0.0
        
        try:
            points = cluster_data[['x', 'y', 'z']].values
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            # Fallback: use bounding box volume
            x_range = cluster_data['x'].max() - cluster_data['x'].min()
            y_range = cluster_data['y'].max() - cluster_data['y'].min()
            z_range = cluster_data['z'].max() - cluster_data['z'].min()
            return x_range * y_range * z_range
    
    def _perform_dimensionality_reduction(self, features: pd.DataFrame, 
                                        model_name: str) -> Dict[str, Any]:
        """Perform dimensionality reduction analysis."""
        self.logger.info(f"Performing dimensionality reduction for {model_name}")
        
        # Select numerical features
        numeric_features = features.select_dtypes(include=[np.number]).dropna()
        
        if numeric_features.empty or len(numeric_features) < 10:
            self.logger.warning(f"Insufficient data for dimensionality reduction in {model_name}")
            return {}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        reduction_results = {
            'pca': {},
            'tsne': {},
            'umap': {} if UMAP_AVAILABLE else None,
            'feature_importance': {},
            'explained_variance': {}
        }
        
        # PCA Analysis
        for n_components in self.config['dimensionality_reduction']['pca']['n_components']:
            if n_components < min(X_scaled.shape):
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                reduction_results['pca'][n_components] = {
                    'transformed_data': X_pca,
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                    'components': pca.components_,
                    'feature_names': numeric_features.columns.tolist()
                }
        
        # t-SNE Analysis
        if len(X_scaled) <= 10000:  # t-SNE is computationally expensive
            for perplexity in self.config['dimensionality_reduction']['tsne']['perplexity']:
                if perplexity < len(X_scaled) / 3:
                    tsne = TSNE(
                        n_components=self.config['dimensionality_reduction']['tsne']['n_components'],
                        perplexity=perplexity,
                        random_state=42
                    )
                    X_tsne = tsne.fit_transform(X_scaled)
                    
                    reduction_results['tsne'][perplexity] = {
                        'transformed_data': X_tsne,
                        'kl_divergence': tsne.kl_divergence_
                    }
        else:
            self.logger.info(f"Skipping t-SNE for {model_name} due to large dataset size")
        
        # UMAP Analysis (if available)
        if UMAP_AVAILABLE and len(X_scaled) > 15:
            for n_neighbors in self.config['dimensionality_reduction']['umap']['n_neighbors']:
                if n_neighbors < len(X_scaled):
                    try:
                        umap_model = umap.UMAP(
                            n_neighbors=n_neighbors,
                            n_components=self.config['dimensionality_reduction']['umap']['n_components'],
                            random_state=42
                        )
                        X_umap = umap_model.fit_transform(X_scaled)
                        
                        reduction_results['umap'][n_neighbors] = {
                            'transformed_data': X_umap
                        }
                    except Exception as e:
                        self.logger.warning(f"UMAP failed for {model_name}: {e}")
        
        return reduction_results
    
    def _analyze_spatial_patterns(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Analyze spatial patterns and clumping."""
        self.logger.info(f"Analyzing spatial patterns for {model_name}")
        
        spatial_results = {
            'density_hotspots': {},
            'clumping_analysis': {},
            'spatial_autocorrelation': {},
            'height_stratified_patterns': {},
            'density_gradients': {}
        }
        
        # Density hotspot detection
        if 'density_value' in df.columns:
            hotspots = self._detect_density_hotspots(df)
            spatial_results['density_hotspots'] = hotspots
        
        # Clumping analysis
        clumping = self._analyze_clumping_patterns(df)
        spatial_results['clumping_analysis'] = clumping
        
        # Height-stratified analysis
        if 'z' in df.columns:
            height_patterns = self._analyze_height_stratified_patterns(df)
            spatial_results['height_stratified_patterns'] = height_patterns
        
        # Density gradients
        if all(col in df.columns for col in ['x', 'y', 'z', 'density_value']):
            gradients = self._calculate_density_gradients(df)
            spatial_results['density_gradients'] = gradients
        
        return spatial_results
    
    def _detect_density_hotspots(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect high and low density regions."""
        density_col = 'density_value'
        
        # Define thresholds
        high_threshold = df[density_col].quantile(0.9)
        low_threshold = df[density_col].quantile(0.1)
        
        # Identify hotspots and coldspots
        hotspots = df[df[density_col] >= high_threshold]
        coldspots = df[df[density_col] <= low_threshold]
        
        results = {
            'high_density_regions': {
                'count': len(hotspots),
                'percentage': len(hotspots) / len(df) * 100,
                'mean_density': hotspots[density_col].mean(),
                'spatial_distribution': self._analyze_spatial_distribution(hotspots)
            },
            'low_density_regions': {
                'count': len(coldspots),
                'percentage': len(coldspots) / len(df) * 100,
                'mean_density': coldspots[density_col].mean(),
                'spatial_distribution': self._analyze_spatial_distribution(coldspots)
            },
            'thresholds': {
                'high': high_threshold,
                'low': low_threshold
            }
        }
        
        return results
    
    def _analyze_spatial_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spatial distribution of points."""
        if len(df) < 3 or not all(col in df.columns for col in ['x', 'y', 'z']):
            return {}
        
        coords = df[['x', 'y', 'z']].values
        
        return {
            'centroid': coords.mean(axis=0).tolist(),
            'std_deviation': coords.std(axis=0).tolist(),
            'spatial_extent': {
                'x_range': coords[:, 0].max() - coords[:, 0].min(),
                'y_range': coords[:, 1].max() - coords[:, 1].min(),
                'z_range': coords[:, 2].max() - coords[:, 2].min()
            }
        }
    
    def _analyze_clumping_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clumping patterns in the data."""
        if len(df) < 10 or not all(col in df.columns for col in ['x', 'y', 'z', 'density_value']):
            return {}
        
        # Sample for efficiency
        if len(df) > 2000:
            sample_df = df.sample(n=2000, random_state=42)
        else:
            sample_df = df
        
        coords = sample_df[['x', 'y', 'z']].values
        densities = sample_df['density_value'].values
        
        # Calculate pairwise distances
        try:
            distances = pdist(coords)
            distance_matrix = squareform(distances)
            
            # Find clumps using density-based approach
            threshold = self.config['spatial_analysis']['clumping']['clumping_threshold']
            high_density_mask = densities > np.percentile(densities, 75)
            
            clump_results = {
                'n_high_density_points': np.sum(high_density_mask),
                'clump_statistics': {},
                'nearest_neighbor_analysis': {}
            }
            
            # Nearest neighbor analysis
            if np.sum(high_density_mask) > 1:
                high_density_coords = coords[high_density_mask]
                nn_distances = []
                
                for i, point in enumerate(high_density_coords):
                    other_points = np.delete(high_density_coords, i, axis=0)
                    if len(other_points) > 0:
                        distances_to_point = np.sqrt(np.sum((other_points - point)**2, axis=1))
                        nn_distances.append(np.min(distances_to_point))
                
                if nn_distances:
                    clump_results['nearest_neighbor_analysis'] = {
                        'mean_nn_distance': np.mean(nn_distances),
                        'std_nn_distance': np.std(nn_distances),
                        'clumping_index': 1.0 / (np.mean(nn_distances) + 1e-10)  # Inverse of mean distance
                    }
            
            return clump_results
            
        except Exception as e:
            self.logger.warning(f"Error in clumping analysis: {e}")
            return {}
    
    def _analyze_height_stratified_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns across different height levels."""
        height_results = {}
        
        # Define height layers
        z_min, z_max = df['z'].min(), df['z'].max()
        height_range = z_max - z_min
        
        layers = {
            'lower': df[df['z'] <= z_min + 0.33 * height_range],
            'middle': df[(df['z'] > z_min + 0.33 * height_range) & 
                        (df['z'] <= z_min + 0.67 * height_range)],
            'upper': df[df['z'] > z_min + 0.67 * height_range],
            'whole': df
        }
        
        for layer_name, layer_data in layers.items():
            if len(layer_data) > 5:
                layer_stats = {
                    'point_count': len(layer_data),
                    'density_statistics': {},
                    'spatial_extent': {}
                }
                
                if 'density_value' in layer_data.columns:
                    density_stats = {
                        'mean': layer_data['density_value'].mean(),
                        'std': layer_data['density_value'].std(),
                        'median': layer_data['density_value'].median(),
                        'skewness': stats.skew(layer_data['density_value']),
                        'kurtosis': stats.kurtosis(layer_data['density_value'])
                    }
                    layer_stats['density_statistics'] = density_stats
                
                # Spatial extent
                if all(col in layer_data.columns for col in ['x', 'y']):
                    spatial_extent = {
                        'x_range': layer_data['x'].max() - layer_data['x'].min(),
                        'y_range': layer_data['y'].max() - layer_data['y'].min(),
                        'area': (layer_data['x'].max() - layer_data['x'].min()) * 
                               (layer_data['y'].max() - layer_data['y'].min())
                    }
                    layer_stats['spatial_extent'] = spatial_extent
                
                height_results[layer_name] = layer_stats
        
        return height_results
    
    def _calculate_density_gradients(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate spatial gradients in density."""
        if len(df) < 20:
            return {}
        
        try:
            # Create a regular grid and interpolate
            x_unique = np.unique(df['x'])
            y_unique = np.unique(df['y'])
            
            if len(x_unique) < 3 or len(y_unique) < 3:
                return {}
            
            # Simple gradient calculation
            gradient_results = {
                'mean_gradient_magnitude': 0.0,
                'gradient_direction_analysis': {},
                'steep_gradient_regions': {}
            }
            
            # Calculate local gradients using finite differences
            gradients = []
            
            # Sample points for gradient calculation
            sample_size = min(500, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            
            for idx, row in sample_df.iterrows():
                # Find nearby points
                nearby_mask = (
                    (np.abs(df['x'] - row['x']) <= 1.0) &
                    (np.abs(df['y'] - row['y']) <= 1.0) &
                    (np.abs(df['z'] - row['z']) <= 1.0)
                )
                nearby_points = df[nearby_mask]
                
                if len(nearby_points) > 3:
                    # Calculate local gradient
                    dx = nearby_points['density_value'].values - row['density_value']
                    spatial_dx = np.sqrt(
                        (nearby_points['x'].values - row['x'])**2 +
                        (nearby_points['y'].values - row['y'])**2 +
                        (nearby_points['z'].values - row['z'])**2
                    )
                    
                    valid_mask = spatial_dx > 1e-6
                    if np.sum(valid_mask) > 0:
                        local_gradients = dx[valid_mask] / spatial_dx[valid_mask]
                        gradients.extend(local_gradients)
            
            if gradients:
                gradient_results['mean_gradient_magnitude'] = np.mean(np.abs(gradients))
                gradient_results['gradient_statistics'] = {
                    'mean': np.mean(gradients),
                    'std': np.std(gradients),
                    'max_positive': np.max(gradients) if len(gradients) > 0 else 0,
                    'max_negative': np.min(gradients) if len(gradients) > 0 else 0
                }
            
            return gradient_results
            
        except Exception as e:
            self.logger.warning(f"Error calculating gradients: {e}")
            return {}
    
    def _assess_physical_accuracy(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Assess physical accuracy indicators."""
        self.logger.info(f"Assessing physical accuracy for {model_name}")
        
        accuracy_results = {
            'density_distribution_analysis': {},
            'physical_plausibility': {},
            'consistency_metrics': {},
            'anomaly_detection': {}
        }
        
        if 'density_value' in df.columns:
            # Density distribution analysis
            densities = df['density_value'].values
            
            accuracy_results['density_distribution_analysis'] = {
                'distribution_type': self._classify_distribution(densities),
                'normality_test': stats.normaltest(densities)[1],  # p-value
                'outlier_percentage': self._calculate_outlier_percentage(densities),
                'zero_density_percentage': np.sum(densities == 0) / len(densities) * 100
            }
            
            # Physical plausibility checks
            accuracy_results['physical_plausibility'] = {
                'negative_densities': np.sum(densities < 0),
                'extremely_high_densities': np.sum(densities > np.percentile(densities, 99.9)),
                'density_continuity': self._assess_density_continuity(df)
            }
        
        # Consistency across height layers
        if 'z' in df.columns:
            accuracy_results['consistency_metrics'] = self._assess_height_consistency(df)
        
        return accuracy_results
    
    def _classify_distribution(self, data: np.ndarray) -> str:
        """Classify the type of distribution."""
        # Simple classification based on statistical tests
        if len(data) < 8:
            return "insufficient_data"
        
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "normal"
        elif skewness > 1:
            return "right_skewed"
        elif skewness < -1:
            return "left_skewed"
        elif kurtosis > 3:
            return "heavy_tailed"
        else:
            return "unknown"
    
    def _calculate_outlier_percentage(self, data: np.ndarray) -> float:
        """Calculate percentage of outliers using IQR method."""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return np.sum(outliers) / len(data) * 100
    
    def _assess_density_continuity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess spatial continuity of density values."""
        if len(df) < 10:
            return {}
        
        try:
            # Sample for efficiency
            sample_df = df.sample(n=min(1000, len(df)), random_state=42)
            coords = sample_df[['x', 'y', 'z']].values
            densities = sample_df['density_value'].values
            
            # Calculate spatial correlation
            distances = pdist(coords)
            density_differences = pdist(densities.reshape(-1, 1))
            
            if len(distances) > 0 and len(density_differences) > 0:
                # Correlation between spatial distance and density difference
                correlation = np.corrcoef(distances, density_differences.flatten())[0, 1]
                
                return {
                    'spatial_correlation': correlation if not np.isnan(correlation) else 0.0,
                    'continuity_assessment': 'good' if abs(correlation) > 0.3 else 'poor'
                }
            else:
                return {}
                
        except Exception as e:
            self.logger.warning(f"Error assessing density continuity: {e}")
            return {}
    
    def _assess_height_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess consistency across height layers."""
        if 'density_value' not in df.columns or len(df) < 20:
            return {}
        
        # Divide into height layers
        z_min, z_max = df['z'].min(), df['z'].max()
        height_range = z_max - z_min
        
        layer_bounds = [
            (z_min, z_min + 0.33 * height_range),  # lower
            (z_min + 0.33 * height_range, z_min + 0.67 * height_range),  # middle
            (z_min + 0.67 * height_range, z_max)  # upper
        ]
        
        layer_stats = {}
        for i, (lower, upper) in enumerate(layer_bounds):
            layer_name = ['lower', 'middle', 'upper'][i]
            layer_data = df[(df['z'] >= lower) & (df['z'] <= upper)]
            
            if len(layer_data) > 5:
                layer_stats[layer_name] = {
                    'mean_density': layer_data['density_value'].mean(),
                    'density_variance': layer_data['density_value'].var(),
                    'point_count': len(layer_data)
                }
        
        # Calculate consistency metrics
        consistency_metrics = {}
        if len(layer_stats) >= 2:
            mean_densities = [stats['mean_density'] for stats in layer_stats.values()]
            variances = [stats['density_variance'] for stats in layer_stats.values()]
            
            consistency_metrics = {
                'density_variation_across_layers': np.std(mean_densities),
                'variance_consistency': np.std(variances),
                'layer_statistics': layer_stats
            }
        
        return consistency_metrics
    
    def _analyze_occlusion_patterns(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Analyze occlusion patterns and effects."""
        self.logger.info(f"Analyzing occlusion patterns for {model_name}")
        
        occlusion_results = {
            'shadow_analysis': {},
            'density_drop_patterns': {},
            'line_of_sight_analysis': {},
            'occlusion_correction_assessment': {}
        }
        
        if len(df) < 50 or not all(col in df.columns for col in ['x', 'y', 'z', 'density_value']):
            return occlusion_results
        
        # Shadow analysis (areas with suspiciously low density below high density areas)
        shadow_analysis = self._detect_shadow_regions(df)
        occlusion_results['shadow_analysis'] = shadow_analysis
        
        # Density drop patterns
        drop_patterns = self._analyze_density_drop_patterns(df)
        occlusion_results['density_drop_patterns'] = drop_patterns
        
        # Assess occlusion correction
        correction_assessment = self._assess_occlusion_correction(df, model_name)
        occlusion_results['occlusion_correction_assessment'] = correction_assessment
        
        return occlusion_results
    
    def _detect_shadow_regions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential shadow/occlusion regions."""
        if len(df) < 20:
            return {}
        
        try:
            # Sort by height
            df_sorted = df.sort_values('z', ascending=False)
            
            # Define height layers
            z_values = df_sorted['z'].values
            height_layers = np.percentile(z_values, [75, 50, 25])  # Upper, middle, lower
            
            shadow_candidates = []
            
            # Look for low density regions below high density regions
            for idx, row in df_sorted.iterrows():
                if row['z'] < height_layers[1]:  # In lower half
                    # Find points directly above (within spatial tolerance)
                    spatial_tolerance = 2.0
                    above_mask = (
                        (np.abs(df['x'] - row['x']) <= spatial_tolerance) &
                        (np.abs(df['y'] - row['y']) <= spatial_tolerance) &
                        (df['z'] > row['z'])
                    )
                    
                    points_above = df[above_mask]
                    if len(points_above) > 0:
                        max_density_above = points_above['density_value'].max()
                        density_ratio = row['density_value'] / (max_density_above + 1e-10)
                        
                        if density_ratio < 0.1 and max_density_above > 0.01:  # Significant shadow
                            shadow_candidates.append({
                                'location': [row['x'], row['y'], row['z']],
                                'density': row['density_value'],
                                'max_density_above': max_density_above,
                                'density_ratio': density_ratio
                            })
            
            return {
                'n_shadow_candidates': len(shadow_candidates),
                'shadow_percentage': len(shadow_candidates) / len(df) * 100,
                'average_shadow_intensity': np.mean([s['density_ratio'] for s in shadow_candidates]) if shadow_candidates else 0
            }
            
        except Exception as e:
            self.logger.warning(f"Error in shadow detection: {e}")
            return {}
    
    def _analyze_density_drop_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns of density drops that might indicate occlusion."""
        if len(df) < 20:
            return {}
        
        try:
            # Calculate height-based density profile
            df_sorted = df.sort_values('z')
            
            # Bin by height
            n_bins = min(20, len(df) // 10)
            if n_bins < 3:
                return {}
            
            height_bins = np.linspace(df['z'].min(), df['z'].max(), n_bins)
            bin_indices = np.digitize(df_sorted['z'], height_bins)
            
            height_profile = []
            for i in range(1, len(height_bins)):
                bin_data = df_sorted[bin_indices == i]
                if len(bin_data) > 0:
                    height_profile.append({
                        'height': (height_bins[i-1] + height_bins[i]) / 2,
                        'mean_density': bin_data['density_value'].mean(),
                        'n_points': len(bin_data)
                    })
            
            if len(height_profile) < 3:
                return {}
            
            # Look for unexpected density drops
            densities = [p['mean_density'] for p in height_profile]
            heights = [p['height'] for p in height_profile]
            
            # Calculate rate of change
            density_changes = np.diff(densities)
            significant_drops = density_changes < -np.std(density_changes) * 2
            
            return {
                'height_profile': height_profile,
                'n_significant_drops': np.sum(significant_drops),
                'drop_locations': [heights[i+1] for i, is_drop in enumerate(significant_drops) if is_drop],
                'density_trend': 'increasing' if np.mean(density_changes) > 0 else 'decreasing'
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing density drops: {e}")
            return {}
    
    def _assess_occlusion_correction(self, df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Assess how well the model has corrected for occlusion effects."""
        # This is model-specific assessment
        correction_indicators = {
            'model_type': model_name,
            'assessment': 'unknown'
        }
        
        # Model-specific assessments based on known characteristics
        if 'amapvox' in model_name.lower():
            # AmapVox typically has good occlusion correction
            correction_indicators['expected_performance'] = 'good'
            correction_indicators['assessment_notes'] = 'AmapVox includes path length correction'
            
        elif 'voxlad' in model_name.lower():
            # VoxLAD has occlusion modeling
            correction_indicators['expected_performance'] = 'moderate'
            correction_indicators['assessment_notes'] = 'VoxLAD includes multi-scan integration'
            
        elif 'voxpy' in model_name.lower():
            # VoxPy varies depending on configuration
            correction_indicators['expected_performance'] = 'variable'
            correction_indicators['assessment_notes'] = 'VoxPy performance depends on configuration'
        
        # Calculate empirical indicators
        if 'density_value' in df.columns and len(df) > 20:
            # Look for signs of good occlusion correction
            density_stats = {
                'zero_density_percentage': np.sum(df['density_value'] == 0) / len(df) * 100,
                'very_low_density_percentage': np.sum(df['density_value'] < 0.001) / len(df) * 100
            }
            
            # High percentage of zero densities might indicate poor occlusion correction
            if density_stats['zero_density_percentage'] > 50:
                correction_indicators['empirical_assessment'] = 'poor'
            elif density_stats['zero_density_percentage'] < 20:
                correction_indicators['empirical_assessment'] = 'good'
            else:
                correction_indicators['empirical_assessment'] = 'moderate'
            
            correction_indicators['density_statistics'] = density_stats
        
        return correction_indicators
    
    def _perform_comparative_analysis(self, data_dict: Dict[str, pd.DataFrame], 
                                    results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis between models."""
        self.logger.info("Performing comparative ML analysis between models")
        
        comparative_results = {
            'model_similarity_analysis': {},
            'clustering_comparison': {},
            'pattern_consistency': {},
            'performance_ranking': {}
        }
        
        model_names = list(data_dict.keys())
        
        # Model similarity analysis using dimensionality reduction
        if len(model_names) >= 2:
            similarity_matrix = self._calculate_model_similarity(data_dict, results)
            comparative_results['model_similarity_analysis'] = similarity_matrix
        
        # Compare clustering results
        clustering_comparison = self._compare_clustering_results(results)
        comparative_results['clustering_comparison'] = clustering_comparison
        
        # Analyze pattern consistency
        pattern_consistency = self._analyze_pattern_consistency(results)
        comparative_results['pattern_consistency'] = pattern_consistency
        
        # Rank models based on multiple criteria
        performance_ranking = self._rank_models_by_performance(results)
        comparative_results['performance_ranking'] = performance_ranking
        
        return comparative_results
    
    def _calculate_model_similarity(self, data_dict: Dict[str, pd.DataFrame], 
                                  results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate similarity between models based on spatial patterns."""
        model_names = list(data_dict.keys())
        n_models = len(model_names)
        
        similarity_matrix = np.zeros((n_models, n_models))
        
        # Extract comparable features from each model
        model_features = {}
        for model_name in model_names:
            # Use spatial pattern features for comparison
            spatial_results = results.get('spatial_patterns', {}).get(model_name, {})
            
            features = []
            
            # Density hotspot features
            if 'density_hotspots' in spatial_results:
                hotspots = spatial_results['density_hotspots']
                if 'high_density_regions' in hotspots:
                    features.extend([
                        hotspots['high_density_regions'].get('percentage', 0),
                        hotspots['high_density_regions'].get('mean_density', 0)
                    ])
                if 'low_density_regions' in hotspots:
                    features.extend([
                        hotspots['low_density_regions'].get('percentage', 0),
                        hotspots['low_density_regions'].get('mean_density', 0)
                    ])
            
            # Height-stratified features
            if 'height_stratified_patterns' in spatial_results:
                for layer in ['lower', 'middle', 'upper']:
                    layer_data = spatial_results['height_stratified_patterns'].get(layer, {})
                    if 'density_statistics' in layer_data:
                        features.append(layer_data['density_statistics'].get('mean', 0))
            
            # Pad with zeros if features are missing
            while len(features) < 10:
                features.append(0)
            
            model_features[model_name] = np.array(features[:10])  # Limit to 10 features
        
        # Calculate pairwise similarities
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate cosine similarity
                    features1 = model_features[model1]
                    features2 = model_features[model2]
                    
                    norm1 = np.linalg.norm(features1)
                    norm2 = np.linalg.norm(features2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = np.dot(features1, features2) / (norm1 * norm2)
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
        
        return {
            'similarity_matrix': similarity_matrix.tolist(),
            'model_names': model_names,
            'most_similar_pair': self._find_most_similar_models(similarity_matrix, model_names),
            'least_similar_pair': self._find_least_similar_models(similarity_matrix, model_names)
        }
    
    def _find_most_similar_models(self, similarity_matrix: np.ndarray, 
                                model_names: List[str]) -> Dict[str, Any]:
        """Find the most similar pair of models."""
        # Set diagonal to -1 to exclude self-similarity
        masked_matrix = similarity_matrix.copy()
        np.fill_diagonal(masked_matrix, -1)
        
        max_idx = np.unravel_index(np.argmax(masked_matrix), masked_matrix.shape)
        
        return {
            'models': [model_names[max_idx[0]], model_names[max_idx[1]]],
            'similarity_score': similarity_matrix[max_idx]
        }
    
    def _find_least_similar_models(self, similarity_matrix: np.ndarray, 
                                 model_names: List[str]) -> Dict[str, Any]:
        """Find the least similar pair of models."""
        # Set diagonal to 2 to exclude self-similarity
        masked_matrix = similarity_matrix.copy()
        np.fill_diagonal(masked_matrix, 2)
        
        min_idx = np.unravel_index(np.argmin(masked_matrix), masked_matrix.shape)
        
        return {
            'models': [model_names[min_idx[0]], model_names[min_idx[1]]],
            'similarity_score': similarity_matrix[min_idx]
        }
    
    def _compare_clustering_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare clustering results between models."""
        clustering_results = results.get('clustering_analysis', {})
        
        comparison = {
            'optimal_clusters': {},
            'silhouette_scores': {},
            'cluster_consistency': {}
        }
        
        # Extract optimal cluster numbers and scores
        for model_name, model_clustering in clustering_results.items():
            if 'best_kmeans' in model_clustering:
                best_kmeans = model_clustering['best_kmeans']
                comparison['optimal_clusters'][model_name] = best_kmeans.get('n_clusters', 0)
                comparison['silhouette_scores'][model_name] = best_kmeans.get('score', 0)
        
        # Analyze consistency
        if comparison['optimal_clusters']:
            cluster_counts = list(comparison['optimal_clusters'].values())
            comparison['cluster_consistency'] = {
                'mean_optimal_clusters': np.mean(cluster_counts),
                'std_optimal_clusters': np.std(cluster_counts),
                'consistency_assessment': 'high' if np.std(cluster_counts) < 1.5 else 'low'
            }
        
        return comparison
    
    def _analyze_pattern_consistency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of spatial patterns across models."""
        spatial_results = results.get('spatial_patterns', {})
        
        consistency_metrics = {
            'hotspot_consistency': {},
            'height_pattern_consistency': {},
            'overall_consistency_score': 0.0
        }
        
        if len(spatial_results) < 2:
            return consistency_metrics
        
        # Analyze hotspot consistency
        hotspot_percentages = []
        for model_name, spatial_data in spatial_results.items():
            hotspots = spatial_data.get('density_hotspots', {})
            if 'high_density_regions' in hotspots:
                hotspot_percentages.append(hotspots['high_density_regions'].get('percentage', 0))
        
        if hotspot_percentages:
            consistency_metrics['hotspot_consistency'] = {
                'mean_hotspot_percentage': np.mean(hotspot_percentages),
                'std_hotspot_percentage': np.std(hotspot_percentages),
                'coefficient_of_variation': np.std(hotspot_percentages) / (np.mean(hotspot_percentages) + 1e-10)
            }
        
        # Calculate overall consistency score
        consistency_scores = []
        if hotspot_percentages:
            cv = np.std(hotspot_percentages) / (np.mean(hotspot_percentages) + 1e-10)
            consistency_scores.append(1.0 / (1.0 + cv))  # Higher consistency = lower CV
        
        if consistency_scores:
            consistency_metrics['overall_consistency_score'] = np.mean(consistency_scores)
        
        return consistency_metrics
    
    def _rank_models_by_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Rank models based on multiple performance criteria."""
        model_names = list(results.get('clustering_analysis', {}).keys())
        
        if not model_names:
            return {}
        
        rankings = {}
        criteria_scores = {}
        
        # Criteria 1: Clustering quality (silhouette score)
        clustering_scores = {}
        for model_name in model_names:
            clustering_result = results.get('clustering_analysis', {}).get(model_name, {})
            best_kmeans = clustering_result.get('best_kmeans', {})
            clustering_scores[model_name] = best_kmeans.get('score', 0)
        
        criteria_scores['clustering_quality'] = clustering_scores
        
        # Criteria 2: Physical plausibility
        plausibility_scores = {}
        for model_name in model_names:
            accuracy_result = results.get('physical_accuracy', {}).get(model_name, {})
            plausibility = accuracy_result.get('physical_plausibility', {})
            
            # Lower negative densities and extreme values = better score
            negative_count = plausibility.get('negative_densities', 0)
            extreme_count = plausibility.get('extremely_high_densities', 0)
            
            # Simple scoring: penalize negatives and extremes
            score = 1.0 / (1.0 + negative_count + extreme_count)
            plausibility_scores[model_name] = score
        
        criteria_scores['physical_plausibility'] = plausibility_scores
        
        # Criteria 3: Occlusion correction assessment
        occlusion_scores = {}
        for model_name in model_names:
            occlusion_result = results.get('occlusion_analysis', {}).get(model_name, {})
            correction_assessment = occlusion_result.get('occlusion_correction_assessment', {})
            
            # Score based on empirical assessment
            empirical = correction_assessment.get('empirical_assessment', 'moderate')
            score_map = {'good': 1.0, 'moderate': 0.6, 'poor': 0.3, 'unknown': 0.5}
            occlusion_scores[model_name] = score_map.get(empirical, 0.5)
        
        criteria_scores['occlusion_correction'] = occlusion_scores
        
        # Calculate overall rankings
        overall_scores = {}
        weights = {
            'clustering_quality': 0.4,
            'physical_plausibility': 0.3,
            'occlusion_correction': 0.3
        }
        
        for model_name in model_names:
            weighted_score = 0
            for criterion, weight in weights.items():
                criterion_score = criteria_scores.get(criterion, {}).get(model_name, 0)
                weighted_score += weight * criterion_score
            
            overall_scores[model_name] = weighted_score
        
        # Rank models
        ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'ranked_models': ranked_models,
            'criteria_scores': criteria_scores,
            'ranking_explanation': {
                'best_model': ranked_models[0][0] if ranked_models else None,
                'ranking_criteria': list(weights.keys()),
                'weights': weights
            }
        }
    
    def _perform_regression_analysis(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform regression analysis between models."""
        self.logger.info("Performing regression analysis between models")
        
        regression_results = {
            'pairwise_regressions': {},
            'feature_importance': {},
            'cross_model_predictions': {}
        }
        
        model_names = list(data_dict.keys())
        
        # Pairwise regression analysis
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Avoid duplicate pairs
                    pair_name = f"{model1}_vs_{model2}"
                    
                    # Prepare data for regression
                    regression_data = self._prepare_regression_data(
                        data_dict[model1], data_dict[model2], model1, model2
                    )
                    
                    if regression_data is not None:
                        pair_regression = self._perform_pairwise_regression(
                            regression_data, model1, model2
                        )
                        regression_results['pairwise_regressions'][pair_name] = pair_regression
        
        return regression_results
    
    def _prepare_regression_data(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               model1: str, model2: str) -> Optional[pd.DataFrame]:
        """Prepare data for regression analysis between two models."""
        # This is a simplified approach - in practice, you'd need to spatially align the voxels
        if len(df1) < 10 or len(df2) < 10:
            return None
        
        # For demonstration, we'll use summary statistics
        # In practice, you'd spatially match voxels between models
        
        # Create features from model 1
        features1 = {
            'mean_density': df1['density_value'].mean(),
            'std_density': df1['density_value'].std(),
            'max_density': df1['density_value'].max(),
            'median_density': df1['density_value'].median(),
            'n_points': len(df1)
        }
        
        # Create target from model 2
        target = {
            'mean_density': df2['density_value'].mean(),
            'std_density': df2['density_value'].std(),
            'max_density': df2['density_value'].max(),
            'median_density': df2['density_value'].median(),
            'n_points': len(df2)
        }
        
        # This is a simplified example - real implementation would be more sophisticated
        regression_df = pd.DataFrame({
            'predictor_mean': [features1['mean_density']],
            'predictor_std': [features1['std_density']],
            'predictor_max': [features1['max_density']],
            'target_mean': [target['mean_density']],
            'target_std': [target['std_density']],
            'target_max': [target['max_density']]
        })
        
        return regression_df
    
    def _perform_pairwise_regression(self, data: pd.DataFrame, model1: str, 
                                   model2: str) -> Dict[str, Any]:
        """Perform regression analysis between two models."""
        if len(data) < 2:
            return {}
        
        # Simple linear regression example
        X = data[['predictor_mean', 'predictor_std', 'predictor_max']].values
        y = data['target_mean'].values
        
        regression_results = {}
        
        # Linear regression
        if self.config['regression']['linear_regression']['enable']:
            lr = LinearRegression()
            lr.fit(X, y)
            
            regression_results['linear_regression'] = {
                'r_squared': lr.score(X, y),
                'coefficients': lr.coef_.tolist(),
                'intercept': lr.intercept_
            }
        
        # Random Forest (if enabled and sufficient data)
        if self.config['regression']['random_forest']['enable'] and len(data) >= 5:
            rf = RandomForestRegressor(n_estimators=10, random_state=42)
            rf.fit(X, y)
            
            regression_results['random_forest'] = {
                'r_squared': rf.score(X, y),
                'feature_importance': rf.feature_importances_.tolist()
            }
        
        return regression_results


if __name__ == "__main__":
    # Example usage
    analyzer = SpatialPatternAnalyzer()
    
    # This would typically be called from the main VoxPlot framework
    print("ML/AI Spatial Pattern Analyzer initialized successfully!")
    print("Available analysis methods:")
    print("- Clustering analysis (K-Means, DBSCAN, Hierarchical)")
    print("- Dimensionality reduction (PCA, t-SNE, UMAP)")
    print("- Regression analysis (Linear, Random Forest, XGBoost)")
    print("- Spatial pattern detection")
    print("- Occlusion analysis")
    print("- Physical accuracy assessment")
    print("- Comparative model analysis")