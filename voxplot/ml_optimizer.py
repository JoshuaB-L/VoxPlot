#!/usr/bin/env python3
"""
ML Optimization Module for VoxPlot Analysis Framework
====================================================

This module provides comprehensive optimization techniques for machine learning
analysis of 3D forest structure data, including:

- Intelligent sampling strategies (stratified spatial, density-aware, height-stratified)
- GPU acceleration support with CUDA
- Mini-batch and standard clustering algorithms
- Parallel processing optimization
- Memory and compute efficiency enhancements

Author: Claude Code & Joshua B-L
Date: 2025-09-12
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

# Core ML libraries
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# GPU acceleration support with enhanced detection
GPU_AVAILABLE = False
cp = None
cuml = None
cuKMeans = None
cuDBSCAN = None
GPU_ERROR_MSG = None

try:
    import cupy as cp
    # Test if CUDA is actually available
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU_CUDA_AVAILABLE = True
    except:
        GPU_CUDA_AVAILABLE = False
        GPU_ERROR_MSG = "CUDA runtime not available or no CUDA devices found"
    
    if GPU_CUDA_AVAILABLE:
        try:
            import cuml
            from cuml.cluster import KMeans as cuKMeans
            from cuml.cluster import DBSCAN as cuDBSCAN
            GPU_AVAILABLE = True
        except ImportError as e:
            GPU_ERROR_MSG = f"cuML library not available: {str(e)}"
    
except ImportError as e:
    GPU_ERROR_MSG = f"CuPy library not available: {str(e)}"

# Parallel processing
import multiprocessing as mp
from joblib import Parallel, delayed

warnings.filterwarnings('ignore', category=UserWarning)


class DataSampler:
    """
    Intelligent data sampling strategies for 3D forest structure data.
    
    Provides multiple sampling strategies optimized for voxel-based forest data:
    - Random sampling
    - Stratified spatial sampling 
    - Density-aware sampling
    - Height-stratified sampling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize sampler with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.random_seed = config.get('random_seed', 42)
        np.random.seed(self.random_seed)
        
    def sample_data(self, data: pd.DataFrame, max_size: int, 
                   strategy: str = "stratified_spatial") -> pd.DataFrame:
        """
        Sample data using specified strategy.
        
        Args:
            data: Input dataframe with voxel data
            max_size: Maximum number of samples to return
            strategy: Sampling strategy to use
            
        Returns:
            Sampled dataframe
        """
        if len(data) <= max_size:
            return data.copy()
            
        self.logger.info(f"Sampling {len(data)} voxels to {max_size} using {strategy} strategy")
        
        if strategy == "random":
            return self._random_sampling(data, max_size)
        elif strategy == "stratified_spatial":
            return self._stratified_spatial_sampling(data, max_size)
        elif strategy == "density_aware":
            return self._density_aware_sampling(data, max_size)
        elif strategy == "height_stratified":
            return self._height_stratified_sampling(data, max_size)
        else:
            self.logger.warning(f"Unknown sampling strategy: {strategy}. Using random sampling.")
            return self._random_sampling(data, max_size)
    
    def _random_sampling(self, data: pd.DataFrame, max_size: int) -> pd.DataFrame:
        """Simple random sampling."""
        return data.sample(n=max_size, random_state=self.random_seed)
    
    def _stratified_spatial_sampling(self, data: pd.DataFrame, max_size: int) -> pd.DataFrame:
        """
        Stratified spatial sampling maintaining 3D spatial distribution.
        
        Divides space into 3D grid cells and samples proportionally from each cell.
        """
        # Determine optimal grid resolution
        x_range = data['x'].max() - data['x'].min()
        y_range = data['y'].max() - data['y'].min() 
        z_range = data['z'].max() - data['z'].min()
        
        # Calculate grid dimensions for roughly equal-sized cells
        total_volume = x_range * y_range * z_range
        target_cells = min(100, max_size // 10)  # Target number of grid cells
        cell_volume = total_volume / target_cells
        cell_size = cell_volume ** (1/3)
        
        # Create spatial bins
        x_bins = max(1, int(x_range / cell_size))
        y_bins = max(1, int(y_range / cell_size))  
        z_bins = max(1, int(z_range / cell_size))
        
        # Assign voxels to spatial bins
        data_copy = data.copy()
        data_copy['x_bin'] = pd.cut(data_copy['x'], bins=x_bins, labels=False)
        data_copy['y_bin'] = pd.cut(data_copy['y'], bins=y_bins, labels=False)
        data_copy['z_bin'] = pd.cut(data_copy['z'], bins=z_bins, labels=False)
        
        # Create combined spatial bin identifier
        data_copy['spatial_bin'] = (
            data_copy['x_bin'].astype(str) + '_' + 
            data_copy['y_bin'].astype(str) + '_' + 
            data_copy['z_bin'].astype(str)
        )
        
        # Sample proportionally from each spatial bin
        bin_counts = data_copy['spatial_bin'].value_counts()
        samples = []
        
        for spatial_bin, count in bin_counts.items():
            bin_data = data_copy[data_copy['spatial_bin'] == spatial_bin]
            # Sample proportionally based on bin size
            n_samples = max(1, int((count / len(data)) * max_size))
            n_samples = min(n_samples, count)  # Don't oversample
            
            if n_samples > 0:
                bin_sample = bin_data.sample(n=n_samples, random_state=self.random_seed)
                samples.append(bin_sample)
        
        # Combine all samples
        sampled_data = pd.concat(samples, ignore_index=True)
        
        # If we have too many samples, randomly subsample
        if len(sampled_data) > max_size:
            sampled_data = sampled_data.sample(n=max_size, random_state=self.random_seed)
        
        # Remove temporary columns
        columns_to_drop = ['x_bin', 'y_bin', 'z_bin', 'spatial_bin']
        sampled_data = sampled_data.drop(columns=[col for col in columns_to_drop if col in sampled_data.columns])
        
        self.logger.info(f"Stratified spatial sampling: {len(bin_counts)} spatial bins, "
                        f"{len(sampled_data)} samples selected")
        
        return sampled_data
    
    def _density_aware_sampling(self, data: pd.DataFrame, max_size: int) -> pd.DataFrame:
        """
        Density-aware sampling that oversamples high-density regions.
        
        More forest structure information is typically in high-density voxels.
        """
        # Get density values
        density_col = 'density_value'
        if density_col not in data.columns:
            self.logger.warning("No density_value column found. Using random sampling.")
            return self._random_sampling(data, max_size)
        
        densities = data[density_col].values
        
        # Remove zero/negative densities for meaningful analysis
        valid_mask = densities > 0
        if valid_mask.sum() == 0:
            return self._random_sampling(data, max_size)
        
        valid_data = data[valid_mask].copy()
        valid_densities = valid_data[density_col].values
        
        # Create density-based sampling probabilities
        # Higher density = higher probability of selection
        density_percentiles = np.percentile(valid_densities, [25, 50, 75, 90])
        
        # Assign sampling weights based on density percentiles
        sampling_weights = np.ones(len(valid_data))
        sampling_weights[valid_densities >= density_percentiles[3]] *= 4.0  # 90th percentile: 4x weight
        sampling_weights[valid_densities >= density_percentiles[2]] *= 2.0  # 75th percentile: 2x weight
        sampling_weights[valid_densities >= density_percentiles[1]] *= 1.5  # 50th percentile: 1.5x weight
        
        # Normalize weights to probabilities
        sampling_probabilities = sampling_weights / sampling_weights.sum()
        
        # Sample based on density-weighted probabilities
        n_samples = min(max_size, len(valid_data))
        sample_indices = np.random.choice(
            len(valid_data), 
            size=n_samples, 
            replace=False,
            p=sampling_probabilities
        )
        
        sampled_data = valid_data.iloc[sample_indices].copy()
        
        self.logger.info(f"Density-aware sampling: prioritized high-density regions, "
                        f"{len(sampled_data)} samples selected")
        
        return sampled_data
    
    def _height_stratified_sampling(self, data: pd.DataFrame, max_size: int) -> pd.DataFrame:
        """
        Height-stratified sampling maintaining vertical forest structure.
        
        Samples proportionally from different canopy layers.
        """
        # Define height layers based on percentiles
        height_percentiles = np.percentile(data['z'], [33, 67])
        
        # Assign height layers
        data_copy = data.copy()
        conditions = [
            data_copy['z'] <= height_percentiles[0],
            (data_copy['z'] > height_percentiles[0]) & (data_copy['z'] <= height_percentiles[1]),
            data_copy['z'] > height_percentiles[1]
        ]
        layer_names = ['lower', 'middle', 'upper']
        data_copy['height_layer'] = np.select(conditions, layer_names)
        
        # Sample proportionally from each height layer
        layer_counts = data_copy['height_layer'].value_counts()
        samples = []
        
        for layer, count in layer_counts.items():
            layer_data = data_copy[data_copy['height_layer'] == layer]
            # Sample proportionally based on layer size
            n_samples = max(1, int((count / len(data)) * max_size))
            n_samples = min(n_samples, count)
            
            if n_samples > 0:
                layer_sample = layer_data.sample(n=n_samples, random_state=self.random_seed)
                samples.append(layer_sample)
        
        # Combine all samples
        sampled_data = pd.concat(samples, ignore_index=True)
        
        # If we have too many samples, randomly subsample
        if len(sampled_data) > max_size:
            sampled_data = sampled_data.sample(n=max_size, random_state=self.random_seed)
        
        # Remove temporary column
        sampled_data = sampled_data.drop(columns=['height_layer'])
        
        self.logger.info(f"Height-stratified sampling: {len(layer_counts)} height layers, "
                        f"{len(sampled_data)} samples selected")
        
        return sampled_data


class OptimizedClustering:
    """
    Optimized clustering algorithms with GPU acceleration and user-configurable options.
    
    Supports:
    - Standard K-Means
    - Mini-Batch K-Means  
    - GPU-accelerated clustering (cuML)
    - Parallel processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize optimizer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Extract performance config from full config  
        perf_config = config.get('analysis', {}).get('ml_analysis', {}).get('performance', {})
        self.compute_backend = perf_config.get('compute_backend', 'cpu')
        self.use_gpu = perf_config.get('use_gpu_acceleration', False) and GPU_AVAILABLE
        
        if self.use_gpu and not GPU_AVAILABLE:
            if GPU_ERROR_MSG:
                self.logger.info(f"GPU acceleration requested but not available: {GPU_ERROR_MSG}")
                self.logger.info("To enable GPU acceleration:")
                self.logger.info("1. Install CUDA Toolkit (11.x or 12.x)")
                self.logger.info("2. Install CuPy: pip install cupy-cuda11x (or cupy-cuda12x)")
                self.logger.info("3. Install cuML: pip install cuml")
                self.logger.info("Falling back to CPU processing.")
            else:
                self.logger.info("GPU acceleration requested but libraries not available. Using CPU.")
            self.use_gpu = False
            
    def perform_kmeans_clustering(self, X: np.ndarray, n_clusters: int, 
                                 algorithm: str = "standard") -> Dict[str, Any]:
        """
        Perform K-means clustering with specified algorithm.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            algorithm: Algorithm type ("standard", "mini_batch", "gpu_accelerated", "auto")
            
        Returns:
            Dictionary with clustering results
        """
        # Enhanced GPU prioritization
        if algorithm == "auto":
            if self.use_gpu and GPU_AVAILABLE:
                self.logger.info(f"Auto-selecting GPU-accelerated K-means for {len(X)} samples")
                return self._gpu_kmeans(X, n_clusters)
            elif len(X) > 50000:
                self.logger.info(f"Auto-selecting mini-batch K-means for {len(X)} samples")
                return self._mini_batch_kmeans(X, n_clusters)
            else:
                return self._standard_kmeans(X, n_clusters)
        elif algorithm == "gpu_accelerated" and self.use_gpu:
            self.logger.info(f"Using GPU-accelerated K-means for {len(X)} samples")
            return self._gpu_kmeans(X, n_clusters)
        elif algorithm == "mini_batch":
            return self._mini_batch_kmeans(X, n_clusters)
        else:
            return self._standard_kmeans(X, n_clusters)
    
    def _standard_kmeans(self, X: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Standard K-means clustering."""
        kmeans_config = self.config.get('ml_config', {}).get('clustering', {}).get('kmeans', {})
        n_init = kmeans_config.get('n_init', 10)
        max_iter = kmeans_config.get('max_iter', 300)
        perf_config = self.config.get('analysis', {}).get('ml_analysis', {}).get('performance', {})
        n_jobs = perf_config.get('n_jobs', -1)
        
        # Handle n_jobs parameter compatibility
        kmeans_params = {
            'n_clusters': n_clusters,
            'n_init': n_init,
            'max_iter': max_iter,
            'random_state': 42
        }
        
        # Only add n_jobs if greater than 1 (avoid sklearn compatibility issues)
        if n_jobs != 1:
            try:
                kmeans = KMeans(**kmeans_params, n_jobs=n_jobs)
            except TypeError:
                # Fallback for older sklearn versions
                kmeans = KMeans(**kmeans_params)
        else:
            kmeans = KMeans(**kmeans_params)
        
        labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score for evaluation
        score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'score': score,
            'algorithm': 'standard_kmeans',
            'n_clusters': n_clusters
        }
    
    def _mini_batch_kmeans(self, X: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """Mini-batch K-means clustering for large datasets."""
        kmeans_config = self.config.get('ml_config', {}).get('clustering', {}).get('kmeans', {})
        batch_size = kmeans_config.get('batch_size', 1000)
        max_iter = kmeans_config.get('max_iter', 300)
        max_no_improvement = kmeans_config.get('max_no_improvement', 10)
        
        # Ensure batch size is appropriate
        batch_size = min(batch_size, len(X) // 2)
        batch_size = max(batch_size, n_clusters * 2)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            max_no_improvement=max_no_improvement,
            random_state=42
        )
        
        labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score
        score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
        
        return {
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'score': score,
            'algorithm': 'mini_batch_kmeans',
            'n_clusters': n_clusters,
            'batch_size': batch_size
        }
    
    def _gpu_kmeans(self, X: np.ndarray, n_clusters: int) -> Dict[str, Any]:
        """GPU-accelerated K-means clustering using cuML."""
        if not self.use_gpu:
            return self._standard_kmeans(X, n_clusters)
        
        try:
            # Convert to GPU arrays with logging
            self.logger.info(f"Transferring {X.shape[0]} samples to GPU memory...")
            X_gpu = cp.asarray(X, dtype=cp.float32)
            gpu_memory = cp.cuda.Device().mem_info
            self.logger.info(f"GPU memory used: {(gpu_memory[1] - gpu_memory[0]) / 1024**3:.2f} GB / {gpu_memory[1] / 1024**3:.2f} GB")
            
            kmeans_config = self.config.get('ml_config', {}).get('clustering', {}).get('kmeans', {})
            max_iter = kmeans_config.get('max_iter', 300)
            
            # Initialize cuML K-means
            kmeans = cuKMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=42
            )
            
            # Fit and predict
            labels_gpu = kmeans.fit_predict(X_gpu)
            
            # Convert results back to CPU
            labels = cp.asnumpy(labels_gpu)
            centroids = cp.asnumpy(kmeans.cluster_centers_)
            
            # Calculate silhouette score on CPU (more stable)
            score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            
            return {
                'labels': labels,
                'centroids': centroids,
                'inertia': float(kmeans.inertia_),
                'score': score,
                'algorithm': 'gpu_kmeans',
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            self.logger.warning(f"GPU K-means failed: {e}. Falling back to CPU.")
            return self._standard_kmeans(X, n_clusters)


class MLOptimizer:
    """
    Main ML optimization coordinator integrating all optimization techniques.
    
    Provides unified interface for:
    - Data sampling
    - Algorithm optimization 
    - GPU acceleration
    - Parallel processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ML optimizer with full configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract configuration sections
        self.analysis_config = config.get('analysis', {})
        self.ml_analysis_config = self.analysis_config.get('ml_analysis', {})
        self.performance_config = self.ml_analysis_config.get('performance', {})
        
        # Initialize components
        self.sampler = DataSampler(self.ml_analysis_config)
        self.clustering = OptimizedClustering(self.config)  # Pass full config
        
        self.logger.info(f"ML Optimizer initialized with backend: {self.performance_config.get('compute_backend', 'cpu')}")
        if self.performance_config.get('use_gpu_acceleration', False):
            if GPU_AVAILABLE:
                self.logger.info("GPU acceleration enabled with cuML/CuPy")
            else:
                if GPU_ERROR_MSG:
                    self.logger.info(f"GPU acceleration requested but not available: {GPU_ERROR_MSG}")
                else:
                    self.logger.info("GPU acceleration requested but libraries not available. Using CPU optimizations.")
    
    def optimize_data_for_analysis(self, data: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Apply all data-level optimizations to prepare for ML analysis.
        
        Args:
            data: Raw voxel data
            model_name: Name of the model (for logging)
            
        Returns:
            Optimized data ready for ML analysis
        """
        original_size = len(data)
        
        # Apply sampling if enabled
        if self.ml_analysis_config.get('enable_sampling', False):
            max_size = self.ml_analysis_config.get('max_sample_size', 10000)
            sampling_strategy = self.ml_analysis_config.get('sampling_strategy', 'stratified_spatial')
            
            if original_size > max_size:
                data = self.sampler.sample_data(data, max_size, sampling_strategy)
                self.logger.info(f"{model_name}: Sampled {original_size} → {len(data)} voxels "
                               f"using {sampling_strategy} strategy")
        
        # Apply data type optimization if enabled
        if self.performance_config.get('data_type_optimization', False):
            data = self._optimize_data_types(data)
        
        return data
    
    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        optimized_data = data.copy()
        
        # Convert float64 to float32 for numeric columns (except indices)
        float_columns = optimized_data.select_dtypes(include=['float64']).columns
        for col in float_columns:
            optimized_data[col] = optimized_data[col].astype(np.float32)
        
        # Convert int64 to int32 where appropriate
        int_columns = optimized_data.select_dtypes(include=['int64']).columns
        for col in int_columns:
            if optimized_data[col].min() >= -2147483648 and optimized_data[col].max() <= 2147483647:
                optimized_data[col] = optimized_data[col].astype(np.int32)
        
        return optimized_data
    
    def get_optimal_clustering_algorithm(self, data_size: int) -> str:
        """
        Recommend optimal clustering algorithm based on data size and configuration.
        
        Args:
            data_size: Number of data points
            
        Returns:
            Recommended algorithm name
        """
        algorithm = self.config.get('ml_config', {}).get('clustering', {}).get('kmeans', {}).get('algorithm', 'standard')
        
        # Auto-select based on data size and GPU availability
        if algorithm == "auto":
            # Prioritize GPU acceleration when available
            if self.performance_config.get('use_gpu_acceleration', False) and GPU_AVAILABLE:
                self.logger.info(f"Auto-selecting GPU-accelerated clustering for {data_size} samples")
                return "gpu_accelerated"
            elif data_size > 50000:
                self.logger.info(f"Auto-selecting mini-batch clustering for {data_size} samples")
                return "mini_batch" 
            elif data_size > 10000:
                self.logger.info(f"Auto-selecting mini-batch clustering for {data_size} samples")
                return "mini_batch"
            else:
                self.logger.info(f"Auto-selecting standard clustering for {data_size} samples")
                return "standard"
        
        return algorithm