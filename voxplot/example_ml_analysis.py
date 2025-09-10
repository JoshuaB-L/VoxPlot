#!/usr/bin/env python3
"""
Example ML Analysis Script for VoxPlot
======================================
Demonstrates how to use the ML-enhanced VoxPlot analysis framework
for comprehensive spatial pattern analysis of 3D forest structure data.

This example shows:
- Basic ML analysis workflow
- Custom configuration setup
- Programmatic access to results
- Advanced visualization options
- Integration with existing VoxPlot workflows

Author: Joshua B-L & Claude Code
Date: 2025-09-10
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add VoxPlot modules to path
sys.path.append(str(Path(__file__).parent))

from ml_main import MLEnhancedVoxPlotAnalysis
from ml_analyzer import SpatialPatternAnalyzer
from ml_visualizer import MLVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_ml_analysis():
    """
    Example 1: Basic ML analysis using configuration file.
    This is the simplest way to run ML analysis.
    """
    logger.info("=== Example 1: Basic ML Analysis ===")
    
    # Configuration file path (assumes config_ml.yaml exists)
    config_path = "config_ml.yaml"
    
    # Check if config exists
    if not Path(config_path).exists():
        logger.error(f"Configuration file {config_path} not found")
        logger.info("Please ensure config_ml.yaml is in the current directory")
        return None
    
    try:
        # Initialize ML analysis system
        ml_analysis = MLEnhancedVoxPlotAnalysis(
            config_path=config_path,
            verbose=True
        )
        
        # Run complete analysis for all models
        results = ml_analysis.run_complete_ml_analysis()
        
        if results:
            logger.info("✓ Basic ML analysis completed successfully")
            return results
        else:
            logger.error("✗ Basic ML analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"Basic ML analysis failed: {e}")
        return None


def example_selective_analysis():
    """
    Example 2: Selective analysis - analyze specific models and methods.
    """
    logger.info("=== Example 2: Selective Analysis ===")
    
    config_path = "config_ml.yaml"
    
    if not Path(config_path).exists():
        logger.warning(f"Configuration file {config_path} not found, skipping selective analysis")
        return None
    
    try:
        # Initialize system
        ml_analysis = MLEnhancedVoxPlotAnalysis(
            config_path=config_path,
            verbose=False
        )
        
        # Run analysis for specific models only
        specific_models = ["AmapVox_TLS", "VoxLAD_TLS"]
        
        # Run specific analysis types
        specific_analyses = ["clustering", "spatial_patterns", "comparative_analysis"]
        
        results = ml_analysis.run_complete_ml_analysis(
            models=specific_models,
            analysis_types=specific_analyses
        )
        
        if results:
            logger.info("✓ Selective analysis completed successfully")
            # Print some key results
            _print_key_results(results, specific_models)
            return results
        else:
            logger.error("✗ Selective analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"Selective analysis failed: {e}")
        return None


def example_programmatic_analysis():
    """
    Example 3: Programmatic analysis using components directly.
    This gives you full control over the analysis process.
    """
    logger.info("=== Example 3: Programmatic Analysis ===")
    
    try:
        # Create synthetic test data for demonstration
        test_data = _create_synthetic_test_data()
        
        # Initialize ML analyzer with custom configuration
        custom_config = {
            'clustering': {
                'kmeans_clusters': [3, 5, 8],
                'dbscan_eps': [1.0, 2.0],
                'dbscan_min_samples': [5, 10]
            },
            'spatial_analysis': {
                'height_layers': ['whole', 'upper', 'lower'],
                'clumping_threshold': 0.1,
                'neighborhood_radius': 1.5
            }
        }
        
        # Initialize analyzer
        analyzer = SpatialPatternAnalyzer(custom_config)
        
        # Run analysis
        logger.info("Running spatial pattern analysis on synthetic data...")
        results = analyzer.analyze_spatial_patterns(test_data)
        
        if results:
            logger.info("✓ Programmatic analysis completed successfully")
            
            # Create custom visualizations
            visualizer = MLVisualizer()
            
            # Create specific visualizations
            logger.info("Creating custom visualizations...")
            
            # Create comprehensive dashboard
            fig = visualizer.create_comprehensive_ml_dashboard(
                results, 
                test_data,
                output_path=Path("example_outputs")
            )
            
            # Create 3D clustering visualization for one model
            if test_data:
                model_name = list(test_data.keys())[0]
                fig_3d = visualizer.create_3d_clustering_visualization(
                    test_data,
                    results,
                    model_name,
                    output_path=Path("example_outputs")
                )
            
            logger.info("✓ Custom visualizations created")
            return results
        else:
            logger.error("✗ Programmatic analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"Programmatic analysis failed: {e}")
        return None


def example_custom_configuration():
    """
    Example 4: Using custom configuration programmatically.
    """
    logger.info("=== Example 4: Custom Configuration ===")
    
    try:
        # Create custom configuration
        custom_config = {
            'analysis': {
                'crown_base_height': 4.0,
                'voxel_size': 0.5,
                'min_density': 0.005,
                'output_dir': 'custom_ml_results',
                'density_types': ['lad', 'pad']
            },
            'visualization': {
                'dpi': 300,
                'figsize_large': [16, 20],
                'colors': {
                    'model_palette': ['#FF5733', '#33C1FF', '#33FF57', '#FF33C1'],
                    'sequential_cmap': 'plasma'
                }
            }
        }
        
        # Initialize analyzer with custom config
        analyzer = SpatialPatternAnalyzer(custom_config)
        visualizer = MLVisualizer(custom_config)
        
        # Create test data
        test_data = _create_synthetic_test_data(n_models=2)
        
        # Run analysis
        results = analyzer.analyze_spatial_patterns(test_data)
        
        if results:
            logger.info("✓ Custom configuration analysis completed")
            
            # Generate visualizations with custom styling
            fig = visualizer.create_comprehensive_ml_dashboard(
                results,
                test_data,
                output_path=Path("custom_outputs")
            )
            
            return results
        else:
            logger.error("✗ Custom configuration analysis failed")
            return None
            
    except Exception as e:
        logger.error(f"Custom configuration analysis failed: {e}")
        return None


def _create_synthetic_test_data(n_models=3, n_points_per_model=1000):
    """
    Create synthetic 3D forest structure data for testing.
    
    Args:
        n_models: Number of models to simulate
        n_points_per_model: Number of voxels per model
        
    Returns:
        Dictionary of synthetic data
    """
    logger.info(f"Creating synthetic test data ({n_models} models, {n_points_per_model} points each)")
    
    np.random.seed(42)  # For reproducible results
    test_data = {}
    
    model_names = [f"TestModel_{i+1}" for i in range(n_models)]
    
    for i, model_name in enumerate(model_names):
        # Create 3D grid with some clustering patterns
        x = np.random.normal(50, 20, n_points_per_model)
        y = np.random.normal(50, 20, n_points_per_model)
        z = np.random.exponential(10, n_points_per_model) + 1  # Height distribution
        
        # Create density patterns (different for each model)
        if i == 0:  # Model 1: High density in upper canopy
            density = np.where(z > 15, 
                             np.random.exponential(0.5), 
                             np.random.exponential(0.1))
        elif i == 1:  # Model 2: Uniform density
            density = np.random.exponential(0.3, n_points_per_model)
        else:  # Model 3+: Clustered density
            # Add some spatial clustering
            cluster_centers = np.random.rand(5, 2) * 100  # 5 random cluster centers
            density = np.zeros(n_points_per_model)
            
            for j in range(n_points_per_model):
                # Distance to nearest cluster center
                distances = np.sqrt((x[j] - cluster_centers[:, 0])**2 + 
                                  (y[j] - cluster_centers[:, 1])**2)
                min_dist = np.min(distances)
                # Higher density near cluster centers
                density[j] = np.random.exponential(0.8 / (1 + min_dist/10))
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'density_value': density
        })
        
        # Filter out extreme outliers
        df = df[
            (df['x'] >= 0) & (df['x'] <= 100) &
            (df['y'] >= 0) & (df['y'] <= 100) &
            (df['z'] >= 0) & (df['z'] <= 50) &
            (df['density_value'] >= 0) & (df['density_value'] <= 5)
        ]
        
        test_data[model_name] = df
        logger.info(f"Created {len(df)} synthetic voxels for {model_name}")
    
    return test_data


def _print_key_results(results, models=None):
    """Print key results from ML analysis."""
    logger.info("=== Key Results Summary ===")
    
    # Clustering results
    clustering_results = results.get('clustering_analysis', {})
    if clustering_results:
        logger.info("Clustering Analysis:")
        for model_name in (models or clustering_results.keys()):
            if model_name in clustering_results:
                model_clustering = clustering_results[model_name]
                best_kmeans = model_clustering.get('best_kmeans', {})
                if best_kmeans:
                    logger.info(f"  {model_name}: Best K={best_kmeans.get('n_clusters', 'N/A')}, "
                               f"Silhouette={best_kmeans.get('score', 0):.3f}")
    
    # Spatial patterns
    spatial_results = results.get('spatial_patterns', {})
    if spatial_results:
        logger.info("Spatial Patterns:")
        for model_name in (models or spatial_results.keys()):
            if model_name in spatial_results:
                spatial_data = spatial_results[model_name]
                hotspots = spatial_data.get('density_hotspots', {})
                if 'high_density_regions' in hotspots:
                    high_pct = hotspots['high_density_regions'].get('percentage', 0)
                    logger.info(f"  {model_name}: {high_pct:.1f}% high-density regions")
    
    # Model rankings
    comparative_results = results.get('comparative_analysis', {})
    if comparative_results:
        performance_ranking = comparative_results.get('performance_ranking', {})
        if 'ranked_models' in performance_ranking:
            ranked_models = performance_ranking['ranked_models']
            logger.info("Model Performance Rankings:")
            for i, (model_name, score) in enumerate(ranked_models[:3]):  # Top 3
                logger.info(f"  {i+1}. {model_name}: {score:.3f}")


def run_all_examples():
    """Run all examples in sequence."""
    logger.info("Running all ML analysis examples...")
    
    examples = [
        ("Basic ML Analysis", example_basic_ml_analysis),
        ("Selective Analysis", example_selective_analysis),
        ("Programmatic Analysis", example_programmatic_analysis),
        ("Custom Configuration", example_custom_configuration)
    ]
    
    results = {}
    
    for name, example_func in examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info('='*60)
        
        try:
            result = example_func()
            results[name] = result
            
            if result:
                logger.info(f"✓ {name} completed successfully")
            else:
                logger.warning(f"⚠ {name} completed with warnings")
                
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")
            results[name] = None
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EXAMPLES SUMMARY")
    logger.info('='*60)
    
    successful = sum(1 for result in results.values() if result is not None)
    total = len(results)
    
    logger.info(f"Completed: {successful}/{total} examples")
    
    for name, result in results.items():
        status = "✓ SUCCESS" if result else "✗ FAILED"
        logger.info(f"  {name}: {status}")
    
    return results


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        example_name = sys.argv[1].lower()
        
        if example_name == "basic":
            example_basic_ml_analysis()
        elif example_name == "selective":
            example_selective_analysis()
        elif example_name == "programmatic":
            example_programmatic_analysis()
        elif example_name == "custom":
            example_custom_configuration()
        elif example_name == "all":
            run_all_examples()
        else:
            print("Usage: python example_ml_analysis.py [basic|selective|programmatic|custom|all]")
            print("\nAvailable examples:")
            print("  basic        - Basic ML analysis using configuration file")
            print("  selective    - Analyze specific models and methods")
            print("  programmatic - Direct use of ML components")
            print("  custom       - Custom configuration example")
            print("  all          - Run all examples")
    else:
        # Default: run programmatic example (doesn't require config file)
        logger.info("Running default programmatic example...")
        logger.info("Use 'python example_ml_analysis.py all' to run all examples")
        example_programmatic_analysis()