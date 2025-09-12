#!/usr/bin/env python3
"""
Debug visualization test to isolate array ambiguity error.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add current directory to path
import sys
sys.path.insert(0, '.')

from ml_visualizer import MLVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dummy_ml_results():
    """Create dummy ML results that might trigger the array ambiguity error."""
    
    # Create dummy data that could cause issues
    outlier_percentages = np.array([5.2, 10.1, 3.7])  # Multi-element array
    zero_density_percentages = np.array([2.1, 8.5, 12.3])  # Multi-element array  
    negative_densities = np.array([0.1, 0.5, 0.2])  # Multi-element array
    shadow_percentages = np.array([15.2, 22.8, 8.9])  # Multi-element array
    
    # Create data matrix that could cause issues
    data_matrix = np.array([
        [0.5, 0.8, 0.3],
        [0.2, 0.9, 0.6],
        [0.7, 0.1, 0.4]
    ])
    
    # Create similarity matrix
    similarity_matrix = np.array([
        [1.0, 0.3, 0.6],
        [0.3, 1.0, 0.8],
        [0.6, 0.8, 1.0]
    ])
    
    return {
        'clustering_analysis': {
            'AmapVox_TLS_pad': {
                'kmeans': {3: {'silhouette_score': 0.5}},
                'best_kmeans': {'score': 0.5, 'n_clusters': 3}
            },
            'AmapVox_TLS_wad': {
                'kmeans': {5: {'silhouette_score': 0.3}},
                'best_kmeans': {'score': 0.3, 'n_clusters': 5}
            }
        },
        'dimensionality_reduction': {
            'AmapVox_TLS_pad': {
                'pca': {'explained_variance_ratio': [0.6, 0.3, 0.1]}
            }
        },
        'spatial_patterns': {
            'AmapVox_TLS_pad': {
                'height_stratified_density': {
                    'data_matrix': data_matrix,
                    'models': ['AmapVox_TLS', 'VoxLAD_TLS', 'VoxPy'],
                    'height_layers': ['lower', 'middle', 'upper']
                }
            }
        },
        'physical_accuracy': {
            'AmapVox_TLS_pad': {
                'outlier_percentage': outlier_percentages,
                'zero_density_percentage': zero_density_percentages,
                'negative_density_count': negative_densities
            },
            'AmapVox_TLS_wad': {
                'outlier_percentage': outlier_percentages,
                'zero_density_percentage': zero_density_percentages, 
                'negative_density_count': negative_densities
            }
        },
        'occlusion_analysis': {
            'AmapVox_TLS_pad': {
                'shadow_percentage': shadow_percentages
            },
            'AmapVox_TLS_wad': {
                'shadow_percentage': shadow_percentages
            }
        },
        'comparative_analysis': {
            'similarity_analysis': {
                'similarity_matrix': similarity_matrix,
                'model_names': ['AmapVox_TLS', 'VoxLAD_TLS', 'VoxPy']
            }
        }
    }

def test_individual_plots(visualizer, ml_results):
    """Test each plot function individually to isolate the error."""
    
    plot_tests = [
        ("clustering comparison", lambda ax: visualizer._plot_clustering_comparison(ax, ml_results)),
        ("cluster characteristics", lambda ax: visualizer._plot_cluster_characteristics(ax, ml_results)),
        ("PCA analysis", lambda ax: visualizer._plot_pca_analysis(ax, ml_results)),
        ("feature importance", lambda ax: visualizer._plot_feature_importance(ax, ml_results)),
        ("explained variance", lambda ax: visualizer._plot_explained_variance(ax, ml_results)),
        ("height stratified patterns", lambda ax: visualizer._plot_height_stratified_patterns(ax, ml_results)),
        ("physical accuracy assessment", lambda ax: visualizer._plot_physical_accuracy_assessment(ax, ml_results)),
        ("occlusion analysis", lambda ax: visualizer._plot_occlusion_analysis(ax, ml_results)),
        ("model similarity matrix", lambda ax: visualizer._plot_model_similarity_matrix(ax, ml_results)),
        ("performance rankings", lambda ax: visualizer._plot_performance_rankings(ax, ml_results))
    ]
    
    for name, plot_func in plot_tests:
        try:
            print(f"Testing {name}...")
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_func(ax)
            plt.close(fig)
            print(f"✅ {name} - SUCCESS")
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            if "ambiguous" in str(e).lower():
                print(f"🎯 FOUND THE CULPRIT: {name}")
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                return name
    
    return None

def main():
    """Main debug test function."""
    print("🔍 DEBUG: Testing individual visualization functions for array ambiguity")
    
    # Load real config to match actual conditions
    import yaml
    config = yaml.safe_load(open('config_ml.yaml'))
    viz_config = config.get('visualization', {})
    
    # Create visualizer and dummy data
    visualizer = MLVisualizer(viz_config)
    ml_results = create_dummy_ml_results()
    
    # Test individual plots
    culprit = test_individual_plots(visualizer, ml_results)
    
    if culprit:
        print(f"\n🎯 IDENTIFIED: The array ambiguity error is in '{culprit}' function")
    else:
        print("\n❓ No array ambiguity error found in individual tests")
        print("Testing full dashboard creation...")
        
        try:
            fig = visualizer.create_comprehensive_ml_dashboard(ml_results)
            plt.close(fig)
            print("✅ Full dashboard creation succeeded")
        except Exception as e:
            print(f"❌ Full dashboard creation failed: {e}")
            if "ambiguous" in str(e).lower():
                import traceback
                print("Full traceback:")
                traceback.print_exc()

if __name__ == "__main__":
    main()