#!/usr/bin/env python3
"""
VoxPlot Example Usage Script

This script demonstrates various ways to use the VoxPlot system for 
forest structure analysis, including programmatic configuration,
batch processing, and custom analysis workflows.
"""

import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Import VoxPlot modules
from config_manager import ConfigManager, create_example_config
from main import VoxPlotAnalysis
from data_loader import DataLoader
from data_analyzer import ForestStructureAnalyzer
from utils import setup_logging


def create_synthetic_dataset(n_voxels: int = 1000, density_type: str = 'lad') -> pd.DataFrame:
    """
    Create a synthetic dataset for testing and demonstration purposes.
    
    This generates a realistic-looking forest structure with crown layers
    and spatial distribution patterns.
    """
    np.random.seed(42)  # For reproducible results
    
    # Create spatial coordinates
    x = np.random.uniform(-5, 5, n_voxels)
    y = np.random.uniform(-5, 5, n_voxels)
    
    # Create height with crown-like distribution
    # Ground level (0-2m): minimal density
    # Trunk level (2-5m): low density
    # Crown level (5-15m): high density with layers
    z = np.random.uniform(0, 15, n_voxels)
    
    # Create density values based on height and distance from center
    distance_from_center = np.sqrt(x**2 + y**2)
    
    # Height-based density profile
    density = np.zeros(n_voxels)
    
    # Crown region (higher density)
    crown_mask = z >= 5
    density[crown_mask] = np.random.exponential(1.5, np.sum(crown_mask))
    
    # Trunk region (lower density)
    trunk_mask = (z >= 2) & (z < 5)
    density[trunk_mask] = np.random.exponential(0.3, np.sum(trunk_mask))
    
    # Ground region (minimal density)
    ground_mask = z < 2
    density[ground_mask] = np.random.exponential(0.1, np.sum(ground_mask))
    
    # Apply radial decay (density decreases away from center)
    density *= np.exp(-distance_from_center / 3)
    
    # Add some noise and ensure non-negative values
    density += np.random.normal(0, 0.05, n_voxels)
    density = np.maximum(density, 0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y, 
        'z': z,
        density_type: density,
        'model_name': 'Synthetic',
        'model_type': 'synthetic',
        'density_type': density_type,
        'display_name': f'Synthetic_{density_type.upper()}'
    })
    
    return df


def example_1_basic_usage():
    """Example 1: Basic usage with configuration file."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage with Configuration File")
    print("="*60)
    
    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config_content = create_example_config()
        f.write(config_content)
        config_path = f.name
    
    try:
        # Load and modify configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(config_path)
        
        # Override with realistic test settings
        config['analysis']['output_dir'] = 'example_1_results'
        config['analysis']['density_types'] = ['lad']
        config['analysis']['crown_base_height'] = 2.0
        
        # Clear models and add synthetic data
        config['models'] = {}
        
        print("Configuration loaded successfully!")
        print(f"Output directory: {config['analysis']['output_dir']}")
        print(f"Density types: {config['analysis']['density_types']}")
        
    except Exception as e:
        print(f"Error in basic usage example: {e}")
    finally:
        # Clean up temporary file
        Path(config_path).unlink(missing_ok=True)


def example_2_programmatic_configuration():
    """Example 2: Programmatic configuration and analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Programmatic Configuration")
    print("="*60)
    
    # Create synthetic datasets
    datasets = {}
    model_configs = {}
    
    for model_name in ['Model_A', 'Model_B', 'Model_C']:
        # Create synthetic data files
        synthetic_data = create_synthetic_dataset(500, 'lad')
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            synthetic_data[['x', 'y', 'z', 'lad']].to_csv(f.name, index=False)
            temp_path = f.name
        
        datasets[model_name] = temp_path
        model_configs[model_name] = {
            'model_type': 'amapvox',
            'file_paths': {'lad': temp_path}
        }
    
    # Create configuration programmatically
    config = {
        'analysis': {
            'crown_base_height': 2.0,
            'voxel_size': 0.5,
            'min_density': 0.01,
            'comparison_mode': 'same_density_type_different_model_type',
            'density_types': ['lad'],
            'output_dir': 'example_2_results',
            'visualization': {
                'figsize_large': [15, 18],
                'figsize_medium': [12, 10],
                'figsize_small': [10, 6],
                'dpi': 150,
                'color_scale_max': 20.0
            }
        },
        'models': model_configs
    }
    
    try:
        # Run analysis
        analysis = VoxPlotAnalysis(config)
        print("Analysis initialized successfully!")
        print(f"Models configured: {list(model_configs.keys())}")
        print(f"Voxel size: {config['analysis']['voxel_size']} m")
        print(f"Crown base height: {config['analysis']['crown_base_height']} m")
        
        # Note: Actual analysis would be run with analysis.run_analysis()
        # Commented out to avoid creating large output in example
        # success = analysis.run_analysis()
        # print(f"Analysis completed: {success}")
        
    except Exception as e:
        print(f"Error in programmatic configuration: {e}")
    finally:
        # Clean up temporary files
        for temp_path in datasets.values():
            Path(temp_path).unlink(missing_ok=True)


def example_3_custom_analysis():
    """Example 3: Custom analysis using individual components."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Analysis Using Individual Components")
    print("="*60)
    
    try:
        # Create synthetic dataset
        df = create_synthetic_dataset(1000, 'lad')
        print(f"Created synthetic dataset with {len(df)} voxels")
        
        # Initialize analyzer
        analyzer = ForestStructureAnalyzer(voxel_size=0.5)
        
        # Perform crown layer analysis
        crown_analysis = analyzer.analyze_crown_layers(
            df, 'lad', crown_base_height=2.0, min_density=0.01
        )
        
        # Display results
        print("\nCrown Layer Analysis Results:")
        print("-" * 40)
        
        for layer_name, metrics in crown_analysis['metrics'].items():
            print(f"\n{layer_name.upper()} Layer:")
            print(f"  Total Area: {metrics['total_area']:.2f} m²")
            print(f"  Mean LAI: {metrics['mean_area_index']:.3f}")
            print(f"  Max LAI: {metrics['max_area_index']:.3f}")
            print(f"  Voxel Count: {metrics['voxel_count']:,}")
            print(f"  Grid Cells: {metrics['grid_cells']:,}")
        
        # Analyze vertical profile
        profile_analysis = analyzer.analyze_vertical_profile(
            df, 'lad', crown_base_height=2.0, bin_size=1.0
        )
        
        print(f"\nVertical Profile Analysis:")
        print(f"  Height range: {profile_analysis['height_range'][0]:.1f} - {profile_analysis['height_range'][1]:.1f} m")
        print(f"  Number of height bins: {len(profile_analysis['heights'])}")
        print(f"  Mean density across profile: {np.mean(profile_analysis['densities']):.3f}")
        
        # Spatial distribution analysis
        spatial_analysis = analyzer.analyze_spatial_distribution(
            df, 'lad', crown_base_height=2.0
        )
        
        print(f"\nSpatial Distribution Analysis:")
        spatial_stats = spatial_analysis['spatial_stats']
        print(f"  Total voxels: {spatial_stats['total_voxels']:,}")
        print(f"  Non-zero voxels: {spatial_stats['nonzero_voxels']:,}")
        print(f"  Density coverage: {spatial_stats['density_coverage']:.1%}")
        print(f"  Spatial extent: {spatial_stats['spatial_extent']['x_range']:.1f} x {spatial_stats['spatial_extent']['y_range']:.1f} m")
        print(f"  Total area: {spatial_stats['spatial_extent']['area']:.1f} m²")
        
    except Exception as e:
        print(f"Error in custom analysis: {e}")


def example_4_batch_processing():
    """Example 4: Batch processing multiple configurations."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing Multiple Configurations")
    print("="*60)
    
    # Define multiple analysis scenarios
    scenarios = [
        {
            'name': 'High_Resolution',
            'voxel_size': 0.25,
            'crown_base_height': 1.0,
            'min_density': 0.05
        },
        {
            'name': 'Medium_Resolution', 
            'voxel_size': 0.5,
            'crown_base_height': 2.0,
            'min_density': 0.02
        },
        {
            'name': 'Low_Resolution',
            'voxel_size': 1.0,
            'crown_base_height': 2.0,
            'min_density': 0.01
        }
    ]
    
    results_summary = []
    
    for scenario in scenarios:
        try:
            print(f"\nProcessing scenario: {scenario['name']}")
            
            # Create synthetic data
            df = create_synthetic_dataset(800, 'lad')
            
            # Initialize analyzer with scenario parameters
            analyzer = ForestStructureAnalyzer(voxel_size=scenario['voxel_size'])
            
            # Run analysis
            crown_analysis = analyzer.analyze_crown_layers(
                df, 'lad',
                crown_base_height=scenario['crown_base_height'],
                min_density=scenario['min_density']
            )
            
            # Collect results
            whole_metrics = crown_analysis['metrics']['whole']
            scenario_results = {
                'scenario': scenario['name'],
                'voxel_size': scenario['voxel_size'],
                'crown_base_height': scenario['crown_base_height'],
                'min_density': scenario['min_density'],
                'total_area': whole_metrics['total_area'],
                'mean_lai': whole_metrics['mean_area_index'],
                'voxel_count': whole_metrics['voxel_count'],
                'grid_cells': whole_metrics['grid_cells']
            }
            
            results_summary.append(scenario_results)
            print(f"  ✓ Completed - Total Area: {whole_metrics['total_area']:.2f} m², Mean LAI: {whole_metrics['mean_area_index']:.3f}")
            
        except Exception as e:
            print(f"  ✗ Failed - Error: {e}")
    
    # Display summary
    if results_summary:
        print(f"\nBatch Processing Summary:")
        print("-" * 80)
        header = f"{'Scenario':<15} {'Voxel':<8} {'Crown':<8} {'Min':<8} {'Total':<12} {'Mean':<8} {'Voxels':<10}"
        print(header)
        print(f"{'':15} {'Size':<8} {'Base':<8} {'Density':<8} {'Area':<12} {'LAI':<8} {'Count':<10}")
        print("-" * 80)
        
        for result in results_summary:
            row = (f"{result['scenario']:<15} "
                   f"{result['voxel_size']:<8.2f} "
                   f"{result['crown_base_height']:<8.1f} "
                   f"{result['min_density']:<8.3f} "
                   f"{result['total_area']:<12.2f} "
                   f"{result['mean_lai']:<8.3f} "
                   f"{result['voxel_count']:<10,}")
            print(row)


def example_5_data_validation():
    """Example 5: Data validation and quality assessment."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Data Validation and Quality Assessment")
    print("="*60)
    
    try:
        # Create datasets with different quality issues
        datasets = {
            'Good_Data': create_synthetic_dataset(1000, 'lad'),
            'Sparse_Data': create_synthetic_dataset(100, 'lad'),
            'Noisy_Data': create_synthetic_dataset(1000, 'lad')
        }
        
        # Add noise to noisy dataset
        datasets['Noisy_Data']['lad'] += np.random.normal(0, 0.5, len(datasets['Noisy_Data']))
        datasets['Noisy_Data']['lad'] = np.maximum(datasets['Noisy_Data']['lad'], 0)  # Keep non-negative
        
        # Create dataset with missing crown data
        low_data = datasets['Good_Data'].copy()
        low_data = low_data[low_data['z'] < 3]  # Remove crown data
        datasets['No_Crown_Data'] = low_data
        
        print("Validating datasets...")
        
        for name, df in datasets.items():
            print(f"\n{name}:")
            print(f"  Records: {len(df):,}")
            print(f"  Height range: {df['z'].min():.1f} - {df['z'].max():.1f} m")
            print(f"  Density range: {df['lad'].min():.3f} - {df['lad'].max():.3f}")
            print(f"  Non-zero density: {(df['lad'] > 0).sum():,} ({(df['lad'] > 0).mean():.1%})")
            print(f"  Crown voxels (>2m): {(df['z'] > 2).sum():,}")
            
            # Quick quality assessment
            if len(df) < 500:
                print("  ⚠️  WARNING: Low voxel count")
            if (df['lad'] > 0).mean() < 0.1:
                print("  ⚠️  WARNING: Very sparse density data")
            if (df['z'] > 5).sum() < 100:
                print("  ⚠️  WARNING: Limited crown data")
            if df['lad'].std() > df['lad'].mean():
                print("  ⚠️  WARNING: High density variability")
            
            # Try analysis to check for issues
            try:
                analyzer = ForestStructureAnalyzer(voxel_size=0.5)
                analysis = analyzer.analyze_crown_layers(df, 'lad', crown_base_height=2.0)
                if analysis['total_voxels'] == 0:
                    print("  ❌ ERROR: No crown voxels found")
                else:
                    print(f"  ✅ Analysis possible: {analysis['total_voxels']} crown voxels")
            except Exception as e:
                print(f"  ❌ ERROR: Analysis failed - {e}")
        
    except Exception as e:
        print(f"Error in data validation example: {e}")


def main():
    """Run all examples."""
    print("VoxPlot Example Usage Demonstrations")
    print("=" * 60)
    print("This script demonstrates various ways to use the VoxPlot system")
    print("for forest structure analysis.")
    
    # Set up logging
    logger = setup_logging(verbose=True)
    
    try:
        # Run examples
        example_1_basic_usage()
        example_2_programmatic_configuration()
        example_3_custom_analysis()
        example_4_batch_processing()
        example_5_data_validation()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nTo run a real analysis with your data:")
        print("1. Create a configuration file (see config.yaml example)")
        print("2. Update file paths to point to your data")
        print("3. Run: python main.py --config your_config.yaml")
        print("\nFor more information, see README.md")
        
    except Exception as e:
        print(f"\nExample script failed: {e}")
        logger.error(f"Example script error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())