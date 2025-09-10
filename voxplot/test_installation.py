#!/usr/bin/env python3
"""
VoxPlot Installation Test Script

This script validates that VoxPlot is correctly installed and all dependencies
are available. It performs basic functionality tests without requiring real data files.
"""

import sys
import tempfile
import traceback
from pathlib import Path
import warnings

# Suppress common warnings during testing
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    required_modules = [
        ('numpy', 'np'),
        ('pandas', 'pd'), 
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('scipy.stats', None),
        ('yaml', None)
    ]
    
    failed_imports = []
    
    for module_name, alias in required_modules:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name} - {e}")
            failed_imports.append(module_name)
    
    # Test VoxPlot modules
    voxplot_modules = [
        'config_manager',
        'data_loader', 
        'data_analyzer',
        'visualizer',
        'utils',
        'main'
    ]
    
    print("\nTesting VoxPlot modules...")
    for module_name in voxplot_modules:
        try:
            exec(f"import {module_name}")
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name} - {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n❌ Import test failed. Missing modules: {failed_imports}")
        return False
    else:
        print(f"\n✅ All imports successful!")
        return True


def test_config_manager():
    """Test configuration management functionality."""
    print("\nTesting configuration manager...")
    
    try:
        from config_manager import ConfigManager, create_example_config
        
        # Test example config creation
        example_config = create_example_config()
        if len(example_config) > 100:  # Basic sanity check
            print("  ✓ Example config creation")
        else:
            print("  ✗ Example config too short")
            return False
        
        # Test config manager initialization
        config_manager = ConfigManager()
        print("  ✓ ConfigManager initialization")
        
        # Test config validation with valid config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
analysis:
  crown_base_height: 0.7
  voxel_size: 0.25
  min_density: 0.05
  comparison_mode: "same_density_type_different_model_type"
  density_types: ["lad"]
  output_dir: "test_results"

models:
  TestModel:
    model_type: "amapvox"
    file_paths:
      lad: "/nonexistent/path/test.csv"
""")
            temp_config_path = f.name
        
        try:
            # This should fail due to nonexistent file, but config structure should be valid
            config_manager.load_config(temp_config_path)
            print("  ✗ Config validation should have failed due to missing file")
            return False
        except FileNotFoundError:
            print("  ✓ Config validation (correctly caught missing file)")
        except Exception as e:
            print(f"  ✗ Unexpected config validation error: {e}")
            return False
        finally:
            Path(temp_config_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Config manager test failed: {e}")
        return False


def test_data_processing():
    """Test data processing functionality with synthetic data."""
    print("\nTesting data processing...")
    
    try:
        import numpy as np
        import pandas as pd
        from data_loader import DataLoader
        from data_analyzer import ForestStructureAnalyzer
        from utils import validate_dataframe, normalize_height, calculate_statistics
        
        # Create synthetic test data
        np.random.seed(42)
        n_points = 500
        test_data = pd.DataFrame({
            'x': np.random.uniform(-5, 5, n_points),
            'y': np.random.uniform(-5, 5, n_points),
            'z': np.random.uniform(0, 15, n_points),
            'lad': np.random.exponential(1.0, n_points),
            'model_name': 'TestModel',
            'model_type': 'test',
            'density_type': 'lad',
            'display_name': 'TestModel_LAD'
        })
        
        # Test utilities
        try:
            validate_dataframe(test_data, ['x', 'y', 'z', 'lad'], 'test dataset')
            print("  ✓ DataFrame validation")
        except Exception as e:
            print(f"  ✗ DataFrame validation failed: {e}")
            return False
        
        # Test height normalization
        normalized_data = normalize_height(test_data, 'z')
        if normalized_data['z'].min() == 0.0:
            print("  ✓ Height normalization")
        else:
            print("  ✗ Height normalization failed")
            return False
        
        # Test statistics calculation
        stats = calculate_statistics(test_data['lad'])
        if all(key in stats for key in ['mean', 'std', 'min', 'max', 'count']):
            print("  ✓ Statistics calculation")
        else:
            print("  ✗ Statistics calculation failed")
            return False
        
        # Test forest structure analyzer
        analyzer = ForestStructureAnalyzer(voxel_size=0.5)
        
        # Test crown layer analysis
        crown_analysis = analyzer.analyze_crown_layers(
            test_data, 'lad', crown_base_height=1.0, min_density=0.01
        )
        
        if 'metrics' in crown_analysis and 'whole' in crown_analysis['metrics']:
            print("  ✓ Crown layer analysis")
        else:
            print("  ✗ Crown layer analysis failed")
            return False
        
        # Test vertical profile analysis
        profile_analysis = analyzer.analyze_vertical_profile(
            test_data, 'lad', crown_base_height=1.0, bin_size=1.0
        )
        
        if 'heights' in profile_analysis and len(profile_analysis['heights']) > 0:
            print("  ✓ Vertical profile analysis")
        else:
            print("  ✗ Vertical profile analysis failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Data processing test failed: {e}")
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from visualizer import ForestStructureVisualizer
        
        # Create test configuration
        test_config = {
            'analysis': {
                'visualization': {
                    'figsize_large': [12, 14],
                    'figsize_medium': [10, 8],
                    'figsize_small': [8, 6],
                    'dpi': 100,
                    'color_scale_max': 20.0
                }
            }
        }
        
        # Test visualizer initialization
        visualizer = ForestStructureVisualizer(test_config)
        print("  ✓ Visualizer initialization")
        
        # Test empty figure creation
        empty_fig = visualizer._create_empty_figure("Test message")
        if empty_fig is not None:
            print("  ✓ Empty figure creation")
            plt.close(empty_fig)
        else:
            print("  ✗ Empty figure creation failed")
            return False
        
        # Test density label function
        label = visualizer._get_density_label('lad')
        if label == 'Leaf Area':
            print("  ✓ Density label function")
        else:
            print("  ✗ Density label function failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Visualization test failed: {e}")
        return False


def test_file_operations():
    """Test file operations and output management."""
    print("\nTesting file operations...")
    
    try:
        from utils import ensure_directory
        from visualizer import ResultsManager
        import tempfile
        import shutil
        
        # Test directory creation
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir) / "test_subdir"
            created_dir = ensure_directory(test_dir)
            
            if created_dir.exists() and created_dir.is_dir():
                print("  ✓ Directory creation")
            else:
                print("  ✗ Directory creation failed")
                return False
        
        # Test results manager
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = {
                'analysis': {
                    'visualization': {
                        'figsize_large': [10, 12],
                        'dpi': 100,
                        'color_scale_max': 20.0
                    }
                }
            }
            
            results_manager = ResultsManager(temp_dir, test_config)
            
            # Check that subdirectories were created
            expected_dirs = ['figures', 'data', 'reports']
            for dir_name in expected_dirs:
                dir_path = results_manager.output_dir / dir_name
                if dir_path.exists():
                    print(f"  ✓ {dir_name} directory creation")
                else:
                    print(f"  ✗ {dir_name} directory creation failed")
                    return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ File operations test failed: {e}")
        return False


def test_integration():
    """Test basic integration without real files."""
    print("\nTesting integration...")
    
    try:
        from main import VoxPlotAnalysis
        import tempfile
        
        # Create minimal valid configuration
        config = {
            'analysis': {
                'crown_base_height': 1.0,
                'voxel_size': 0.5,
                'min_density': 0.01,
                'comparison_mode': 'same_density_type_different_model_type',
                'density_types': ['lad'],
                'output_dir': 'test_integration_output',
                'visualization': {
                    'figsize_large': [10, 12],
                    'figsize_medium': [8, 10],
                    'figsize_small': [6, 8],
                    'dpi': 100,
                    'color_scale_max': 20.0
                }
            },
            'models': {}  # Empty models - this should be handled gracefully
        }
        
        # Test analysis initialization
        analysis = VoxPlotAnalysis(config)
        print("  ✓ VoxPlotAnalysis initialization")
        
        # Note: We don't run the full analysis since we have no data files
        # but initialization should work
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and provide summary."""
    print("VoxPlot Installation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Manager", test_config_manager),
        ("Data Processing", test_data_processing),
        ("Visualization", test_visualization),
        ("File Operations", test_file_operations),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ✗ Test suite error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("VoxPlot is correctly installed and ready to use.")
        print("\nNext steps:")
        print("1. Create a configuration file (see config.yaml example)")
        print("2. Update file paths to point to your data")
        print("3. Run: python main.py --config your_config.yaml")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED")
        print("Some components may not work correctly.")
        print("Please check the error messages above and:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that all VoxPlot modules are in the correct location")
        print("3. Verify Python version compatibility (3.8+)")
        return False


def main():
    """Main entry point for test script."""
    try:
        success = run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n\nTest suite failed with unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())