#!/usr/bin/env python3
"""
Test Script for Statistical Analysis Framework
==============================================
This script tests the statistical analysis modules to ensure they work correctly
with the VoxPlot codebase.
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from statistical_analyzer import StatisticalAnalyzer, ModelComparison
from statistical_visualizer import StatisticalVisualizer


class TestStatisticalAnalyzer(unittest.TestCase):
    """Test cases for the StatisticalAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = StatisticalAnalyzer(alpha=0.05, bootstrap_n=100)
        
        # Create sample voxel data
        np.random.seed(42)
        n_voxels = 1000
        
        # Create sample data for three models
        self.model_data = {}
        
        # Model 1: Close to ground truth
        x = np.random.uniform(-10, 10, n_voxels)
        y = np.random.uniform(-10, 10, n_voxels)
        z = np.random.uniform(0, 20, n_voxels)
        lad = np.random.exponential(0.5, n_voxels) * (20 - z) / 20  # Decrease with height
        
        self.model_data['AmapVox_TLS'] = pd.DataFrame({
            'x': x, 'y': y, 'z': z,
            'lad': lad,
            'model_name': 'AmapVox_TLS',
            'model_type': 'amapvox',
            'density_type': 'lad',
            'display_name': 'AmapVox TLS LAD'
        })
        
        # Model 2: Slightly overestimating
        self.model_data['VoxLAD_TLS'] = pd.DataFrame({
            'x': x + np.random.normal(0, 0.1, n_voxels),
            'y': y + np.random.normal(0, 0.1, n_voxels),
            'z': z,
            'lad': lad * 1.2 + np.random.normal(0, 0.1, n_voxels),
            'model_name': 'VoxLAD_TLS',
            'model_type': 'voxlad',
            'density_type': 'lad',
            'display_name': 'VoxLAD TLS LAD'
        })
        
        # Model 3: Underestimating
        self.model_data['VoxPy_Combined'] = pd.DataFrame({
            'x': x + np.random.normal(0, 0.1, n_voxels),
            'y': y + np.random.normal(0, 0.1, n_voxels),
            'z': z,
            'lad': lad * 0.8 + np.random.normal(0, 0.05, n_voxels),
            'model_name': 'VoxPy_Combined',
            'model_type': 'voxpy',
            'density_type': 'lad',
            'display_name': 'VoxPy Combined LAD'
        })
        
        # Create sample ground truth data
        self.ground_truth = pd.DataFrame({
            'Method': ['Litter Fall', 'Solariscope', 'HemispheR'],
            'Method Type': ['Ground Truth', 'Ground Truth', 'Ground Truth'],
            'LAI': [2.19, 1.96, 1.98],
            'PAI': [np.nan, 2.70, 2.50],
            'WAI': [np.nan, 0.74, 0.52]
        })
    
    def test_reference_value_extraction(self):
        """Test extraction of reference values from ground truth."""
        # Test LAI (should use Litter Fall)
        lai_ref = self.analyzer._get_reference_value(self.ground_truth, 'LAI')
        self.assertAlmostEqual(lai_ref, 2.19, places=2)
        
        # Test WAI (should use mean of available values)
        wai_ref = self.analyzer._get_reference_value(self.ground_truth, 'WAI')
        expected_wai = np.mean([0.74, 0.52])
        self.assertAlmostEqual(wai_ref, expected_wai, places=2)
    
    def test_integrated_indices_calculation(self):
        """Test calculation of integrated area indices from voxel data."""
        indices = self.analyzer._calculate_integrated_indices(self.model_data, 'lad')
        
        # Check that all models have indices
        self.assertEqual(len(indices), 3)
        self.assertIn('AmapVox_TLS', indices)
        self.assertIn('VoxLAD_TLS', indices)
        self.assertIn('VoxPy_Combined', indices)
        
        # Check that indices are positive
        for model, index in indices.items():
            self.assertGreater(index, 0, f"Index for {model} should be positive")
    
    def test_statistical_tests(self):
        """Test statistical significance tests."""
        indices = {'Model1': 2.1, 'Model2': 2.3, 'Model3': 1.9}
        reference = 2.19
        
        results = self.analyzer._perform_statistical_tests(indices, reference)
        
        # Check that test results exist
        self.assertIn('one_sample_ttest', results)
        self.assertIn('individual_tests', results)
        
        # Check individual model tests
        for model in indices:
            self.assertIn(model, results['individual_tests'])
            model_test = results['individual_tests'][model]
            self.assertIn('value', model_test)
            self.assertIn('difference', model_test)
            self.assertIn('percent_difference', model_test)
    
    def test_error_metrics_calculation(self):
        """Test calculation of error metrics."""
        indices = {'Model1': 2.1, 'Model2': 2.3, 'Model3': 1.9}
        reference = 2.19
        
        metrics = self.analyzer._calculate_error_metrics(indices, reference)
        
        # Check that all models have metrics
        self.assertEqual(len(metrics), 3)
        
        # Check metric types
        for model, model_metrics in metrics.items():
            self.assertIn('mae', model_metrics)
            self.assertIn('rmse', model_metrics)
            self.assertIn('bias', model_metrics)
            self.assertIn('percent_error', model_metrics)
            self.assertIn('rank', model_metrics)
            
            # MAE should be non-negative
            self.assertGreaterEqual(model_metrics['mae'], 0)
    
    def test_model_agreement(self):
        """Test model agreement assessment."""
        agreement = self.analyzer._assess_model_agreement(self.model_data, 'lad')
        
        self.assertIn('pairwise', agreement)
        
        # Should have pairwise comparisons for 3 models
        if 'pairwise' in agreement:
            # C(3,2) = 3 pairwise comparisons
            self.assertLessEqual(len(agreement['pairwise']), 3)
    
    def test_spatial_error_analysis(self):
        """Test spatial error analysis."""
        spatial = self.analyzer._analyze_spatial_errors(self.model_data, 'lad')
        
        # Should have comparisons between models
        self.assertIsInstance(spatial, dict)
        
        # Check structure of spatial analysis results
        for comparison, analysis in spatial.items():
            self.assertIn('mean_error', analysis)
            self.assertIn('std_error', analysis)
            self.assertIn('error_by_height', analysis)
    
    def test_bootstrap_confidence(self):
        """Test bootstrap confidence interval calculation."""
        indices = {'Model1': 2.1, 'Model2': 2.3}
        reference = 2.19
        
        bootstrap = self.analyzer._calculate_bootstrap_confidence(indices, reference)
        
        # Check that all models have bootstrap results
        self.assertEqual(len(bootstrap), 2)
        
        for model, ci_info in bootstrap.items():
            self.assertIn('mean_estimate', ci_info)
            self.assertIn('ci_95_lower', ci_info)
            self.assertIn('ci_95_upper', ci_info)
            self.assertIn('includes_reference', ci_info)
            
            # CI should contain the mean
            self.assertLessEqual(ci_info['ci_95_lower'], ci_info['mean_estimate'])
            self.assertGreaterEqual(ci_info['ci_95_upper'], ci_info['mean_estimate'])
    
    def test_complete_comparison(self):
        """Test complete comparison to ground truth."""
        results = self.analyzer.compare_to_ground_truth(
            self.model_data,
            self.ground_truth,
            'lad'
        )
        
        # Check main result sections
        self.assertIn('reference_value', results)
        self.assertIn('model_indices', results)
        self.assertIn('statistical_tests', results)
        self.assertIn('error_metrics', results)
        self.assertIn('agreement_analysis', results)
        self.assertIn('spatial_analysis', results)
        self.assertIn('bootstrap_ci', results)


class TestModelComparison(unittest.TestCase):
    """Test cases for the ModelComparison class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparison = ModelComparison()
        
        # Create sample multi-density data
        np.random.seed(42)
        n_voxels = 500
        
        x = np.random.uniform(-10, 10, n_voxels)
        y = np.random.uniform(-10, 10, n_voxels)
        z = np.random.uniform(0, 20, n_voxels)
        
        # Create model data with multiple density types
        self.model_data = {
            'Model1': {
                'lad': pd.DataFrame({
                    'x': x, 'y': y, 'z': z,
                    'lad': np.random.exponential(0.5, n_voxels),
                    'model_name': 'Model1',
                    'model_type': 'type1',
                    'display_name': 'Model1 LAD'
                }),
                'wad': pd.DataFrame({
                    'x': x, 'y': y, 'z': z,
                    'wad': np.random.exponential(0.3, n_voxels),
                    'model_name': 'Model1',
                    'model_type': 'type1',
                    'display_name': 'Model1 WAD'
                })
            },
            'Model2': {
                'lad': pd.DataFrame({
                    'x': x, 'y': y, 'z': z,
                    'lad': np.random.exponential(0.6, n_voxels),
                    'model_name': 'Model2',
                    'model_type': 'type2',
                    'display_name': 'Model2 LAD'
                })
            }
        }
        
        # Create ground truth
        self.ground_truth = pd.DataFrame({
            'Method': ['Litter Fall'],
            'Method Type': ['Ground Truth'],
            'LAI': [2.19],
            'WAI': [0.74],
            'PAI': [2.93]
        })
    
    def test_compare_all_models(self):
        """Test comprehensive model comparison."""
        results = self.comparison.compare_all_models(
            self.model_data,
            self.ground_truth,
            ['lad', 'wad']
        )
        
        # Check main sections
        self.assertIn('by_density_type', results)
        self.assertIn('overall_ranking', results)
        self.assertIn('best_model_by_metric', results)
        self.assertIn('summary', results)
    
    def test_overall_ranking(self):
        """Test overall model ranking calculation."""
        # Create mock density results
        density_results = {
            'lad': {
                'error_metrics': {
                    'Model1': {'rank': 1, 'mae': 0.1, 'rmse': 0.15, 'bias': 0.05},
                    'Model2': {'rank': 2, 'mae': 0.2, 'rmse': 0.25, 'bias': -0.1}
                }
            }
        }
        
        ranking = self.comparison._calculate_overall_ranking(density_results)
        
        # Check ranking structure
        self.assertIn('Model1', ranking)
        self.assertIn('Model2', ranking)
        
        for model, scores in ranking.items():
            self.assertIn('mean_rank', scores)
            self.assertIn('mean_mae', scores)
            self.assertIn('final_rank', scores)
    
    def test_best_model_identification(self):
        """Test identification of best models by metric."""
        density_results = {
            'lad': {
                'error_metrics': {
                    'Model1': {'mae': 0.1, 'rmse': 0.15, 'bias': 0.05},
                    'Model2': {'mae': 0.2, 'rmse': 0.12, 'bias': -0.02}
                }
            }
        }
        
        best_models = self.comparison._identify_best_models(density_results)
        
        # Check that best models are identified
        self.assertIn('lowest_mae', best_models)
        self.assertIn('lowest_rmse', best_models)
        self.assertIn('lowest_bias', best_models)
        
        # Model1 should have lowest MAE
        if best_models['lowest_mae']['model']:
            self.assertEqual(best_models['lowest_mae']['model'], 'Model1')


class TestStatisticalVisualizer(unittest.TestCase):
    """Test cases for the StatisticalVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = StatisticalVisualizer()
        
        # Create sample comparison results
        self.comparison_results = {
            'by_density_type': {
                'lad': {
                    'reference_value': 2.19,
                    'model_indices': {
                        'Model1': 2.1,
                        'Model2': 2.3,
                        'Model3': 1.9
                    },
                    'error_metrics': {
                        'Model1': {'mae': 0.09, 'rmse': 0.09, 'bias': -0.09, 
                                  'percent_error': -4.1, 'rank': 1},
                        'Model2': {'mae': 0.11, 'rmse': 0.11, 'bias': 0.11, 
                                  'percent_error': 5.0, 'rank': 2},
                        'Model3': {'mae': 0.29, 'rmse': 0.29, 'bias': -0.29, 
                                  'percent_error': -13.2, 'rank': 3}
                    },
                    'statistical_tests': {
                        'one_sample_ttest': {'statistic': -1.5, 'p_value': 0.15, 
                                            'significant': False}
                    }
                }
            },
            'overall_ranking': {
                'Model1': {'mean_rank': 1.0, 'mean_mae': 0.09, 'final_rank': 1},
                'Model2': {'mean_rank': 2.0, 'mean_mae': 0.11, 'final_rank': 2},
                'Model3': {'mean_rank': 3.0, 'mean_mae': 0.29, 'final_rank': 3}
            },
            'summary': {
                'n_models': 3,
                'n_density_types': 1,
                'best_overall_model': 'Model1',
                'key_findings': ['Model1 ranked best overall']
            }
        }
    
    def test_visualization_initialization(self):
        """Test that visualizer initializes correctly."""
        self.assertIsNotNone(self.visualizer.config)
        self.assertIn('visualization', self.visualizer.config)
    
    def test_dashboard_creation(self):
        """Test creation of comprehensive dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Create dashboard
            fig = self.visualizer.create_comprehensive_dashboard(
                self.comparison_results,
                output_path=output_path
            )
            
            # Check that figure was created
            self.assertIsNotNone(fig)
            
            # Check that files were saved
            saved_files = list(output_path.glob('*.png'))
            self.assertGreater(len(saved_files), 0, "Dashboard should be saved")
    
    def test_model_comparison_plot(self):
        """Test creation of model comparison plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            
            # Create comparison plot
            fig = self.visualizer.create_model_comparison_plot(
                self.comparison_results,
                'lad',
                output_path=output_path
            )
            
            # Check that figure was created
            self.assertIsNotNone(fig)
            
            # Check that files were saved
            saved_files = list(output_path.glob('*.png'))
            self.assertGreater(len(saved_files), 0, "Comparison plot should be saved")


def run_integration_test():
    """Run a simple integration test of the statistical analysis framework."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TEST")
    print("="*60)
    
    try:
        # Initialize components
        print("\n1. Initializing statistical analyzer...")
        analyzer = StatisticalAnalyzer()
        print("   ✓ StatisticalAnalyzer initialized")
        
        print("\n2. Initializing model comparison...")
        comparison = ModelComparison(analyzer)
        print("   ✓ ModelComparison initialized")
        
        print("\n3. Initializing visualizer...")
        visualizer = StatisticalVisualizer()
        print("   ✓ StatisticalVisualizer initialized")
        
        # Create sample data
        print("\n4. Creating sample data...")
        np.random.seed(42)
        n_voxels = 100
        
        model_data = {
            'TestModel': pd.DataFrame({
                'x': np.random.uniform(-10, 10, n_voxels),
                'y': np.random.uniform(-10, 10, n_voxels),
                'z': np.random.uniform(0, 20, n_voxels),
                'lad': np.random.exponential(0.5, n_voxels),
                'model_name': 'TestModel',
                'model_type': 'test',
                'display_name': 'Test Model LAD'
            })
        }
        
        ground_truth = pd.DataFrame({
            'Method': ['Test Method'],
            'Method Type': ['Ground Truth'],
            'LAI': [2.0]
        })
        print("   ✓ Sample data created")
        
        # Run analysis
        print("\n5. Running statistical comparison...")
        results = analyzer.compare_to_ground_truth(model_data, ground_truth, 'lad')
        print("   ✓ Statistical comparison completed")
        
        # Check results
        print("\n6. Verifying results...")
        assert 'reference_value' in results
        assert 'model_indices' in results
        assert 'error_metrics' in results
        print("   ✓ Results structure verified")
        
        print("\n" + "="*60)
        print("INTEGRATION TEST PASSED")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        return False


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalVisualizer))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run integration test
    if result.wasSuccessful():
        integration_success = run_integration_test()
        
        if integration_success:
            print("\n✓ All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n✗ Integration test failed")
            sys.exit(1)
    else:
        print("\n✗ Unit tests failed")
        sys.exit(1)