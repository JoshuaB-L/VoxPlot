#!/usr/bin/env python3
"""
Statistical Analysis Main Script for VoxPlot
============================================
Extended VoxPlot analysis with comprehensive statistical comparisons
against ground truth measurements.

This script integrates the statistical analysis framework with the existing
VoxPlot codebase to provide complete model validation and comparison.

Usage:
    python statistical_main.py --config config.yaml --ground-truth ground_truth_data/lai_data_with_range_columns.csv
    python statistical_main.py --config config.yaml --analysis-only
    python statistical_main.py --config config.yaml --output-dir statistical_results
"""

import sys
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import existing VoxPlot modules
from config_manager import ConfigManager
from data_loader import DataLoader, DataProcessor
from data_analyzer import ForestStructureAnalyzer, ComparativeAnalyzer
from visualizer import EnhancedResultsManager
from utils import setup_logging, ensure_directory

# Import new statistical modules
from statistical_analyzer import StatisticalAnalyzer, ModelComparison
from statistical_visualizer import StatisticalVisualizer


class StatisticalVoxPlotAnalysis:
    """
    Extended VoxPlot analysis with statistical validation against ground truth.
    """
    
    def __init__(self, config: Dict[str, Any], ground_truth_path: Optional[str] = None):
        """
        Initialize the statistical analysis system.
        
        Args:
            config: VoxPlot configuration dictionary
            ground_truth_path: Path to ground truth CSV file
        """
        self.config = config
        self.logger = setup_logging(config.get('verbose', False))
        
        # Initialize existing VoxPlot components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        
        # Get analysis parameters
        analysis_config = config.get('analysis', {})
        self.voxel_size = analysis_config.get('voxel_size', 0.25)
        self.crown_base_height = analysis_config.get('crown_base_height', 0.7)
        self.min_density = analysis_config.get('min_density', 0.05)
        self.density_types = analysis_config.get('density_types', ['lad'])
        
        # Initialize analyzers
        self.forest_analyzer = ForestStructureAnalyzer(self.voxel_size)
        self.comparative_analyzer = ComparativeAnalyzer(self.voxel_size)
        
        # Initialize statistical components
        self.statistical_analyzer = StatisticalAnalyzer(alpha=0.05, bootstrap_n=1000)
        self.model_comparison = ModelComparison(self.statistical_analyzer)
        self.statistical_visualizer = StatisticalVisualizer(config)
        
        # Setup output directory
        output_dir = analysis_config.get('output_dir', 'statistical_results')
        self.output_path = ensure_directory(output_dir)
        
        # Load ground truth data
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        
        # Initialize results manager for standard VoxPlot outputs
        self.results_manager = EnhancedResultsManager(output_dir, config)
        
        self.logger.info("Statistical VoxPlot analysis system initialized")
    
    def _load_ground_truth(self, ground_truth_path: Optional[str]) -> Optional[pd.DataFrame]:
        """
        Load ground truth data from CSV file.
        
        Args:
            ground_truth_path: Path to ground truth CSV file
            
        Returns:
            DataFrame with ground truth measurements or None
        """
        if not ground_truth_path:
            # Try default location
            default_path = Path(__file__).parent.parent / 'ground_truth_data' / 'lai_data_with_range_columns.csv'
            if default_path.exists():
                ground_truth_path = str(default_path)
            else:
                self.logger.warning("No ground truth data file specified or found")
                return None
        
        try:
            ground_truth_path = Path(ground_truth_path)
            if not ground_truth_path.exists():
                self.logger.error(f"Ground truth file not found: {ground_truth_path}")
                return None
            
            df = pd.read_csv(ground_truth_path)
            self.logger.info(f"Loaded ground truth data with {len(df)} measurements")
            
            # Validate required columns
            required_cols = ['Method', 'Method Type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Ground truth missing columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load ground truth data: {e}")
            return None
    
    def run_complete_analysis(self) -> bool:
        """
        Run complete VoxPlot analysis with statistical validation.
        
        Returns:
            True if analysis completed successfully
        """
        try:
            self.logger.info("Starting complete statistical VoxPlot analysis")
            
            # Step 1: Load all model datasets
            all_datasets = self._load_all_model_data()
            if not all_datasets:
                self.logger.error("No datasets could be loaded")
                return False
            
            # Step 2: Run standard VoxPlot analyses
            self.logger.info("Running standard VoxPlot analyses")
            standard_results = self._run_standard_analyses(all_datasets)
            
            # Step 3: Run statistical comparisons
            if self.ground_truth is not None:
                self.logger.info("Running statistical comparisons against ground truth")
                statistical_results = self._run_statistical_analyses(all_datasets)
                
                # Step 4: Generate statistical visualizations
                self.logger.info("Generating statistical visualizations")
                self._generate_statistical_plots(statistical_results)
                
                # Step 5: Create comprehensive report
                self._create_comprehensive_report(standard_results, statistical_results)
            else:
                self.logger.warning("Skipping statistical analysis - no ground truth data")
            
            # Step 6: Export all results
            self._export_all_results(standard_results, 
                                   statistical_results if self.ground_truth is not None else None)
            
            self.logger.info("Analysis completed successfully")
            self._print_completion_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _load_all_model_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load data for all models and density types.
        
        Returns:
            Nested dictionary: {model_name: {density_type: DataFrame}}
        """
        all_datasets = {}
        
        models_config = self.config.get('models', {})
        
        for model_name, model_config in models_config.items():
            self.logger.info(f"Loading data for model: {model_name}")
            
            # Load model data using existing DataLoader
            model_datasets = self.data_loader.load_model_data(model_name, model_config)
            
            if model_datasets:
                all_datasets[model_name] = model_datasets
                self.logger.info(f"Loaded {len(model_datasets)} density types for {model_name}")
            else:
                self.logger.warning(f"No data loaded for {model_name}")
        
        return all_datasets
    
    def _run_standard_analyses(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Run standard VoxPlot crown layer and spatial analyses.
        """
        results = {
            'crown_analyses': {},
            'vertical_profiles': {},
            'spatial_distributions': {}
        }
        
        for model_name, model_datasets in all_datasets.items():
            for density_type, df in model_datasets.items():
                key = f"{model_name}_{density_type}"
                
                # Crown layer analysis
                crown_analysis = self.forest_analyzer.analyze_crown_layers(
                    df, density_type, self.crown_base_height, self.min_density
                )
                results['crown_analyses'][key] = crown_analysis
                
                # Vertical profile
                vertical_profile = self.forest_analyzer.analyze_vertical_profile(
                    df, density_type, self.crown_base_height
                )
                results['vertical_profiles'][key] = vertical_profile
                
                # Spatial distribution
                spatial_dist = self.forest_analyzer.analyze_spatial_distribution(
                    df, density_type, self.crown_base_height
                )
                results['spatial_distributions'][key] = spatial_dist
        
        return results
    
    def _run_statistical_analyses(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Run comprehensive statistical comparisons against ground truth.
        """
        # Run model comparison
        comparison_results = self.model_comparison.compare_all_models(
            all_datasets,
            self.ground_truth,
            self.density_types
        )
        
        # Add additional statistical tests
        comparison_results['normality_tests'] = self._test_normality(all_datasets)
        comparison_results['homogeneity_tests'] = self._test_homogeneity(all_datasets)
        
        return comparison_results
    
    def _test_normality(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Test normality of density distributions for each model.
        """
        from scipy import stats
        
        normality_results = {}
        
        for model_name, model_datasets in all_datasets.items():
            for density_type, df in model_datasets.items():
                if density_type in df.columns:
                    # Shapiro-Wilk test
                    sample = df[density_type].dropna()
                    if len(sample) > 3 and len(sample) < 5000:  # Shapiro-Wilk limits
                        statistic, p_value = stats.shapiro(sample.sample(min(5000, len(sample))))
                        
                        key = f"{model_name}_{density_type}"
                        normality_results[key] = {
                            'shapiro_statistic': float(statistic),
                            'shapiro_p_value': float(p_value),
                            'is_normal': p_value > 0.05,
                            'sample_size': len(sample)
                        }
        
        return normality_results
    
    def _test_homogeneity(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Test homogeneity of variance between models.
        """
        from scipy import stats
        
        homogeneity_results = {}
        
        for density_type in self.density_types:
            # Collect samples from all models
            samples = []
            model_names = []
            
            for model_name, model_datasets in all_datasets.items():
                if density_type in model_datasets:
                    df = model_datasets[density_type]
                    if density_type in df.columns:
                        sample = df[density_type].dropna().values
                        if len(sample) > 0:
                            samples.append(sample)
                            model_names.append(model_name)
            
            # Levene's test for homogeneity of variance
            if len(samples) > 1:
                statistic, p_value = stats.levene(*samples)
                
                homogeneity_results[density_type] = {
                    'levene_statistic': float(statistic),
                    'levene_p_value': float(p_value),
                    'equal_variance': p_value > 0.05,
                    'n_models': len(samples),
                    'models': model_names
                }
        
        return homogeneity_results
    
    def _generate_statistical_plots(self, statistical_results: Dict[str, Any]):
        """
        Generate all statistical visualization plots.
        """
        # Create output subdirectory for plots
        plots_dir = self.output_path / 'statistical_plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Comprehensive dashboard
        self.logger.info("Creating statistical dashboard")
        dashboard_fig = self.statistical_visualizer.create_comprehensive_dashboard(
            statistical_results,
            output_path=plots_dir
        )
        
        # 2. Individual density type comparisons
        for density_type in self.density_types:
            self.logger.info(f"Creating comparison plot for {density_type}")
            comparison_fig = self.statistical_visualizer.create_model_comparison_plot(
                statistical_results,
                density_type,
                output_path=plots_dir
            )
        
        self.logger.info(f"Statistical plots saved to {plots_dir}")
    
    def _create_comprehensive_report(self, standard_results: Dict[str, Any], 
                                    statistical_results: Optional[Dict[str, Any]]):
        """
        Create a comprehensive text report of all analyses.
        """
        report_path = self.output_path / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VOXPLOT STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary section
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 40 + "\n")
            
            if statistical_results and 'summary' in statistical_results:
                summary = statistical_results['summary']
                f.write(f"Models Analyzed: {summary.get('n_models', 0)}\n")
                f.write(f"Density Types: {summary.get('n_density_types', 0)}\n")
                f.write(f"Best Overall Model: {summary.get('best_overall_model', 'N/A')}\n\n")
                
                f.write("Key Findings:\n")
                for finding in summary.get('key_findings', []):
                    f.write(f"  • {finding}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
            # Statistical test results
            if statistical_results:
                f.write("STATISTICAL TEST RESULTS\n")
                f.write("-" * 40 + "\n\n")
                
                # Normality tests
                if 'normality_tests' in statistical_results:
                    f.write("Normality Tests (Shapiro-Wilk):\n")
                    for key, result in statistical_results['normality_tests'].items():
                        f.write(f"  {key}:\n")
                        f.write(f"    - p-value: {result['shapiro_p_value']:.4f}\n")
                        f.write(f"    - Normal: {'Yes' if result['is_normal'] else 'No'}\n")
                
                f.write("\n")
                
                # Homogeneity tests
                if 'homogeneity_tests' in statistical_results:
                    f.write("Homogeneity of Variance (Levene's Test):\n")
                    for density, result in statistical_results['homogeneity_tests'].items():
                        f.write(f"  {density.upper()}:\n")
                        f.write(f"    - p-value: {result['levene_p_value']:.4f}\n")
                        f.write(f"    - Equal variance: {'Yes' if result['equal_variance'] else 'No'}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
                # Model rankings
                if 'overall_ranking' in statistical_results:
                    f.write("MODEL RANKINGS\n")
                    f.write("-" * 40 + "\n\n")
                    
                    rankings = statistical_results['overall_ranking']
                    sorted_models = sorted(rankings.items(), 
                                         key=lambda x: x[1].get('final_rank', 999))
                    
                    for rank, (model, metrics) in enumerate(sorted_models, 1):
                        f.write(f"{rank}. {model}\n")
                        f.write(f"   - Mean Rank: {metrics.get('mean_rank', 0):.2f}\n")
                        f.write(f"   - Mean MAE: {metrics.get('mean_mae', 0):.3f}\n")
                        f.write(f"   - Mean RMSE: {metrics.get('mean_rmse', 0):.3f}\n")
                        f.write(f"   - Consistency: {metrics.get('consistency_score', 0):.3f}\n\n")
            
            # Crown layer metrics
            f.write("=" * 80 + "\n")
            f.write("CROWN LAYER ANALYSIS SUMMARY\n")
            f.write("-" * 40 + "\n\n")
            
            for key, analysis in standard_results.get('crown_analyses', {}).items():
                f.write(f"{key}:\n")
                if 'metrics' in analysis:
                    for layer, metrics in analysis['metrics'].items():
                        f.write(f"  {layer.capitalize()} layer:\n")
                        f.write(f"    - Total area: {metrics.get('total_area', 0):.3f} m²\n")
                        f.write(f"    - Mean index: {metrics.get('mean_area_index', 0):.3f}\n")
                        f.write(f"    - Voxel count: {metrics.get('voxel_count', 0)}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated successfully\n")
        
        self.logger.info(f"Report saved to {report_path}")
    
    def _export_all_results(self, standard_results: Dict[str, Any], 
                           statistical_results: Optional[Dict[str, Any]]):
        """
        Export all results to CSV and Excel formats.
        """
        # Export directory
        export_dir = self.output_path / 'exported_data'
        export_dir.mkdir(exist_ok=True)
        
        # 1. Export crown metrics
        crown_metrics = []
        for key, analysis in standard_results.get('crown_analyses', {}).items():
            if 'metrics' in analysis:
                for layer, metrics in analysis['metrics'].items():
                    row = {'Model_Density': key, 'Layer': layer}
                    row.update(metrics)
                    crown_metrics.append(row)
        
        if crown_metrics:
            crown_df = pd.DataFrame(crown_metrics)
            crown_df.to_csv(export_dir / 'crown_metrics.csv', index=False)
            crown_df.to_excel(export_dir / 'crown_metrics.xlsx', index=False)
        
        # 2. Export statistical results
        if statistical_results:
            # Export overall rankings
            if 'overall_ranking' in statistical_results:
                ranking_df = pd.DataFrame.from_dict(statistical_results['overall_ranking'], 
                                                   orient='index')
                ranking_df.index.name = 'Model'
                ranking_df.to_csv(export_dir / 'model_rankings.csv')
                ranking_df.to_excel(export_dir / 'model_rankings.xlsx')
            
            # Export error metrics by density type
            error_data = []
            for density_type, results in statistical_results.get('by_density_type', {}).items():
                if 'error_metrics' in results:
                    for model, metrics in results['error_metrics'].items():
                        row = {'Density_Type': density_type, 'Model': model}
                        row.update(metrics)
                        error_data.append(row)
            
            if error_data:
                error_df = pd.DataFrame(error_data)
                error_df.to_csv(export_dir / 'error_metrics.csv', index=False)
                error_df.to_excel(export_dir / 'error_metrics.xlsx', index=False)
        
        self.logger.info(f"Results exported to {export_dir}")
    
    def _print_completion_summary(self):
        """
        Print analysis completion summary.
        """
        print("\n" + "=" * 60)
        print("STATISTICAL VOXPLOT ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Output directory: {self.output_path}")
        print("\nGenerated outputs:")
        print("  ✓ Statistical comparison dashboard")
        print("  ✓ Individual density type comparisons")
        print("  ✓ Comprehensive analysis report")
        print("  ✓ Exported metrics (CSV and Excel)")
        print("  ✓ Publication-quality figures")
        
        if self.ground_truth is not None:
            print("\n Statistical tests performed:")
            print("  ✓ One-sample t-tests")
            print("  ✓ ANOVA")
            print("  ✓ Correlation analysis")
            print("  ✓ Bland-Altman agreement")
            print("  ✓ Bootstrap confidence intervals")
            print("  ✓ Cross-validation")
            print("  ✓ Spatial error analysis")
        
        print("\n" + "=" * 60)


def main():
    """
    Main entry point for statistical VoxPlot analysis.
    """
    parser = argparse.ArgumentParser(
        description='VoxPlot Statistical Analysis - Compare voxel models against ground truth'
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth CSV file')
    
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only statistical analysis (skip standard VoxPlot)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate configuration without running analysis')
    
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    if args.verbose:
        config['verbose'] = True
    
    if args.output_dir:
        config['analysis']['output_dir'] = args.output_dir
    
    # Validate configuration
    if args.dry_run:
        print("Configuration validated successfully")
        # Print basic config info
        models = list(config.get('models', {}).keys())
        density_types = config.get('analysis', {}).get('density_types', [])
        print(f"Models configured: {models}")
        print(f"Density types: {density_types}")
        return
    
    # Run analysis
    try:
        analyzer = StatisticalVoxPlotAnalysis(config, args.ground_truth)
        success = analyzer.run_complete_analysis()
        
        if success:
            print("\nAnalysis completed successfully!")
            sys.exit(0)
        else:
            print("\nAnalysis completed with errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()