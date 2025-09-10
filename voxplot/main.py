#!/usr/bin/env python3
"""
VoxPlot: Enhanced Publication-Quality Voxel-based Forest Structure Analysis

This enhanced version provides publication-quality visualizations optimized for 
Nature journal standards with improved typography, color palettes, and export options.

Usage:
    python main.py --config enhanced_config.yaml
    python main.py --config enhanced_config.yaml --plot-mode single_plots
    python main.py --config enhanced_config.yaml --output-dir publication_results
"""

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility

from config_manager import ConfigManager
from data_loader import DataLoader, DataProcessor
from data_analyzer import ForestStructureAnalyzer, ComparativeAnalyzer
from visualizer import EnhancedResultsManager  # Updated import
from utils import setup_logging


class EnhancedVoxPlotAnalysis:
    """Enhanced analysis coordinator for publication-quality VoxPlot system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging(config.get('verbose', False))
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        
        # Get analysis parameters
        analysis_config = config.get('analysis', {})
        self.voxel_size = analysis_config.get('voxel_size', 0.25)
        self.crown_base_height = analysis_config.get('crown_base_height', 0.7)
        self.min_density = analysis_config.get('min_density', 0.05)
        self.comparison_mode = analysis_config.get('comparison_mode', 'same_density_type_different_model_type')
        self.density_types = analysis_config.get('density_types', ['lad'])
        
        # Initialize analyzers
        self.forest_analyzer = ForestStructureAnalyzer(self.voxel_size)
        self.comparative_analyzer = ComparativeAnalyzer(self.voxel_size)
        
        # Initialize enhanced results manager
        output_dir = analysis_config.get('output_dir', 'voxplot_enhanced_results')
        self.results_manager = EnhancedResultsManager(output_dir, config)
        
        # Get plot mode for logging
        viz_config = analysis_config.get('visualization', {})
        plot_mode = viz_config.get('plot_mode', 'combined_plots')
        
        self.logger.info(f"Enhanced VoxPlot analysis system initialized - Mode: {plot_mode}")
        self.logger.info(f"Publication quality settings: DPI={viz_config.get('dpi', 600)}")
    
    def run_enhanced_analysis(self) -> bool:
        """
        Run the enhanced VoxPlot analysis pipeline with publication-quality outputs.
        
        Returns:
            True if analysis completed successfully, False otherwise
        """
        try:
            self.logger.info("Starting enhanced VoxPlot analysis pipeline")
            
            # Step 1: Load all datasets
            all_datasets = self._load_all_datasets()
            if not all_datasets:
                self.logger.error("No datasets could be loaded. Analysis cannot proceed.")
                return False
            
            # Step 2: Create and export dataset summary
            dataset_summary = self.data_processor.create_dataset_summary(all_datasets)
            self._log_dataset_summary(dataset_summary)
            self._export_dataset_summary(dataset_summary)
            
            # Step 3: Run enhanced comparative analyses
            analysis_success = self._run_enhanced_comparative_analyses(all_datasets)
            
            # Step 4: Generate enhanced summary report
            self.results_manager.create_enhanced_summary_report({}, dataset_summary)
            
            # Step 5: Create publication information file
            self._create_publication_info()
            
            if analysis_success:
                self.logger.info("Enhanced VoxPlot analysis completed successfully")
                self._print_enhanced_completion_summary()
                return True
            else:
                self.logger.warning("Enhanced analysis completed with some errors")
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced analysis failed with error: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _load_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Load all datasets with enhanced error handling."""
        self.logger.info("Loading datasets with enhanced validation")
        
        models_config = self.config.get('models', {})
        all_datasets = {}
        
        for model_name, model_config in models_config.items():
            try:
                self.logger.info(f"Processing model: {model_name}")
                datasets = self.data_loader.load_model_data(model_name, model_config)
                
                if datasets:
                    # Add color index information for consistent coloring
                    color_index = model_config.get('display_color_index', len(all_datasets))
                    for density_type, df in datasets.items():
                        if not df.empty:
                            df['color_index'] = color_index
                    
                    all_datasets[model_name] = datasets
                    loaded_types = list(datasets.keys())
                    self.logger.info(f"Successfully loaded {model_name} with density types: {loaded_types}")
                else:
                    self.logger.warning(f"No data loaded for model: {model_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load data for model {model_name}: {e}")
                continue
        
        total_datasets = sum(len(datasets) for datasets in all_datasets.values())
        self.logger.info(f"Enhanced loading complete: {total_datasets} datasets from {len(all_datasets)} models")
        
        return all_datasets
    
    def _run_enhanced_comparative_analyses(self, all_datasets: Dict[str, Dict[str, Any]]) -> bool:
        """Run enhanced comparative analyses with publication-quality outputs."""
        self.logger.info(f"Running enhanced comparative analysis in mode: {self.comparison_mode}")
        
        analysis_success = True
        
        for density_type in self.density_types:
            try:
                self.logger.info(f"Processing enhanced analysis for density type: {density_type.upper()}")
                
                # Get datasets for this density type
                datasets_for_analysis = self._get_datasets_for_density_type(all_datasets, density_type)
                
                if not datasets_for_analysis:
                    self.logger.warning(f"No datasets found for density type: {density_type}")
                    continue
                
                # Organize datasets for comparison
                comparison_groups = self.data_processor.organize_datasets_by_comparison_mode(
                    {model: {density_type: df} for model, df in datasets_for_analysis.items()},
                    self.comparison_mode
                )
                
                if not comparison_groups:
                    self.logger.warning(f"No valid comparison groups for {density_type} in mode {self.comparison_mode}")
                    continue
                
                # Run enhanced analysis for each comparison group
                for group_idx, datasets_group in enumerate(comparison_groups):
                    group_name = f"{density_type}_group_{group_idx + 1}"
                    self.logger.info(f"Processing enhanced comparison group: {group_name}")
                    
                    success = self._analyze_dataset_group_enhanced(datasets_group, density_type, group_name)
                    if not success:
                        analysis_success = False
                        
            except Exception as e:
                self.logger.error(f"Error in enhanced analysis for density type {density_type}: {e}")
                analysis_success = False
                continue
        
        return analysis_success
    
    def _analyze_dataset_group_enhanced(self, datasets: List[Any], density_type: str, group_name: str) -> bool:
        """Analyze dataset group with enhanced publication-quality outputs."""
        try:
            self.logger.info(f"Running enhanced comparative analysis for {len(datasets)} datasets")
            
            # Crown layer analysis with enhanced metrics
            crown_results = self.comparative_analyzer.compare_crown_metrics(
                datasets, density_type, self.crown_base_height, self.min_density
            )
            
            # Vertical profile analysis
            profile_results = self.comparative_analyzer.compare_vertical_profiles(
                datasets, density_type, self.crown_base_height
            )
            
            # Spatial distribution analysis
            distribution_results = self.comparative_analyzer.compare_spatial_distributions(
                datasets, density_type, self.crown_base_height
            )
            
            # Save enhanced results with multiple formats and improved styling
            self.results_manager.save_enhanced_comparison_results(
                crown_results, density_type, group_name
            )
            
            # Save enhanced distribution results
            self.results_manager.save_enhanced_distribution_results(
                datasets, density_type, group_name
            )
            
            # Save enhanced 3D visualization
            self.results_manager.save_enhanced_3d_results(
                datasets, density_type, group_name
            )
            
            # Save vertical profile (using original method but with enhanced styling)
            self.results_manager.save_profile_results(
                profile_results, density_type, group_name
            )
            
            self.logger.info(f"Enhanced analysis completed successfully for {group_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed enhanced analysis for dataset group {group_name}: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _get_datasets_for_density_type(self, all_datasets: Dict[str, Dict[str, Any]], 
                                      density_type: str) -> Dict[str, Any]:
        """Extract datasets for a specific density type with color index preservation."""
        datasets_for_type = {}
        
        for model_name, model_datasets in all_datasets.items():
            if density_type in model_datasets:
                datasets_for_type[model_name] = model_datasets[density_type]
            else:
                self.logger.info(f"Model {model_name} does not have {density_type} data available")
        
        if not datasets_for_type:
            self.logger.warning(f"No models have {density_type} data available")
        else:
            self.logger.info(f"Found {len(datasets_for_type)} models with {density_type} data")
        
        return datasets_for_type
    
    def _export_dataset_summary(self, summary_df):
        """Export dataset summary in multiple formats."""
        try:
            # Export to CSV
            csv_path = self.results_manager.data_dir / "dataset_summary.csv"
            summary_df.to_csv(csv_path, index=False)
            
            # Export to Excel if available
            try:
                xlsx_path = self.results_manager.data_dir / "dataset_summary.xlsx"
                summary_df.to_excel(xlsx_path, index=False, sheet_name='Dataset_Summary')
                self.logger.info(f"Dataset summary exported to Excel: {xlsx_path}")
            except ImportError:
                self.logger.info("Excel export not available (openpyxl not installed)")
            
            self.logger.info(f"Dataset summary exported to: {csv_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export dataset summary: {e}")
    
    def _create_publication_info(self):
        """Create publication information file with technical details."""
        try:
            pub_info_path = self.results_manager.reports_dir / "publication_information.md"
            
            viz_config = self.config.get('analysis', {}).get('visualization', {})
            typography = viz_config.get('typography', {})
            colors = viz_config.get('colors', {})
            
            with open(pub_info_path, 'w') as f:
                f.write("# VoxPlot Publication Information\n\n")
                
                f.write("## Technical Specifications\n\n")
                f.write(f"- **Resolution**: {viz_config.get('dpi', 600)} DPI\n")
                f.write(f"- **Plot Mode**: {viz_config.get('plot_mode', 'combined_plots')}\n")
                f.write(f"- **Primary Font**: {typography.get('title_font', 'Helvetica')}\n")
                f.write(f"- **Color Palette**: {len(colors.get('model_palette', []))} publication-quality colors\n")
                f.write(f"- **Export Formats**: {', '.join(viz_config.get('export', {}).get('formats', ['png']))}\n\n")
                
                f.write("## Figure Quality Standards\n\n")
                f.write("- Typography optimized for Nature journal requirements\n")
                f.write("- Color palette designed for accessibility and print reproduction\n")
                f.write("- Publication-ready resolution and formatting\n")
                f.write("- Consistent styling across all visualizations\n")
                f.write("- Enhanced 3D visualizations with proper depth perception\n\n")
                
                f.write("## Export Information\n\n")
                f.write(f"- **Figures Directory**: {self.results_manager.figures_dir}\n")
                f.write(f"- **Data Tables**: {self.results_manager.tables_dir}\n")
                f.write(f"- **Raw Data**: {self.results_manager.data_dir}\n")
                f.write(f"- **Reports**: {self.results_manager.reports_dir}\n\n")
                
                if viz_config.get('plot_mode') == 'single_plots':
                    f.write("## Single Plot Organization\n\n")
                    subfolder_structure = viz_config.get('export', {}).get('single_plots', {}).get('subfolder_structure', {})
                    for analysis_type, folder_name in subfolder_structure.items():
                        f.write(f"- **{analysis_type.replace('_', ' ').title()}**: {folder_name}/\n")
                
                f.write("\n## Citation Recommendation\n\n")
                f.write("When using these visualizations in publications, please cite:\n")
                f.write("VoxPlot: Advanced Voxel-based Forest Structure Analysis Framework\n")
                f.write("with publication-quality visualization capabilities.\n")
            
            self.logger.info(f"Publication information saved: {pub_info_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create publication information: {e}")
    
    def _log_dataset_summary(self, summary_df):
        """Log enhanced dataset summary information."""
        self.logger.info("Enhanced Dataset Summary:")
        self.logger.info("-" * 60)
        
        for _, row in summary_df.iterrows():
            self.logger.info(f"{row['Model']} ({row['Model_Type']}) - {row['Density_Type']}:")
            self.logger.info(f"  Records: {row['Records']:,}")
            self.logger.info(f"  Spatial extent: X={row['X_Range']}, Y={row['Y_Range']}, Z={row['Z_Range']}")
            self.logger.info(f"  Density range: {row['Density_Range']}")
            self.logger.info(f"  Non-zero voxels: {row['Nonzero_Count']:,}")
    
    def _print_enhanced_completion_summary(self):
        """Print enhanced completion summary to console."""
        viz_config = self.config.get('analysis', {}).get('visualization', {})
        
        print("\n" + "="*80)
        print("ENHANCED VOXPLOT ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Output directory: {self.results_manager.output_dir}")
        print(f"Publication-quality figures: {self.results_manager.figures_dir}")
        print(f"Exportable data tables: {self.results_manager.tables_dir}")
        print(f"Raw data files: {self.results_manager.data_dir}")
        print(f"Analysis reports: {self.results_manager.reports_dir}")
        
        print("\nPublication Quality Features:")
        print(f"- Resolution: {viz_config.get('dpi', 600)} DPI")
        print(f"- Plot mode: {viz_config.get('plot_mode', 'combined_plots')}")
        print(f"- Font: {viz_config.get('typography', {}).get('title_font', 'Helvetica')}")
        print(f"- Export formats: {', '.join(viz_config.get('export', {}).get('formats', ['png']))}")
        
        print("\nAnalysis Configuration:")
        analysis_config = self.config.get('analysis', {})
        print(f"- Comparison mode: {analysis_config.get('comparison_mode', 'N/A')}")
        print(f"- Density types: {', '.join(analysis_config.get('density_types', []))}")
        print(f"- Crown base height: {analysis_config.get('crown_base_height', 'N/A')} m")
        print(f"- Voxel resolution: {analysis_config.get('voxel_size', 'N/A')} m")
        
        if viz_config.get('plot_mode') == 'single_plots':
            print("\nSingle Plot Mode Active:")
            print("- Individual plots organized in themed subfolders")
            print("- Multiple export formats for each visualization")
            print("- Timestamped filenames for version control")
        
        print("="*80 + "\n")


def main():
    """Enhanced main entry point for VoxPlot analysis."""
    try:
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Parse command line arguments
        args = config_manager.parse_arguments()
        
        # Load configuration
        config = config_manager.load_config(args.config)
        
        # Apply command line overrides
        config_manager.override_config(
            output_dir=args.output_dir,
            comparison_mode=args.comparison_mode,
            verbose=args.verbose
        )
        
        # Print enhanced configuration summary
        if args.verbose:
            config_manager.print_config_summary()
            print("\nEnhanced Features:")
            viz_config = config.get('analysis', {}).get('visualization', {})
            print(f"- Plot Mode: {viz_config.get('plot_mode', 'combined_plots')}")
            print(f"- DPI: {viz_config.get('dpi', 600)}")
            print(f"- Font: {viz_config.get('typography', {}).get('title_font', 'Helvetica')}")
        
        # Check for dry run
        if args.dry_run:
            print("Enhanced configuration validation completed successfully.")
            print("Configuration optimized for publication-quality output.")
            return 0
        
        # Initialize and run enhanced analysis
        analysis = EnhancedVoxPlotAnalysis(config)
        success = analysis.run_enhanced_analysis()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nEnhanced analysis interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nFatal error in enhanced analysis: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())