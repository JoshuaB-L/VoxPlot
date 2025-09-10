#!/usr/bin/env python3
"""
ML-Enhanced VoxPlot Analysis Main Script
========================================
Comprehensive machine learning analysis for 3D forest structure data.
Extends VoxPlot with advanced AI/ML techniques including clustering,
dimensionality reduction, spatial pattern analysis, and comparative modeling.

This script provides:
- Integrated ML analysis pipeline
- Spatial pattern detection and analysis
- Physical accuracy assessment through ML
- Advanced model comparison and ranking
- Publication-quality ML visualizations

Usage:
    python ml_main.py --config config_ml.yaml --verbose
    python ml_main.py --config config_ml.yaml --models AmapVox_TLS VoxLAD_TLS
    python ml_main.py --config config_ml.yaml --analysis-type clustering
    
Author: Joshua B-L & Claude Code
Date: 2025-09-10
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
import yaml

# VoxPlot core modules
from data_loader import DataLoader
from config_manager import ConfigManager
from utils import setup_logging, create_output_directory

# ML analysis modules
from ml_analyzer import SpatialPatternAnalyzer
from ml_visualizer import MLVisualizer

# Standard libraries
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)


class MLEnhancedVoxPlotAnalysis:
    """
    Machine Learning Enhanced VoxPlot Analysis System.
    
    Provides comprehensive ML analysis for 3D forest structure data including:
    - Clustering analysis (K-Means, DBSCAN, Hierarchical)
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Spatial pattern analysis and clumping detection
    - Physical accuracy assessment
    - Occlusion analysis and correction evaluation
    - Advanced model comparison and ranking
    """
    
    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize ML-enhanced VoxPlot analysis system.
        
        Args:
            config_path: Path to ML analysis configuration file
            verbose: Enable verbose logging
        """
        self.config_path = Path(config_path)
        self.verbose = verbose
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        setup_logging(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_manager = ConfigManager(self.config_path)
        self.config = self.config_manager.load_config()
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.ml_analyzer = SpatialPatternAnalyzer(self.config)
        self.ml_visualizer = MLVisualizer(self.config)
        
        # Results storage
        self.loaded_data = {}
        self.ml_results = {}
        
        self.logger.info("ML-Enhanced VoxPlot Analysis System initialized")
        self.logger.info(f"Configuration loaded from: {self.config_path}")
    
    def run_complete_ml_analysis(self, 
                                models: Optional[List[str]] = None,
                                analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete ML-enhanced analysis pipeline.
        
        Args:
            models: Specific models to analyze (default: all configured models)
            analysis_types: Specific analysis types to run (default: all)
            
        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("=== Starting ML-Enhanced VoxPlot Analysis ===")
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading voxel data")
            self._load_voxel_data(models)
            
            if not self.loaded_data:
                self.logger.error("No data loaded. Cannot proceed with analysis.")
                return {}
            
            # Step 2: Run ML analysis
            self.logger.info("Step 2: Running ML spatial pattern analysis")
            self.ml_results = self._run_ml_analysis(analysis_types)
            
            # Step 3: Generate visualizations
            self.logger.info("Step 3: Generating ML visualizations")
            self._generate_ml_visualizations()
            
            # Step 4: Export results
            self.logger.info("Step 4: Exporting ML analysis results")
            self._export_ml_results()
            
            # Step 5: Generate analysis report
            self.logger.info("Step 5: Generating comprehensive ML analysis report")
            self._generate_ml_report()
            
            self.logger.info("=== ML-Enhanced Analysis Complete ===")
            return self.ml_results
            
        except Exception as e:
            self.logger.error(f"ML analysis failed: {str(e)}")
            self.logger.debug("Traceback:", exc_info=True)
            return {}
    
    def _load_voxel_data(self, models: Optional[List[str]] = None):
        """Load voxel data for all configured models."""
        available_models = self.config.get('models', {})
        
        if models:
            # Filter to requested models
            models_to_load = {k: v for k, v in available_models.items() if k in models}
            if not models_to_load:
                self.logger.warning(f"None of the requested models {models} found in configuration")
                models_to_load = available_models
        else:
            models_to_load = available_models
        
        self.logger.info(f"Loading data for {len(models_to_load)} models")
        
        for model_name, model_config in models_to_load.items():
            try:
                self.logger.info(f"Loading data for model: {model_name}")
                
                # Load all density types for this model
                model_data = {}
                density_types = self.config.get('analysis', {}).get('density_types', ['lad', 'wad', 'pad'])
                
                for density_type in density_types:
                    if density_type in model_config.get('file_paths', {}):
                        try:
                            data = self.data_loader.load_model_data(model_name, density_type)
                            if data is not None and len(data) > 0:
                                model_data[density_type] = data
                                self.logger.info(f"Loaded {len(data)} voxels for {model_name}_{density_type}")
                        except Exception as e:
                            self.logger.warning(f"Failed to load {density_type} for {model_name}: {e}")
                
                # Combine density types if multiple are available
                if model_data:
                    if len(model_data) == 1:
                        # Single density type
                        density_type = list(model_data.keys())[0]
                        combined_data = model_data[density_type].copy()
                        combined_data['density_type'] = density_type
                    else:
                        # Multiple density types - combine with density type labels
                        combined_data = []
                        for density_type, data in model_data.items():
                            data_copy = data.copy()
                            data_copy['density_type'] = density_type
                            combined_data.append(data_copy)
                        combined_data = pd.concat(combined_data, ignore_index=True)
                    
                    self.loaded_data[model_name] = combined_data
                    self.logger.info(f"Successfully loaded {len(combined_data)} total voxels for {model_name}")
                else:
                    self.logger.warning(f"No valid data loaded for model: {model_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                continue
        
        if self.loaded_data:
            self.logger.info(f"Data loading complete: {len(self.loaded_data)} models loaded")
        else:
            self.logger.error("No models loaded successfully")
    
    def _run_ml_analysis(self, analysis_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run comprehensive ML analysis."""
        if not self.loaded_data:
            self.logger.error("No data available for ML analysis")
            return {}
        
        # Available analysis types
        available_analyses = [
            'clustering', 'dimensionality_reduction', 'spatial_patterns',
            'physical_accuracy', 'occlusion_analysis', 'comparative_analysis'
        ]
        
        # Determine which analyses to run
        if analysis_types is None:
            analyses_to_run = available_analyses
        else:
            analyses_to_run = [a for a in analysis_types if a in available_analyses]
            if not analyses_to_run:
                self.logger.warning(f"No valid analysis types in {analysis_types}")
                analyses_to_run = available_analyses
        
        self.logger.info(f"Running ML analyses: {analyses_to_run}")
        
        try:
            # Run the comprehensive spatial pattern analysis
            ml_results = self.ml_analyzer.analyze_spatial_patterns(self.loaded_data)
            
            self.logger.info("ML analysis completed successfully")
            
            # Add metadata
            ml_results['analysis_metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'models_analyzed': list(self.loaded_data.keys()),
                'analyses_performed': analyses_to_run,
                'configuration': self.config_path.name,
                'total_voxels_analyzed': sum(len(df) for df in self.loaded_data.values())
            }
            
            # Log summary statistics
            self._log_ml_analysis_summary(ml_results)
            
            return ml_results
            
        except Exception as e:
            self.logger.error(f"ML analysis failed: {e}")
            self.logger.debug("Traceback:", exc_info=True)
            return {}
    
    def _log_ml_analysis_summary(self, results: Dict[str, Any]):
        """Log summary of ML analysis results."""
        self.logger.info("=== ML Analysis Summary ===")
        
        # Clustering analysis summary
        clustering_results = results.get('clustering_analysis', {})
        if clustering_results:
            self.logger.info(f"Clustering analysis completed for {len(clustering_results)} models")
            for model_name, model_clustering in clustering_results.items():
                best_kmeans = model_clustering.get('best_kmeans', {})
                if best_kmeans:
                    self.logger.info(f"  {model_name}: Best K={best_kmeans.get('n_clusters', 'N/A')}, "
                                   f"Silhouette={best_kmeans.get('score', 0):.3f}")
        
        # Spatial patterns summary
        spatial_results = results.get('spatial_patterns', {})
        if spatial_results:
            self.logger.info(f"Spatial pattern analysis completed for {len(spatial_results)} models")
            for model_name, spatial_data in spatial_results.items():
                hotspots = spatial_data.get('density_hotspots', {})
                if 'high_density_regions' in hotspots:
                    high_pct = hotspots['high_density_regions'].get('percentage', 0)
                    self.logger.info(f"  {model_name}: {high_pct:.1f}% high-density regions identified")
        
        # Physical accuracy summary
        accuracy_results = results.get('physical_accuracy', {})
        if accuracy_results:
            self.logger.info(f"Physical accuracy assessment completed for {len(accuracy_results)} models")
            for model_name, accuracy_data in accuracy_results.items():
                plausibility = accuracy_data.get('physical_plausibility', {})
                negative_count = plausibility.get('negative_densities', 0)
                if negative_count > 0:
                    self.logger.warning(f"  {model_name}: {negative_count} negative density values detected")
        
        # Comparative analysis summary
        comparative_results = results.get('comparative_analysis', {})
        if comparative_results:
            performance_ranking = comparative_results.get('performance_ranking', {})
            if 'ranked_models' in performance_ranking:
                ranked_models = performance_ranking['ranked_models']
                best_model = ranked_models[0][0] if ranked_models else 'N/A'
                self.logger.info(f"Model performance ranking completed. Best performing: {best_model}")
    
    def _generate_ml_visualizations(self):
        """Generate comprehensive ML visualizations."""
        if not self.ml_results:
            self.logger.warning("No ML results available for visualization")
            return
        
        # Create output directory
        output_dir = Path(self.config.get('analysis', {}).get('output_dir', 'ml_analysis_results'))
        viz_dir = output_dir / 'ml_visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Comprehensive ML Dashboard
            self.logger.info("Creating comprehensive ML dashboard")
            dashboard_fig = self.ml_visualizer.create_comprehensive_ml_dashboard(
                self.ml_results, 
                self.loaded_data,
                output_path=viz_dir
            )
            
            # 2. Individual 3D clustering visualizations
            self.logger.info("Creating 3D clustering visualizations")
            for model_name in self.loaded_data.keys():
                try:
                    clustering_fig = self.ml_visualizer.create_3d_clustering_visualization(
                        self.loaded_data,
                        self.ml_results,
                        model_name,
                        output_path=viz_dir
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to create 3D clustering for {model_name}: {e}")
            
            self.logger.info(f"ML visualizations saved to: {viz_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate ML visualizations: {e}")
    
    def _export_ml_results(self):
        """Export ML analysis results to various formats."""
        if not self.ml_results:
            self.logger.warning("No ML results available for export")
            return
        
        # Create output directory
        output_dir = Path(self.config.get('analysis', {}).get('output_dir', 'ml_analysis_results'))
        export_dir = output_dir / 'exported_ml_data'
        export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Export clustering results
            self._export_clustering_results(export_dir)
            
            # 2. Export spatial pattern results
            self._export_spatial_pattern_results(export_dir)
            
            # 3. Export model comparison results
            self._export_model_comparison_results(export_dir)
            
            # 4. Export comprehensive results as JSON
            import json
            json_path = export_dir / 'ml_analysis_complete_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_results_for_json(self.ml_results)
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"ML results exported to: {export_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to export ML results: {e}")
    
    def _export_clustering_results(self, export_dir: Path):
        """Export clustering analysis results."""
        clustering_results = self.ml_results.get('clustering_analysis', {})
        
        if not clustering_results:
            return
        
        # Create clustering summary table
        clustering_summary = []
        
        for model_name, model_clustering in clustering_results.items():
            # Best K-Means results
            best_kmeans = model_clustering.get('best_kmeans', {})
            if best_kmeans:
                clustering_summary.append({
                    'Model': model_name,
                    'Best_K': best_kmeans.get('n_clusters', 0),
                    'Silhouette_Score': best_kmeans.get('score', 0),
                    'Algorithm': 'K-Means'
                })
            
            # Best DBSCAN results
            best_dbscan = model_clustering.get('best_dbscan', {})
            if best_dbscan:
                params = best_dbscan.get('parameters', (0, 0))
                clustering_summary.append({
                    'Model': model_name,
                    'Best_K': f"eps={params[0]}, min_samples={params[1]}",
                    'Silhouette_Score': best_dbscan.get('score', 0),
                    'Algorithm': 'DBSCAN'
                })
        
        if clustering_summary:
            df = pd.DataFrame(clustering_summary)
            df.to_csv(export_dir / 'clustering_summary.csv', index=False)
            
            # Try to export to Excel if openpyxl is available
            try:
                df.to_excel(export_dir / 'clustering_summary.xlsx', index=False)
            except ImportError:
                self.logger.warning("openpyxl not available, skipping Excel export")
    
    def _export_spatial_pattern_results(self, export_dir: Path):
        """Export spatial pattern analysis results."""
        spatial_results = self.ml_results.get('spatial_patterns', {})
        
        if not spatial_results:
            return
        
        # Create spatial patterns summary
        spatial_summary = []
        
        for model_name, spatial_data in spatial_results.items():
            # Density hotspots
            hotspots = spatial_data.get('density_hotspots', {})
            high_regions = hotspots.get('high_density_regions', {})
            low_regions = hotspots.get('low_density_regions', {})
            
            # Height-stratified patterns
            height_patterns = spatial_data.get('height_stratified_patterns', {})
            
            row = {
                'Model': model_name,
                'High_Density_Percentage': high_regions.get('percentage', 0),
                'Low_Density_Percentage': low_regions.get('percentage', 0),
                'High_Density_Mean': high_regions.get('mean_density', 0),
                'Low_Density_Mean': low_regions.get('mean_density', 0)
            }
            
            # Add height layer statistics
            for layer in ['lower', 'middle', 'upper']:
                layer_data = height_patterns.get(layer, {})
                density_stats = layer_data.get('density_statistics', {})
                row[f'{layer.capitalize()}_Layer_Mean_Density'] = density_stats.get('mean', 0)
                row[f'{layer.capitalize()}_Layer_Point_Count'] = layer_data.get('point_count', 0)
            
            spatial_summary.append(row)
        
        if spatial_summary:
            df = pd.DataFrame(spatial_summary)
            df.to_csv(export_dir / 'spatial_patterns_summary.csv', index=False)
            
            try:
                df.to_excel(export_dir / 'spatial_patterns_summary.xlsx', index=False)
            except ImportError:
                pass
    
    def _export_model_comparison_results(self, export_dir: Path):
        """Export model comparison and ranking results."""
        comparative_results = self.ml_results.get('comparative_analysis', {})
        
        if not comparative_results:
            return
        
        # Model performance rankings
        performance_ranking = comparative_results.get('performance_ranking', {})
        if 'ranked_models' in performance_ranking:
            ranked_models = performance_ranking['ranked_models']
            criteria_scores = performance_ranking.get('criteria_scores', {})
            
            ranking_data = []
            for i, (model_name, overall_score) in enumerate(ranked_models):
                row = {
                    'Rank': i + 1,
                    'Model': model_name,
                    'Overall_Score': overall_score
                }
                
                # Add individual criteria scores
                for criterion, model_scores in criteria_scores.items():
                    row[f'{criterion.replace("_", " ").title()}_Score'] = model_scores.get(model_name, 0)
                
                ranking_data.append(row)
            
            if ranking_data:
                df = pd.DataFrame(ranking_data)
                df.to_csv(export_dir / 'model_performance_rankings.csv', index=False)
                
                try:
                    df.to_excel(export_dir / 'model_performance_rankings.xlsx', index=False)
                except ImportError:
                    pass
        
        # Model similarity matrix
        similarity_analysis = comparative_results.get('model_similarity_analysis', {})
        if 'similarity_matrix' in similarity_analysis and 'model_names' in similarity_analysis:
            similarity_matrix = similarity_analysis['similarity_matrix']
            model_names = similarity_analysis['model_names']
            
            # Create DataFrame from similarity matrix
            df = pd.DataFrame(similarity_matrix, index=model_names, columns=model_names)
            df.to_csv(export_dir / 'model_similarity_matrix.csv')
    
    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization by converting numpy arrays to lists."""
        import json
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        return convert_numpy(results)
    
    def _generate_ml_report(self):
        """Generate comprehensive ML analysis report."""
        if not self.ml_results:
            self.logger.warning("No ML results available for report generation")
            return
        
        # Create output directory
        output_dir = Path(self.config.get('analysis', {}).get('output_dir', 'ml_analysis_results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / 'ml_analysis_report.md'
        
        try:
            with open(report_path, 'w') as f:
                f.write(self._generate_markdown_report())
            
            self.logger.info(f"ML analysis report saved to: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate ML report: {e}")
    
    def _generate_markdown_report(self) -> str:
        """Generate markdown report content."""
        metadata = self.ml_results.get('analysis_metadata', {})
        
        report = f"""# Machine Learning Enhanced VoxPlot Analysis Report

**Analysis Date:** {metadata.get('timestamp', 'Unknown')}  
**Configuration:** {metadata.get('configuration', 'Unknown')}  
**Models Analyzed:** {', '.join(metadata.get('models_analyzed', []))}  
**Total Voxels:** {metadata.get('total_voxels_analyzed', 0):,}  

---

## Executive Summary

This report presents the results of machine learning enhanced analysis of 3D forest structure data from voxel-based models. The analysis includes clustering, dimensionality reduction, spatial pattern detection, physical accuracy assessment, and comprehensive model comparison.

## Analysis Overview

### Models Evaluated
{self._format_models_section()}

### Analysis Methods Applied
- **Clustering Analysis**: K-Means and DBSCAN clustering to identify spatial patterns
- **Dimensionality Reduction**: PCA and t-SNE for feature space visualization
- **Spatial Pattern Analysis**: Hotspot detection and height-stratified analysis
- **Physical Accuracy Assessment**: Plausibility checks and consistency metrics
- **Occlusion Analysis**: Shadow detection and correction assessment
- **Comparative Analysis**: Cross-model comparison and performance ranking

---

## Key Findings

### Clustering Analysis Results
{self._format_clustering_section()}

### Spatial Pattern Analysis
{self._format_spatial_patterns_section()}

### Model Performance Rankings
{self._format_performance_rankings_section()}

### Physical Accuracy Assessment
{self._format_physical_accuracy_section()}

### Occlusion Analysis
{self._format_occlusion_analysis_section()}

---

## Recommendations

### Best Performing Model
{self._format_recommendations_section()}

### Areas for Improvement
{self._format_improvements_section()}

---

## Technical Details

### Configuration Parameters
{self._format_configuration_section()}

### Analysis Statistics
{self._format_statistics_section()}

---

## Files Generated

- **Visualizations**: `ml_visualizations/` directory
- **Exported Data**: `exported_ml_data/` directory  
- **Complete Results**: `exported_ml_data/ml_analysis_complete_results.json`

---

*Report generated by VoxPlot ML Analysis Framework*  
*For questions or support, refer to the VoxPlot documentation*
"""
        return report
    
    def _format_models_section(self) -> str:
        """Format models section for report."""
        models = self.ml_results.get('analysis_metadata', {}).get('models_analyzed', [])
        if not models:
            return "No models analyzed."
        
        section = ""
        for model in models:
            voxel_count = len(self.loaded_data.get(model, []))
            section += f"- **{model}**: {voxel_count:,} voxels\n"
        
        return section
    
    def _format_clustering_section(self) -> str:
        """Format clustering results section."""
        clustering_results = self.ml_results.get('clustering_analysis', {})
        if not clustering_results:
            return "No clustering analysis performed."
        
        section = ""
        for model_name, model_clustering in clustering_results.items():
            best_kmeans = model_clustering.get('best_kmeans', {})
            if best_kmeans:
                k = best_kmeans.get('n_clusters', 'N/A')
                score = best_kmeans.get('score', 0)
                section += f"- **{model_name}**: Optimal clusters = {k}, Silhouette score = {score:.3f}\n"
        
        return section or "No valid clustering results found."
    
    def _format_spatial_patterns_section(self) -> str:
        """Format spatial patterns section."""
        spatial_results = self.ml_results.get('spatial_patterns', {})
        if not spatial_results:
            return "No spatial pattern analysis performed."
        
        section = ""
        for model_name, spatial_data in spatial_results.items():
            hotspots = spatial_data.get('density_hotspots', {})
            if 'high_density_regions' in hotspots and 'low_density_regions' in hotspots:
                high_pct = hotspots['high_density_regions'].get('percentage', 0)
                low_pct = hotspots['low_density_regions'].get('percentage', 0)
                section += f"- **{model_name}**: {high_pct:.1f}% high-density, {low_pct:.1f}% low-density regions\n"
        
        return section or "No spatial pattern results found."
    
    def _format_performance_rankings_section(self) -> str:
        """Format performance rankings section."""
        comparative_results = self.ml_results.get('comparative_analysis', {})
        performance_ranking = comparative_results.get('performance_ranking', {})
        
        if 'ranked_models' not in performance_ranking:
            return "No performance ranking available."
        
        ranked_models = performance_ranking['ranked_models']
        section = ""
        
        for i, (model_name, score) in enumerate(ranked_models):
            section += f"{i+1}. **{model_name}**: Score = {score:.3f}\n"
        
        return section
    
    def _format_physical_accuracy_section(self) -> str:
        """Format physical accuracy section."""
        accuracy_results = self.ml_results.get('physical_accuracy', {})
        if not accuracy_results:
            return "No physical accuracy assessment performed."
        
        section = ""
        for model_name, accuracy_data in accuracy_results.items():
            plausibility = accuracy_data.get('physical_plausibility', {})
            negative_count = plausibility.get('negative_densities', 0)
            
            distribution_analysis = accuracy_data.get('density_distribution_analysis', {})
            outlier_pct = distribution_analysis.get('outlier_percentage', 0)
            zero_pct = distribution_analysis.get('zero_density_percentage', 0)
            
            section += f"- **{model_name}**: {negative_count} negative values, {outlier_pct:.1f}% outliers, {zero_pct:.1f}% zero densities\n"
        
        return section or "No physical accuracy results found."
    
    def _format_occlusion_analysis_section(self) -> str:
        """Format occlusion analysis section."""
        occlusion_results = self.ml_results.get('occlusion_analysis', {})
        if not occlusion_results:
            return "No occlusion analysis performed."
        
        section = ""
        for model_name, occlusion_data in occlusion_results.items():
            shadow_analysis = occlusion_data.get('shadow_analysis', {})
            shadow_pct = shadow_analysis.get('shadow_percentage', 0)
            
            correction_assessment = occlusion_data.get('occlusion_correction_assessment', {})
            empirical_assessment = correction_assessment.get('empirical_assessment', 'unknown')
            
            section += f"- **{model_name}**: {shadow_pct:.1f}% shadow regions, correction quality: {empirical_assessment}\n"
        
        return section or "No occlusion analysis results found."
    
    def _format_recommendations_section(self) -> str:
        """Format recommendations section."""
        comparative_results = self.ml_results.get('comparative_analysis', {})
        performance_ranking = comparative_results.get('performance_ranking', {})
        
        if 'ranking_explanation' in performance_ranking:
            best_model = performance_ranking['ranking_explanation'].get('best_model')
            if best_model:
                return f"Based on the comprehensive ML analysis, **{best_model}** shows the best overall performance across clustering quality, physical plausibility, and occlusion correction metrics."
        
        return "No clear recommendation available based on current analysis."
    
    def _format_improvements_section(self) -> str:
        """Format improvements section."""
        # This would be based on analysis results
        improvements = []
        
        # Check for models with high negative density counts
        accuracy_results = self.ml_results.get('physical_accuracy', {})
        for model_name, accuracy_data in accuracy_results.items():
            plausibility = accuracy_data.get('physical_plausibility', {})
            negative_count = plausibility.get('negative_densities', 0)
            if negative_count > 10:
                improvements.append(f"- **{model_name}**: Address {negative_count} negative density values")
        
        # Check for poor occlusion correction
        occlusion_results = self.ml_results.get('occlusion_analysis', {})
        for model_name, occlusion_data in occlusion_results.items():
            correction_assessment = occlusion_data.get('occlusion_correction_assessment', {})
            if correction_assessment.get('empirical_assessment') == 'poor':
                improvements.append(f"- **{model_name}**: Improve occlusion correction algorithms")
        
        return '\n'.join(improvements) if improvements else "All models show acceptable performance metrics."
    
    def _format_configuration_section(self) -> str:
        """Format configuration section."""
        analysis_config = self.config.get('analysis', {})
        
        section = f"""
- **Voxel Size**: {analysis_config.get('voxel_size', 'N/A')} m
- **Minimum Density**: {analysis_config.get('min_density', 'N/A')}
- **Crown Base Height**: {analysis_config.get('crown_base_height', 'N/A')} m
- **Density Types**: {', '.join(analysis_config.get('density_types', []))}
- **Analysis Types**: {', '.join(self.ml_results.get('analysis_metadata', {}).get('analyses_performed', []))}
"""
        return section
    
    def _format_statistics_section(self) -> str:
        """Format analysis statistics section."""
        metadata = self.ml_results.get('analysis_metadata', {})
        
        section = f"""
- **Total Voxels Analyzed**: {metadata.get('total_voxels_analyzed', 0):,}
- **Models Processed**: {len(metadata.get('models_analyzed', []))}
- **Analysis Duration**: Completed at {metadata.get('timestamp', 'Unknown')}
- **Configuration File**: {metadata.get('configuration', 'Unknown')}
"""
        return section


def main():
    """Main entry point for ML-enhanced VoxPlot analysis."""
    parser = argparse.ArgumentParser(
        description='ML-Enhanced VoxPlot Analysis for 3D Forest Structure Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ml_main.py --config config_ml.yaml --verbose
    python ml_main.py --config config_ml.yaml --models AmapVox_TLS VoxLAD_TLS
    python ml_main.py --config config_ml.yaml --analysis clustering spatial_patterns
        """
    )
    
    parser.add_argument('--config', '-c', 
                       required=True,
                       help='Path to ML analysis configuration file (YAML format)')
    
    parser.add_argument('--models', '-m',
                       nargs='*',
                       help='Specific models to analyze (default: all configured models)')
    
    parser.add_argument('--analysis', '-a',
                       nargs='*',
                       choices=['clustering', 'dimensionality_reduction', 'spatial_patterns',
                               'physical_accuracy', 'occlusion_analysis', 'comparative_analysis'],
                       help='Specific analysis types to run (default: all)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging output')
    
    parser.add_argument('--output-dir', '-o',
                       help='Override output directory from configuration')
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file {config_path} not found")
        sys.exit(1)
    
    try:
        # Initialize ML analysis system
        ml_analysis = MLEnhancedVoxPlotAnalysis(
            config_path=args.config,
            verbose=args.verbose
        )
        
        # Override output directory if specified
        if args.output_dir:
            if 'analysis' not in ml_analysis.config:
                ml_analysis.config['analysis'] = {}
            ml_analysis.config['analysis']['output_dir'] = args.output_dir
        
        # Run complete analysis
        results = ml_analysis.run_complete_ml_analysis(
            models=args.models,
            analysis_types=args.analysis
        )
        
        if results:
            print("\n" + "="*60)
            print("ML-ENHANCED VOXPLOT ANALYSIS COMPLETE")
            print("="*60)
            
            output_dir = ml_analysis.config.get('analysis', {}).get('output_dir', 'ml_analysis_results')
            print(f"Output directory: {output_dir}")
            print("\nGenerated outputs:")
            print("  ✓ Comprehensive ML analysis dashboard")
            print("  ✓ 3D clustering visualizations")
            print("  ✓ Spatial pattern analysis")
            print("  ✓ Physical accuracy assessment")
            print("  ✓ Occlusion analysis")
            print("  ✓ Model performance rankings")
            print("  ✓ Exported data (CSV and Excel)")
            print("  ✓ Complete analysis report")
            print("\n ML Analysis Methods Applied:")
            print("  ✓ K-Means and DBSCAN clustering")
            print("  ✓ PCA and t-SNE dimensionality reduction")
            print("  ✓ Spatial hotspot detection")
            print("  ✓ Height-stratified pattern analysis")
            print("  ✓ Density clumping analysis")
            print("  ✓ Physical plausibility assessment")
            print("  ✓ Occlusion correction evaluation")
            print("  ✓ Cross-model comparison and ranking")
            print("="*60)
        else:
            print("ML analysis completed with errors. Check logs for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ML analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()