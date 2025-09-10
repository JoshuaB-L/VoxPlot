#!/usr/bin/env python3
"""
Example Usage of Statistical Analysis Framework
===============================================
This script demonstrates how to use the statistical analysis framework
for comparing voxel-based forest structure models against ground truth.

Run this example to:
1. Load and process voxel data from multiple models
2. Compare against ground truth measurements
3. Generate comprehensive statistical analysis
4. Create publication-quality visualizations
"""

import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import ConfigManager
from statistical_main import StatisticalVoxPlotAnalysis


def run_example_analysis():
    """
    Run example statistical analysis with the provided configuration.
    """
    print("=" * 80)
    print("STATISTICAL VOXPLOT ANALYSIS EXAMPLE")
    print("=" * 80)
    
    try:
        # 1. Load configuration
        print("\n1. Loading configuration...")
        config_manager = ConfigManager()
        
        # Use the statistical configuration file
        config_file = Path(__file__).parent / 'config_statistical.yaml'
        
        if not config_file.exists():
            print(f"   ✗ Configuration file not found: {config_file}")
            print("   Please ensure config_statistical.yaml exists")
            return False
            
        config = config_manager.load_config(str(config_file))
        print("   ✓ Configuration loaded successfully")
        
        # 2. Initialize statistical analysis system
        print("\n2. Initializing statistical analysis system...")
        
        # Ground truth file path
        ground_truth_path = Path(__file__).parent.parent / 'ground_truth_data' / 'lai_data_with_range_columns.csv'
        
        if not ground_truth_path.exists():
            print(f"   ✗ Ground truth file not found: {ground_truth_path}")
            print("   The analysis will run without statistical comparisons")
            ground_truth_path = None
        
        analyzer = StatisticalVoxPlotAnalysis(
            config=config,
            ground_truth_path=str(ground_truth_path) if ground_truth_path else None
        )
        print("   ✓ Statistical analysis system initialized")
        
        # 3. Check data availability
        print("\n3. Checking data file availability...")
        models_config = config.get('models', {})
        available_files = 0
        total_files = 0
        
        for model_name, model_config in models_config.items():
            file_paths = model_config.get('file_paths', {})
            print(f"   {model_name}:")
            
            for density_type, file_path in file_paths.items():
                if density_type in ['use_combined_file', 'combined']:
                    continue
                    
                total_files += 1
                if file_path and Path(file_path).exists():
                    print(f"      ✓ {density_type}: Available")
                    available_files += 1
                else:
                    print(f"      ✗ {density_type}: Not found - {file_path}")
        
        print(f"\n   Data availability: {available_files}/{total_files} files found")
        
        if available_files == 0:
            print("   ✗ No data files available for analysis")
            print("\n   To run the analysis with real data:")
            print("   1. Update file paths in config_statistical.yaml")
            print("   2. Ensure data files exist at specified locations")
            print("   3. Run: python statistical_main.py --config config_statistical.yaml")
            return False
        
        # 4. Run analysis (if data is available)
        if available_files > 0:
            print("\n4. Running complete statistical analysis...")
            print("   This may take several minutes...")
            
            success = analyzer.run_complete_analysis()
            
            if success:
                print("   ✓ Analysis completed successfully!")
                
                # 5. Display results summary
                print("\n5. Analysis Results:")
                output_path = analyzer.output_path
                print(f"   Output directory: {output_path}")
                
                # List generated files
                if output_path.exists():
                    png_files = list(output_path.rglob('*.png'))
                    csv_files = list(output_path.rglob('*.csv'))
                    txt_files = list(output_path.rglob('*.txt'))
                    
                    print(f"   Generated files:")
                    print(f"      - {len(png_files)} visualization files (.png)")
                    print(f"      - {len(csv_files)} data export files (.csv)")  
                    print(f"      - {len(txt_files)} report files (.txt)")
                
                return True
            else:
                print("   ✗ Analysis completed with errors")
                return False
    
    except Exception as e:
        print(f"\n   ✗ Example failed: {e}")
        logging.exception("Example analysis failed")
        return False


def demonstrate_framework_capabilities():
    """
    Demonstrate the capabilities of the statistical framework.
    """
    print("\n" + "=" * 80)
    print("FRAMEWORK CAPABILITIES DEMONSTRATION")
    print("=" * 80)
    
    print("""
The Statistical Analysis Framework provides:

📊 STATISTICAL TESTS:
   • One-sample t-tests against ground truth
   • ANOVA for comparing multiple models
   • Normality tests (Shapiro-Wilk)
   • Homogeneity of variance tests (Levene)
   • Non-parametric alternatives (Wilcoxon)

📈 ERROR METRICS:
   • Mean Absolute Error (MAE)
   • Root Mean Square Error (RMSE)
   • Bias and relative bias
   • Percentage error
   • Model ranking by performance

🔗 AGREEMENT ANALYSIS:
   • Pearson and Spearman correlations
   • Lin's Concordance Correlation Coefficient
   • Bland-Altman agreement plots
   • Pairwise model comparisons

🌍 SPATIAL ANALYSIS:
   • Spatial error distribution
   • Error hotspot identification
   • Height-stratified error analysis  
   • Spatial autocorrelation (Moran's I)

🔄 ROBUSTNESS ASSESSMENT:
   • Bootstrap confidence intervals
   • K-fold cross-validation
   • Outlier detection and handling
   • Sensitivity analysis

📊 VISUALIZATIONS:
   • Comprehensive statistical dashboard
   • Model comparison plots
   • Error distribution analyses
   • Correlation matrices
   • Bootstrap confidence intervals
   • Publication-ready figures (Nature journal standards)

📋 REPORTING:
   • Executive summary
   • Detailed statistical report
   • Model rankings and recommendations
   • Exportable data tables (CSV, Excel)
   • Figure captions and references
    """)


def show_usage_instructions():
    """
    Show detailed usage instructions.
    """
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    
    print("""
COMMAND LINE USAGE:

1. Basic statistical analysis:
   python statistical_main.py --config config_statistical.yaml

2. Specify ground truth file:
   python statistical_main.py --config config.yaml --ground-truth ground_truth.csv

3. Custom output directory:
   python statistical_main.py --config config.yaml --output-dir my_results

4. Verbose output:
   python statistical_main.py --config config.yaml --verbose

5. Validate configuration only:
   python statistical_main.py --config config.yaml --dry-run

CONFIGURATION REQUIREMENTS:

• Update model file paths in config_statistical.yaml
• Ensure data files exist at specified locations
• Ground truth CSV should have columns: Method, Method Type, LAI, WAI, PAI
• Voxel data should have columns: x, y, z, density_value

GROUND TRUTH FORMAT:

The ground truth CSV should contain:
- Method: Name of measurement method
- Method Type: "Ground Truth", "Satellite", "LiDAR Terrestrial", etc.
- LAI, WAI, PAI: Area index values
- Litter Fall method is used as gold standard for LAI

OUTPUTS GENERATED:

• Statistical Dashboard: Comprehensive overview of all comparisons
• Model Comparison Plots: Detailed analysis for each density type
• Error Analysis: MAE, RMSE, bias comparisons
• Correlation Plots: Model agreement visualization
• Bootstrap CI: Confidence interval analysis
• Cross-validation: Model stability assessment
• Summary Reports: Text and tabular summaries
• Data Exports: CSV and Excel files for further analysis
    """)


if __name__ == '__main__':
    # Show framework capabilities
    demonstrate_framework_capabilities()
    
    # Show usage instructions
    show_usage_instructions()
    
    # Run example (if data is available)
    print("\n" + "=" * 80)
    print("RUNNING EXAMPLE ANALYSIS")
    print("=" * 80)
    
    success = run_example_analysis()
    
    if success:
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Review generated visualizations and reports")
        print("2. Update configuration for your specific data")
        print("3. Run full analysis with: python statistical_main.py --config config_statistical.yaml")
    else:
        print("\n📝 Example completed (data files not available)")
        print("\nTo run with real data:")
        print("1. Update file paths in config_statistical.yaml")
        print("2. Run: python statistical_main.py --config config_statistical.yaml")
    
    print("\n" + "=" * 80)