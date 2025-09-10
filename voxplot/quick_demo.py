#!/usr/bin/env python3
"""
Quick Demo of Statistical Analysis Framework
============================================
This demonstrates the statistical framework with a subset of your actual data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from statistical_analyzer import StatisticalAnalyzer, ModelComparison
from statistical_visualizer import StatisticalVisualizer

def create_demo_data():
    """Create demo data from actual file samples."""
    print("📊 Creating demonstration with actual data samples...")
    
    # Sample some real data (first 1000 rows from each file)
    data_files = {
        'AmapVox_TLS_PAD': '/mnt/g/My Drive/Bath-Uni/PhD/Structure/Paper-4-PALM-PCM-Study/Code/python/voxpy/plots/wad_lad_pad_comp/leaf_on/pad/amapvox_tls_merged_pad_unfiltered_transect.csv',
        'AmapVox_ULS_PAD': '/mnt/g/My Drive/Bath-Uni/PhD/Structure/Paper-4-PALM-PCM-Study/Code/python/voxpy/plots/wad_lad_pad_comp/leaf_on/pad/amapvox_pad_uls_transect.csv',
        'VoxLAD_TLS_LAD': '/mnt/g/My Drive/Bath-Uni/PhD/Structure/Paper-4-PALM-PCM-Study/Code/python/voxpy/plots/wad_lad_pad_comp/leaf_on/lad/voxlad_v3_britz_oak_seg_scan_pos_42_43_53_g_plagio_vox_0_25_r_0_00035_lad_transect.txt'
    }
    
    model_data = {}
    
    # Load AmapVox TLS PAD (sample)
    try:
        df = pd.read_csv(data_files['AmapVox_TLS_PAD']).head(2000)
        model_data['AmapVox_TLS'] = df.rename(columns={'pad': 'lad'})  # Use as LAD for comparison
        model_data['AmapVox_TLS']['model_name'] = 'AmapVox_TLS'
        model_data['AmapVox_TLS']['model_type'] = 'amapvox'
        model_data['AmapVox_TLS']['display_name'] = 'AmapVox TLS'
        print(f"   ✓ AmapVox TLS: {len(model_data['AmapVox_TLS'])} voxels")
    except Exception as e:
        print(f"   ✗ AmapVox TLS failed: {e}")
    
    # Load AmapVox ULS PAD (sample)  
    try:
        df = pd.read_csv(data_files['AmapVox_ULS_PAD']).head(2000)
        model_data['AmapVox_ULS'] = df.rename(columns={'pad': 'lad'})  # Use as LAD for comparison
        model_data['AmapVox_ULS']['model_name'] = 'AmapVox_ULS'
        model_data['AmapVox_ULS']['model_type'] = 'amapvox'
        model_data['AmapVox_ULS']['display_name'] = 'AmapVox ULS'
        print(f"   ✓ AmapVox ULS: {len(model_data['AmapVox_ULS'])} voxels")
    except Exception as e:
        print(f"   ✗ AmapVox ULS failed: {e}")
    
    # Create ground truth data
    ground_truth = pd.DataFrame({
        'Method': ['Litter Fall', 'Solariscope SOL 300', 'HemispheR (app)'],
        'Method Type': ['Ground Truth', 'Ground Truth', 'Ground Truth'], 
        'LAI': [2.19, 1.96, 1.98],
        'PAI': [np.nan, 2.70, 2.50],
        'WAI': [np.nan, 0.74, 0.52]
    })
    
    return model_data, ground_truth

def run_quick_demo():
    """Run a quick demonstration of the statistical framework."""
    print("🚀 QUICK STATISTICAL ANALYSIS DEMO")
    print("=" * 50)
    
    # 1. Create demo data
    model_data, ground_truth = create_demo_data()
    
    if not model_data:
        print("❌ No data available for demo")
        return False
    
    # 2. Initialize statistical analyzer
    print("\n🔬 Initializing statistical analyzer...")
    analyzer = StatisticalAnalyzer(alpha=0.05, bootstrap_n=100)  # Reduced for speed
    print("   ✓ Statistical analyzer ready")
    
    # 3. Run comparison against ground truth
    print("\n📈 Running statistical comparison...")
    try:
        results = analyzer.compare_to_ground_truth(model_data, ground_truth, 'lad')
        print("   ✓ Statistical comparison completed")
        
        # 4. Show key results
        print("\n📋 KEY RESULTS:")
        print("-" * 30)
        
        # Reference value
        ref_val = results['reference_value']
        print(f"Ground Truth LAI: {ref_val:.3f}")
        
        # Model indices
        print("\nModel LAI Estimates:")
        for model, index in results['model_indices'].items():
            print(f"  • {model}: {index:.3f}")
        
        # Error metrics
        print("\nError Metrics:")
        for model, metrics in results['error_metrics'].items():
            print(f"  • {model}:")
            print(f"    - MAE: {metrics['mae']:.3f}")
            print(f"    - RMSE: {metrics['rmse']:.3f}")
            print(f"    - Bias: {metrics['bias']:.3f}")
            print(f"    - % Error: {metrics['percent_error']:.1f}%")
            print(f"    - Rank: {metrics['rank']}")
        
        # Statistical tests
        if 'statistical_tests' in results:
            tests = results['statistical_tests']
            if 'one_sample_ttest' in tests:
                ttest = tests['one_sample_ttest']
                print(f"\nOne-sample t-test:")
                print(f"  • p-value: {ttest['p_value']:.4f}")
                print(f"  • Significant: {'Yes' if ttest['significant'] else 'No'}")
        
        # Model ranking summary
        error_data = [(k, v['mae']) for k, v in results['error_metrics'].items()]
        error_data.sort(key=lambda x: x[1])
        
        print(f"\n🏆 MODEL RANKING (by MAE):")
        for i, (model, mae) in enumerate(error_data, 1):
            print(f"  {i}. {model} (MAE: {mae:.3f})")
        
        print("\n✅ Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_framework_status():
    """Show the status of framework components."""
    print("\n📦 FRAMEWORK STATUS:")
    print("-" * 30)
    
    components = [
        ("Statistical Analyzer", "statistical_analyzer.py"),
        ("Statistical Visualizer", "statistical_visualizer.py"), 
        ("Statistical Main", "statistical_main.py"),
        ("Test Suite", "test_statistical_analysis.py"),
        ("Configuration", "config_statistical.yaml"),
        ("Ground Truth Data", "../ground_truth_data/lai_data_with_range_columns.csv")
    ]
    
    for name, file_path in components:
        if Path(file_path).exists():
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name} - {file_path}")
    
    print("\n🎯 USAGE COMMANDS:")
    print("-" * 30)
    print("python statistical_main.py --config config_statistical.yaml")
    print("python statistical_main.py --config config_statistical.yaml --dry-run") 
    print("python example_statistical_analysis.py")
    print("python test_statistical_analysis.py")

if __name__ == '__main__':
    print("🌟 VoxPlot Statistical Analysis Framework")
    print("=" * 50)
    
    # Show framework status
    show_framework_status()
    
    # Run quick demo
    success = run_quick_demo()
    
    if success:
        print("\n🎉 Framework is working correctly!")
        print("\n📝 Next Steps:")
        print("1. Run full analysis: python statistical_main.py --config config_statistical.yaml")
        print("2. The full analysis will take several minutes with your large datasets")
        print("3. Results will be saved to: statistical_analysis_results/")
    else:
        print("\n🔧 Framework needs attention - check error messages above")