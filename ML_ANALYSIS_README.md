# VoxPlot Machine Learning Analysis Framework

## Overview

The VoxPlot ML Analysis Framework extends the core VoxPlot functionality with advanced machine learning and AI techniques for comprehensive spatial pattern analysis of 3D forest structure data. This framework provides deep insights into the physical accuracy, spatial patterns, and comparative performance of different voxel-based forest modeling approaches.

## Key Features

### 🧠 Advanced ML Algorithms
- **Clustering Analysis**: K-Means, DBSCAN, and Hierarchical clustering
- **Dimensionality Reduction**: PCA, t-SNE, and UMAP
- **Regression Analysis**: Linear, Random Forest, and XGBoost
- **Classification**: Random Forest and SVM for pattern classification

### 🌳 Forest-Specific Analysis
- **Spatial Pattern Detection**: Automated identification of density hotspots and coldspots
- **Height-Stratified Analysis**: Crown layer-specific pattern analysis (upper, middle, lower)
- **Clumping Analysis**: Detection and quantification of leaf/wood clumping patterns
- **Occlusion Assessment**: Shadow detection and correction evaluation

### 📊 Comparative Model Analysis
- **Physical Accuracy Assessment**: Plausibility checks and consistency metrics
- **Cross-Model Comparison**: Similarity analysis and performance ranking
- **Ground Truth Validation**: Comparison against field measurements (when available)
- **Uncertainty Quantification**: Bootstrap confidence intervals and error propagation

### 📈 Publication-Quality Visualizations
- **3D Interactive Plots**: Cluster visualizations and spatial distributions
- **Comprehensive Dashboards**: Multi-panel analysis summaries
- **Statistical Plots**: Correlation matrices, regression diagnostics, and distribution analyses
- **Export Formats**: PNG, PDF, SVG for publications

## Installation

### Core Requirements
```bash
pip install -r requirements.txt
```

### Optional Advanced Features
For enhanced functionality, install optional packages:

```bash
# UMAP for advanced dimensionality reduction
pip install umap-learn

# XGBoost for gradient boosting regression  
pip install xgboost

# Interactive visualizations
pip install plotly bokeh

# Large dataset handling
pip install dask h5py
```

## Quick Start

### Basic Usage

```bash
# Run complete ML analysis with configuration file
python ml_main.py --config config_ml.yaml --verbose

# Analyze specific models
python ml_main.py --config config_ml.yaml --models AmapVox_TLS VoxLAD_TLS

# Run specific analysis types
python ml_main.py --config config_ml.yaml --analysis clustering spatial_patterns
```

### Programmatic Usage

```python
from ml_main import MLEnhancedVoxPlotAnalysis

# Initialize analysis system
ml_analysis = MLEnhancedVoxPlotAnalysis("config_ml.yaml", verbose=True)

# Run complete analysis
results = ml_analysis.run_complete_ml_analysis()

# Access specific results
clustering_results = results['clustering_analysis']
spatial_patterns = results['spatial_patterns']
model_rankings = results['comparative_analysis']['performance_ranking']
```

## Analysis Methods

### 1. Clustering Analysis

Identifies spatial patterns and groupings in 3D forest structure:

**Algorithms:**
- **K-Means**: Partitioning clusters based on density similarity
- **DBSCAN**: Density-based clustering for irregular shapes
- **Hierarchical**: Tree-based clustering with dendrograms

**Outputs:**
- Optimal cluster numbers and silhouette scores
- 3D cluster visualizations
- Cluster characteristics (size, density, height distribution)
- Spatial distribution of cluster centroids

### 2. Dimensionality Reduction

Reduces high-dimensional voxel feature space for visualization and analysis:

**Methods:**
- **PCA**: Principal Component Analysis for linear dimensionality reduction
- **t-SNE**: Non-linear embedding for local structure preservation
- **UMAP**: Uniform Manifold Approximation for global and local structure

**Applications:**
- Feature space visualization
- Noise reduction and data compression
- Pattern discovery in high-dimensional data
- Model comparison in reduced feature space

### 3. Spatial Pattern Analysis

Comprehensive analysis of spatial distributions and patterns:

**Pattern Detection:**
- **Density Hotspots**: High and low density region identification
- **Clumping Analysis**: Spatial aggregation patterns
- **Height Stratification**: Vertical distribution patterns
- **Spatial Gradients**: Density change patterns

**Metrics:**
- Spatial autocorrelation coefficients
- Clumping indices and aggregation measures
- Height-layer density distributions
- Spatial continuity assessments

### 4. Physical Accuracy Assessment

Evaluates the physical plausibility and consistency of model outputs:

**Plausibility Checks:**
- Negative density detection
- Extreme value identification
- Distribution normality tests
- Outlier analysis using IQR and Z-score methods

**Consistency Metrics:**
- Spatial consistency across neighboring voxels
- Height consistency across crown layers
- Temporal consistency (if available)
- Cross-model consistency analysis

### 5. Occlusion Analysis

Assesses occlusion effects and correction quality:

**Shadow Detection:**
- Identification of potential shadow regions
- Density drop pattern analysis
- Line-of-sight calculations (optional)

**Correction Assessment:**
- Empirical evaluation of occlusion correction
- Model-specific correction quality metrics
- Comparison of correction effectiveness

### 6. Comparative Analysis

Comprehensive comparison between different models:

**Similarity Analysis:**
- Feature-based model similarity matrices
- Cosine similarity and correlation measures
- Cluster agreement analysis

**Performance Ranking:**
- Multi-criteria model ranking
- Weighted scoring across different metrics
- Best model identification for specific use cases

## Configuration

### Main Configuration File: `config_ml.yaml`

The configuration file controls all aspects of the ML analysis:

```yaml
# Analysis parameters
analysis:
  crown_base_height: 5.0
  voxel_size: 0.25
  min_density: 0.001
  density_types: ["lad", "wad", "pad"]
  output_dir: "ml_analysis_results"

# ML-specific configuration
ml_config:
  clustering:
    kmeans:
      cluster_range: [3, 5, 8, 12, 15]
    dbscan:
      eps_values: [0.5, 1.0, 2.0, 3.0]
      min_samples: [5, 10, 15, 20]

  spatial_analysis:
    hotspot_detection:
      high_density_percentile: 90
      low_density_percentile: 10
    clumping:
      neighborhood_radius: 1.0
      clumping_threshold: 0.1

# Model definitions
models:
  AmapVox_TLS:
    model_type: "amapvox"
    file_paths:
      pad: "data/amapvox_tls_pad.csv"
      wad: "data/amapvox_tls_wad.csv"
```

### Key Configuration Sections

- **`analysis`**: Basic analysis parameters and data handling
- **`ml_config`**: ML algorithm parameters and thresholds
- **`visualization`**: Plot styling and export settings
- **`models`**: Model definitions and file paths
- **`performance`**: Memory and computational optimization settings

## Output Structure

The ML analysis generates a comprehensive set of outputs:

```
ml_analysis_results/
├── ml_visualizations/
│   ├── ml_comprehensive_dashboard_YYYYMMDD_HHMMSS.png
│   ├── 3d_clustering_amapvox_tls_YYYYMMDD_HHMMSS.png
│   └── ...
├── exported_ml_data/
│   ├── clustering_summary.csv
│   ├── spatial_patterns_summary.csv
│   ├── model_performance_rankings.csv
│   ├── model_similarity_matrix.csv
│   └── ml_analysis_complete_results.json
└── ml_analysis_report.md
```

### File Types

- **Visualizations**: High-resolution plots (PNG, PDF, SVG)
- **Data Exports**: Structured data (CSV, Excel, JSON)
- **Reports**: Comprehensive analysis summaries (Markdown, HTML)
- **Raw Results**: Complete analysis results (JSON, HDF5)

## Advanced Usage

### Custom Analysis Pipeline

```python
from ml_analyzer import SpatialPatternAnalyzer
from ml_visualizer import MLVisualizer

# Custom configuration
config = {
    'clustering': {'kmeans_clusters': [3, 5, 8]},
    'spatial_analysis': {'clumping_threshold': 0.05}
}

# Initialize components
analyzer = SpatialPatternAnalyzer(config)
visualizer = MLVisualizer(config)

# Load your data
data_dict = {"model1": df1, "model2": df2}

# Run analysis
results = analyzer.analyze_spatial_patterns(data_dict)

# Create visualizations
fig = visualizer.create_comprehensive_ml_dashboard(results, data_dict)
```

### Batch Processing

```python
# Process multiple configurations
configs = ["config1.yaml", "config2.yaml", "config3.yaml"]

for config_file in configs:
    ml_analysis = MLEnhancedVoxPlotAnalysis(config_file)
    results = ml_analysis.run_complete_ml_analysis()
    # Process results...
```

### Integration with Existing VoxPlot

```python
# Combine with standard VoxPlot analysis
from main import EnhancedVoxPlotAnalysis
from ml_main import MLEnhancedVoxPlotAnalysis

# Run standard analysis
voxplot = EnhancedVoxPlotAnalysis("config.yaml")
standard_results = voxplot.run_complete_analysis()

# Run ML analysis on same data
ml_voxplot = MLEnhancedVoxPlotAnalysis("config_ml.yaml")
ml_results = ml_voxplot.run_complete_ml_analysis()

# Combine results for comprehensive reporting
combined_analysis = combine_results(standard_results, ml_results)
```

## Research Applications

### Forest Ecology Studies
- **Canopy Structure Analysis**: 3D distribution patterns of leaves and wood
- **Species Differentiation**: Clustering analysis for species-specific patterns
- **Temporal Changes**: Time-series analysis of forest structure evolution

### Remote Sensing Validation
- **LiDAR Validation**: Comparison between different LiDAR processing methods
- **Sensor Comparison**: TLS vs ULS vs photogrammetry performance assessment  
- **Algorithm Development**: Benchmarking of new voxelization algorithms

### Model Development
- **Algorithm Optimization**: Parameter tuning through ML analysis
- **Uncertainty Quantification**: Error propagation and confidence intervals
- **Physical Constraint Validation**: Ensuring biological plausibility

## Performance Optimization

### Memory Management
- **Chunked Processing**: Large datasets processed in memory-efficient chunks
- **Selective Analysis**: Option to run specific analysis types only
- **Data Sampling**: Random sampling for exploratory analysis

### Computational Efficiency
- **Parallel Processing**: Multi-core utilization for CPU-intensive tasks
- **GPU Acceleration**: Optional GPU support for large-scale clustering
- **Caching**: Intermediate result caching for iterative analysis

### Large Dataset Handling
- **Dask Integration**: Distributed computing for very large datasets  
- **HDF5 Support**: Efficient storage and access of large arrays
- **Progressive Analysis**: Incremental analysis with progress reporting

## API Reference

### Core Classes

#### `MLEnhancedVoxPlotAnalysis`
Main analysis coordinator class.

```python
ml_analysis = MLEnhancedVoxPlotAnalysis(
    config_path: str,
    verbose: bool = False
)

results = ml_analysis.run_complete_ml_analysis(
    models: Optional[List[str]] = None,
    analysis_types: Optional[List[str]] = None
)
```

#### `SpatialPatternAnalyzer`
Core ML analysis engine.

```python
analyzer = SpatialPatternAnalyzer(config: Optional[Dict] = None)
results = analyzer.analyze_spatial_patterns(data_dict: Dict[str, pd.DataFrame])
```

#### `MLVisualizer`
Visualization and plotting engine.

```python
visualizer = MLVisualizer(config: Optional[Dict] = None)
fig = visualizer.create_comprehensive_ml_dashboard(
    results: Dict,
    data_dict: Optional[Dict] = None,
    output_path: Optional[Path] = None
)
```

### Key Methods

- **`analyze_spatial_patterns()`**: Main analysis method
- **`create_comprehensive_ml_dashboard()`**: Generate summary dashboard
- **`create_3d_clustering_visualization()`**: 3D cluster plots
- **`export_results()`**: Data export in multiple formats

## Troubleshooting

### Common Issues

1. **Memory Errors with Large Datasets**
   ```python
   # Solution: Enable sampling
   config['ml_analysis']['enable_sampling'] = True
   config['ml_analysis']['max_sample_size'] = 5000
   ```

2. **Slow t-SNE Performance**
   ```python
   # Solution: Reduce perplexity or disable for large datasets
   config['ml_config']['dimensionality_reduction']['tsne']['enable'] = False
   ```

3. **Missing Optional Dependencies**
   ```bash
   # Install optional packages as needed
   pip install umap-learn xgboost plotly
   ```

4. **Configuration File Errors**
   ```python
   # Validate configuration
   from config_manager import ConfigManager
   config_manager = ConfigManager("config_ml.yaml")
   config = config_manager.validate_config()
   ```

### Performance Tips

1. **For Large Datasets (>100k voxels)**:
   - Enable data sampling
   - Use parallel processing (`n_jobs=-1`)
   - Consider chunked processing

2. **For Many Models (>5)**:
   - Run selective analysis first
   - Use caching for repeated analysis
   - Consider batch processing

3. **For Publication Figures**:
   - Set high DPI (600+)
   - Export in vector formats (SVG, PDF)
   - Use custom color schemes

## Examples

See `example_ml_analysis.py` for complete working examples:

```bash
# Run all examples
python example_ml_analysis.py all

# Run specific example
python example_ml_analysis.py programmatic
```

## Contributing

Contributions to the ML framework are welcome! Key areas for development:

- **New ML Algorithms**: Additional clustering, classification, or regression methods
- **Specialized Analyses**: Forest-specific analysis methods
- **Visualization Enhancements**: Interactive plots and advanced visualizations
- **Performance Optimization**: GPU acceleration and distributed computing
- **Documentation**: Examples, tutorials, and use cases

## Citation

When using VoxPlot's ML Analysis Framework in your research, please cite:

```
VoxPlot ML Framework: Machine Learning Enhanced Analysis of 3D Forest Structure Data
[Your Citation Details]
```

## Support

For questions, issues, or feature requests:

1. Check the documentation and examples
2. Search existing issues on the repository
3. Create a new issue with detailed information
4. Include configuration files and error logs
5. Specify system information and package versions

---

*VoxPlot ML Analysis Framework - Advancing 3D Forest Structure Analysis through Machine Learning*