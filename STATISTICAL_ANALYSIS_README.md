# Statistical Analysis Framework for VoxPlot

## Overview

This statistical analysis framework extends the VoxPlot voxel-based forest structure analysis tool with comprehensive statistical validation capabilities. It enables rigorous comparison of 3D model outputs (AmapVox, VoxLAD, VoxPy) against ground truth measurements following established statistical methods for model validation.

## Key Features

### 🧪 Statistical Tests
- **One-sample t-tests** against reference values
- **ANOVA** for multi-model comparisons
- **Normality testing** (Shapiro-Wilk)
- **Homogeneity of variance** (Levene's test)
- **Non-parametric alternatives** (Wilcoxon signed-rank)

### 📊 Error Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Square Error (RMSE)**
- **Bias and relative bias assessment**
- **Percentage error calculations**
- **Automated model ranking**

### 🔗 Agreement Analysis
- **Pearson and Spearman correlations**
- **Lin's Concordance Correlation Coefficient**
- **Bland-Altman agreement plots**
- **Pairwise model comparisons**

### 🌍 Spatial Analysis
- **Spatial error distribution mapping**
- **Error hotspot identification**
- **Height-stratified error analysis**
- **Spatial autocorrelation** (Moran's I)

### 🔄 Robustness Assessment
- **Bootstrap confidence intervals**
- **K-fold cross-validation**
- **Outlier detection and handling**
- **Model stability assessment**

### 📈 Publication-Quality Visualizations
- **Comprehensive statistical dashboards**
- **Model comparison plots**
- **Error distribution analyses**
- **Correlation matrices**
- **Bootstrap confidence intervals**
- **Nature journal formatting standards**

## Quick Start

### 1. Installation Requirements

Ensure you have the required dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### 2. Basic Usage

```bash
# Run statistical analysis with configuration file
python statistical_main.py --config config_statistical.yaml

# Specify ground truth file
python statistical_main.py --config config.yaml --ground-truth ground_truth.csv

# Custom output directory  
python statistical_main.py --config config.yaml --output-dir results

# Verbose output
python statistical_main.py --config config.yaml --verbose
```

### 3. Example Analysis

```bash
# Run the example to see framework capabilities
python example_statistical_analysis.py

# Test the framework components
python test_statistical_analysis.py
```

## Configuration

### Statistical Analysis Settings

The `config_statistical.yaml` file contains enhanced settings for statistical analysis:

```yaml
analysis:
  statistical:
    alpha: 0.05                    # Significance level
    bootstrap_samples: 1000        # Bootstrap iterations
    cv_folds: 5                   # Cross-validation folds
    spatial_tolerance: 0.1         # Spatial matching tolerance (m)
    outlier_detection: true        # Enable outlier detection
    
    reference_preferences:
      LAI: "Litter Fall"          # Gold standard for LAI
      WAI: "Ground Truth Mean"    # Use mean for WAI
      PAI: "Ground Truth Mean"    # Use mean for PAI
```

### Ground Truth Data Format

The ground truth CSV should contain:

| Column | Description | Example |
|--------|-------------|---------|
| Method | Measurement method name | "Litter Fall" |
| Method Type | Type of measurement | "Ground Truth" |
| LAI | Leaf Area Index | 2.19 |
| WAI | Wood Area Index | 0.74 |
| PAI | Plant Area Index | 2.93 |

## Architecture

### Core Components

1. **`statistical_analyzer.py`**
   - `StatisticalAnalyzer`: Core statistical computation engine
   - `ModelComparison`: High-level model comparison framework

2. **`statistical_visualizer.py`**
   - `StatisticalVisualizer`: Publication-quality visualization engine
   - Nature journal formatting compliance

3. **`statistical_main.py`**
   - `StatisticalVoxPlotAnalysis`: Main analysis coordinator
   - Integration with existing VoxPlot components

### Data Flow

1. **Data Loading**: Load voxel data from multiple models and formats
2. **Index Calculation**: Convert 3D density to integrated area indices
3. **Statistical Testing**: Compare against ground truth references
4. **Error Analysis**: Calculate comprehensive error metrics
5. **Spatial Analysis**: Analyze spatial distribution of errors
6. **Visualization**: Generate publication-ready figures
7. **Reporting**: Create comprehensive analysis reports

## Methodology

### Reference Value Selection

- **LAI**: Litter Fall method is used as the gold standard
- **WAI/PAI**: Mean of available ground truth methods
- **Temporal Matching**: 30-day tolerance for time-series data

### Statistical Approach

1. **Integrated Index Calculation**:
   ```
   Area Index = Σ(density × voxel_volume) / ground_area
   ```

2. **Error Metrics**:
   - MAE = |model_value - reference_value|
   - RMSE = √((model_value - reference_value)²)
   - Bias = model_value - reference_value

3. **Significance Testing**:
   - One-sample t-test: H₀: μ_model = μ_reference
   - ANOVA: H₀: μ₁ = μ₂ = μ₃ (for multiple models)

4. **Agreement Assessment**:
   - Pearson correlation for linear relationships
   - CCC for agreement assessment
   - Bland-Altman plots for bias visualization

## Output Structure

```
statistical_analysis_results/
├── statistical_plots/
│   ├── 01_Statistical_Dashboard/
│   ├── 02_Model_Comparisons/
│   ├── 03_Error_Analysis/
│   ├── 04_Correlation_Analysis/
│   └── 05_Spatial_Analysis/
├── exported_data/
│   ├── crown_metrics.csv
│   ├── model_rankings.csv
│   ├── error_metrics.csv
│   └── statistical_test_results.csv
└── analysis_report.txt
```

## Key Outputs

### 1. Statistical Dashboard
- Comprehensive overview of all statistical comparisons
- Error metrics visualization
- Model ranking heatmaps
- Statistical test results

### 2. Model Comparison Plots
- Individual density type analyses
- Bland-Altman agreement plots
- Residual distributions
- Bootstrap confidence intervals

### 3. Analysis Reports
- Executive summary with key findings
- Detailed statistical test results
- Model rankings and recommendations
- Publication-ready summaries

## Best Practices

### 1. Data Preparation
- Ensure consistent coordinate systems across models
- Validate data quality before analysis
- Handle missing values appropriately

### 2. Statistical Interpretation
- Consider multiple metrics, not just statistical significance
- Account for spatial and temporal variability
- Validate assumptions for parametric tests

### 3. Visualization
- Use consistent color schemes across plots
- Include confidence intervals where appropriate
- Follow journal-specific formatting guidelines

## Integration with Existing VoxPlot

The statistical framework seamlessly integrates with the existing VoxPlot codebase:

- **Data Loading**: Uses existing `DataLoader` and `DataProcessor`
- **Analysis**: Extends `ForestStructureAnalyzer` capabilities
- **Visualization**: Complements existing `EnhancedResultsManager`
- **Configuration**: Compatible with existing YAML configuration system

## Validation and Testing

### Unit Tests
```bash
python test_statistical_analysis.py
```

Tests cover:
- Statistical computation accuracy
- Data processing pipelines
- Visualization generation
- Configuration validation

### Integration Tests
- End-to-end analysis workflows
- Data format compatibility
- Output generation verification

## Troubleshooting

### Common Issues

1. **Data Format Errors**
   - Ensure CSV files have required columns
   - Check coordinate system consistency
   - Validate density value ranges

2. **Statistical Warnings**
   - Small sample sizes may affect test validity
   - Non-normal distributions may require non-parametric tests
   - Outliers can significantly impact results

3. **Memory Issues**
   - Large datasets may require chunked processing
   - Reduce visualization complexity if needed
   - Consider subsampling for exploratory analysis

### Performance Optimization

- Use spatial indexing for large datasets
- Implement parallel processing where applicable
- Cache intermediate results for repeated analyses

## Citation

If you use this statistical analysis framework in your research, please cite:

```bibtex
@software{voxplot_statistical_2024,
  title={VoxPlot Statistical Analysis Framework: Comprehensive Validation for Voxel-based Forest Structure Models},
  author={[Your Name]},
  year={2024},
  url={https://github.com/your-repo/voxplot}
}
```

## Contributing

We welcome contributions to improve the statistical analysis framework:

1. **Bug Reports**: Use GitHub issues for bug reports
2. **Feature Requests**: Suggest new statistical methods or visualizations
3. **Code Contributions**: Follow existing code style and include tests
4. **Documentation**: Help improve documentation and examples

## License

This statistical analysis framework is released under the same license as the main VoxPlot project.

## Support

- 📖 **Documentation**: See inline code documentation
- 🐛 **Bug Reports**: GitHub Issues
- 💬 **Discussions**: GitHub Discussions
- 📧 **Email**: [your.email@institution.edu]

## Acknowledgments

- Nature Plant journal for visualization standards
- Scientific Python community for statistical tools
- VoxPlot development team for the foundational framework
- Forest structure modeling community for validation approaches