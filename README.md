# VoxPlot

**VoxPlot** is a Python framework for analyzing and visualizing forest structure data from voxel-based models (AmapVox, VoxLAD, VoxPy). It performs crown layer analysis, spatial distribution assessment, and comparative visualizations for Wood Area Density (WAD), Leaf Area Density (LAD), and Plant Area Density (PAD) metrics.

## Features

- **Crown Layer Analysis**: Automated analysis of upper, middle, and lower crown layers based on height percentiles
- **Statistical Comparisons**: Bootstrap confidence intervals, t-tests, ANOVA, and correlation analysis
- **Publication-Quality Visualizations**: Nature journal-standard figures with comprehensive dashboards
- **Multi-Format Support**: Compatible with AmapVox, VoxLAD, and VoxPy data formats
- **Ground Truth Validation**: Compare model outputs against field measurements
- **Spatial Error Analysis**: Height-stratified error assessment and spatial distribution mapping
- **Cross-Validation**: Robust model performance evaluation

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
# Standard VoxPlot analysis
python voxplot/main.py --config voxplot/config.yaml --verbose

# Statistical analysis with ground truth comparison
python voxplot/statistical_main.py --config voxplot/config_statistical.yaml --verbose
```

### Configuration

Analysis parameters are specified in YAML configuration files:

```yaml
analysis:
  crown_base_height: 5.0
  voxel_size: 0.5
  min_density: 0.001
  comparison_mode: "same_density_type_different_model_type"
  density_types: ["lad", "wad", "pad"]
  output_dir: "results"

models:
  AmapVox_TLS:
    model_type: "amapvox"
    file_paths:
      lad: "data/amapvox_lad.csv"
      wad: "data/amapvox_wad.csv"
```

## Key Components

- **`main.py`**: Enhanced VoxPlot analysis coordinator
- **`statistical_main.py`**: Statistical comparison framework
- **`data_loader.py`**: Multi-format data loading and processing
- **`statistical_analyzer.py`**: Comprehensive statistical analysis
- **`statistical_visualizer.py`**: Publication-quality visualization system
- **`config_manager.py`**: YAML configuration management

## Data Formats Supported

- **AmapVox**: CSV with columns `x,y,z,density_value`
- **VoxLAD**: Space-delimited with `x y z lad_1 percent_explored_1 ...`
- **VoxPy**: CSV with `x,y,z,density_type` or combined format

## Output

VoxPlot generates:
- Statistical comparison dashboards
- Individual density type comparisons
- Comprehensive analysis reports
- Exportable metrics (CSV and Excel)
- Publication-ready figures (PNG, PDF, SVG)

## Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, Seaborn
- SciPy for statistical analysis
- PyYAML for configuration management

## License

This project is developed for academic research purposes.

## Citation

When using VoxPlot in your research, please cite:

```
VoxPlot: A Python Framework for Forest Structure Analysis from Voxel-Based Models
[Your Publication Details]
```

## Contributing

Contributions are welcome! Please see the development guidelines in `CLAUDE.md`.