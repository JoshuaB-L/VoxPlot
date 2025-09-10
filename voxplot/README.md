# VoxPlot: Advanced Voxel-based Forest Structure Analysis

VoxPlot is a comprehensive Python framework for analyzing and visualizing forest structure data from voxel-based models including AmapVox, VoxLAD, and VoxPy. It provides sophisticated crown layer analysis, spatial distribution assessment, and comparative visualizations for Wood Area Density (WAD), Leaf Area Density (LAD), and Plant Area Density (PAD) metrics.

## Features

### 🌳 Multi-Model Support
- **AmapVox**: Terrestrial and ULS laser scanning voxel models
- **VoxLAD**: Multi-scan LAD estimation models
- **VoxPy**: Advanced voxel-based forest structure models

### 📊 Comprehensive Analysis
- **Crown Layer Analysis**: Automated segmentation into upper, middle, and lower crown layers
- **Spatial Distribution**: 2D raster mapping and spatial statistics
- **Vertical Profiles**: Height-based density distribution analysis
- **3D Visualization**: Interactive 3D voxel representations

### 🔍 Flexible Comparison Modes
- Same density type across different models
- Different density types within same model
- Comprehensive cross-model and cross-density comparisons
- Statistical comparative analysis

### 📈 Professional Visualizations
- High-quality publication-ready figures
- Crown layer comparison matrices
- Vertical profile comparisons
- Distribution analysis plots
- 3D voxel visualizations

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/your-repo/voxplot.git
cd voxplot

# Install dependencies
pip install -r requirements.txt
```

### Development Install
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -r requirements-dev.txt
```

## Quick Start

### 1. Configure Your Analysis

Create a configuration file (e.g., `my_config.yaml`):

```yaml
analysis:
  crown_base_height: 0.7
  voxel_size: 0.25
  min_density: 0.05
  comparison_mode: "same_density_type_different_model_type"
  density_types: ["lad", "wad", "pad"]
  output_dir: "my_results"

models:
  AmapVox_TLS:
    model_type: "amapvox"
    file_paths:
      lad: "/path/to/amapvox_lad.csv"
      wad: "/path/to/amapvox_wad.csv"
      pad: "/path/to/amapvox_pad.csv"
  
  VoxLAD_TLS:
    model_type: "voxlad"
    file_paths:
      lad: "/path/to/voxlad_lad.txt"
      wad: "/path/to/voxlad_wad.txt"
```

### 2. Run Analysis

```bash
# Basic analysis
python main.py --config my_config.yaml

# With custom output directory
python main.py --config my_config.yaml --output-dir custom_results

# Verbose output
python main.py --config my_config.yaml --verbose

# Validate configuration without running
python main.py --config my_config.yaml --dry-run
```

### 3. View Results

Results are organized in the output directory:
```
my_results/
├── figures/           # All generated plots and visualizations
├── data/             # Processed data and metrics tables
└── reports/          # Summary reports and statistics
```

## Configuration Guide

### Analysis Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `crown_base_height` | Height above ground for crown analysis (m) | 0.7 | Any positive float |
| `voxel_size` | Spatial resolution for analysis (m) | 0.25 | Any positive float |
| `min_density` | Minimum density threshold | 0.05 | Any non-negative float |
| `comparison_mode` | Type of comparison to perform | `"same_density_type_different_model_type"` | See below |
| `density_types` | Which density types to analyze | `["lad"]` | `["wad", "lad", "pad"]` |

### Comparison Modes

1. **`same_density_type_same_model_type`**: Compare multiple instances of the same model and density type
2. **`same_density_type_different_model_type`**: Compare different models for the same density type (most common)
3. **`different_density_type_same_model_type`**: Compare different density types within the same model
4. **`different_density_type_different_model_type`**: Comprehensive comparison of all combinations

### Model Types and File Formats

#### AmapVox Format
CSV files with columns: `x,y,z,density_type`
```csv
x,y,z,lad
-3.097,32.878,-0.849,0.080
```

#### VoxLAD Format  
Space-delimited files with multiple density estimates:
```
x y z lad_1 percent_explored_1 lad_2 percent_explored_2 lad_3 percent_explored_3 best_scan
```

#### VoxPy Format
**Individual files**: CSV with `x,y,z,density_type`

**Combined files**: CSV with multiple density types:
```csv
x,y,z,density_type,density_value,voxel_points,scaling_factor,vertical_calibration,calibration_mode
```

## Advanced Usage

### Custom Analysis Scripts

```python
from config_manager import ConfigManager
from main import VoxPlotAnalysis

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("my_config.yaml")

# Run analysis
analysis = VoxPlotAnalysis(config)
success = analysis.run_analysis()
```

### Programmatic Configuration

```python
config = {
    'analysis': {
        'crown_base_height': 1.0,
        'voxel_size': 0.5,
        'min_density': 0.1,
        'comparison_mode': 'same_density_type_different_model_type',
        'density_types': ['lad'],
        'output_dir': 'programmatic_results'
    },
    'models': {
        'MyModel': {
            'model_type': 'amapvox',
            'file_paths': {'lad': 'path/to/data.csv'}
        }
    }
}
```

## Output Interpretation

### Crown Layer Metrics

- **Total Area**: Sum of leaf/wood/plant area in the layer (m²)
- **Area Index**: Mean area index (LAI/WAI/PAI) across the layer (m²/m²)
- **Ground Area**: Total ground area covered by voxels (m²)
- **Voxel Count**: Number of voxels in the layer

### Visualizations

1. **Crown Layer Matrices**: Side-by-side comparison of crown layers across models
2. **Vertical Profiles**: Height-based density distribution plots  
3. **Distribution Plots**: Statistical distribution comparisons
4. **3D Visualizations**: Interactive 3D voxel representations

## Troubleshooting

### Common Issues

**File Not Found Errors**
```bash
# Check file paths in configuration
python main.py --config my_config.yaml --dry-run
```

**Memory Issues with Large Datasets**
- Increase `min_density` threshold to filter sparse data
- Reduce `voxel_size` for coarser analysis
- Process subsets of data separately

**Visualization Issues**
- Ensure matplotlib backend is properly configured
- Check available system memory for large plots
- Reduce figure sizes in configuration if needed

### Debug Mode
```bash
# Enable verbose logging
python main.py --config my_config.yaml --verbose

# Check configuration without running
python main.py --config my_config.yaml --dry-run
```

## Examples

### Example 1: LAD Comparison Between Models
```yaml
analysis:
  comparison_mode: "same_density_type_different_model_type"
  density_types: ["lad"]
  crown_base_height: 0.7
  
models:
  AmapVox: {...}
  VoxLAD: {...}
  VoxPy: {...}
```

### Example 2: Multi-Density Analysis for Single Model
```yaml
analysis:
  comparison_mode: "different_density_type_same_model_type"
  density_types: ["wad", "lad", "pad"]
  
models:
  MyModel: {...}
```

### Example 3: Comprehensive Analysis
```yaml
analysis:
  comparison_mode: "different_density_type_different_model_type"
  density_types: ["wad", "lad", "pad"]
  
models:
  Model1: {...}
  Model2: {...}
  Model3: {...}
```

## Contributing

### Development Setup
```bash
git clone https://github.com/your-repo/voxplot.git
cd voxplot
pip install -e .
pip install -r requirements-dev.txt
```

### Code Style
- Follow PEP 8 standards
- Use type hints where appropriate
- Include comprehensive docstrings
- Add unit tests for new features

### Testing
```bash
pytest tests/
pytest --cov=voxplot tests/
```

## Citation

If you use VoxPlot in your research, please cite:

```bibtex
@software{voxplot2024,
  title={VoxPlot: Advanced Voxel-based Forest Structure Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/voxplot}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 Documentation: [Link to docs]
- 🐛 Bug Reports: [GitHub Issues]
- 💬 Discussions: [GitHub Discussions]
- 📧 Email: your.email@institution.edu

## Acknowledgments

- AmapVox team for the voxel modeling framework
- VoxLAD developers for multi-scan LAD estimation
- VoxPy contributors for advanced forest structure modeling
- Scientific Python community for essential tools and libraries