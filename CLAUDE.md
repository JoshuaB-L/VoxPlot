# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VoxPlot is a Python framework for analyzing and visualizing forest structure data from voxel-based models (AmapVox, VoxLAD, VoxPy). It performs crown layer analysis, spatial distribution assessment, and comparative visualizations for Wood Area Density (WAD), Leaf Area Density (LAD), and Plant Area Density (PAD) metrics.

## Key Architecture

### Core Components
- **main.py**: EnhancedVoxPlotAnalysis coordinator class manages the analysis pipeline
- **config_manager.py**: ConfigManager handles YAML configuration loading and validation
- **data_loader.py**: DataLoader and DataProcessor classes for reading various voxel formats
- **data_analyzer.py**: ForestStructureAnalyzer and ComparativeAnalyzer for metrics calculation
- **visualizer.py**: EnhancedResultsManager for publication-quality figure generation
- **utils.py**: Utility functions for data processing, statistics, and file operations

### Data Flow
1. Configuration loaded from YAML files specifying models, density types, and analysis parameters
2. Data loaded from CSV/TXT files in AmapVox, VoxLAD, or VoxPy formats
3. Crown layers calculated based on height percentiles (upper/middle/lower)
4. Comparative analysis performed based on comparison_mode setting
5. Results exported as figures, data tables, and summary reports

## Development Commands

### Installation
```bash
pip install -r requirements.txt
# For development with additional tools:
pip install -e .
```

### Running Analysis
```bash
# Basic analysis with config file
python main.py --config config.yaml

# With verbose output
python main.py --config config.yaml --verbose

# Validate configuration without running
python main.py --config config.yaml --dry-run
```

### Testing
```bash
# Run installation test
python test_installation.py

# Test with example data
python example_usage.py
```

## Configuration Structure

Analysis configurations use YAML format with two main sections:

**analysis**: Parameters for crown_base_height, voxel_size, min_density, comparison_mode, density_types, output_dir

**models**: Dictionary of model configurations with model_type and file_paths for each density type

Comparison modes:
- same_density_type_different_model_type: Compare LAD across AmapVox vs VoxLAD
- different_density_type_same_model_type: Compare LAD vs WAD within same model
- different_density_type_different_model_type: All combinations

## File Format Requirements

**AmapVox**: CSV with columns x,y,z,density_value

**VoxLAD**: Space-delimited with x y z lad_1 percent_explored_1 (multiple scan columns)

**VoxPy**: CSV with x,y,z,density_type OR combined format with density_type,density_value columns