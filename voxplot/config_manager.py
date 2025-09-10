#!/usr/bin/env python3
"""
Configuration management module for VoxPlot analysis.

This module handles loading and validating YAML configuration files,
CLI argument parsing, and providing configuration access throughout the application.
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys


class ConfigManager:
    """Manages configuration loading, validation, and access."""
    
    def __init__(self):
        self.config = None
        self._valid_model_types = {"amapvox", "voxlad", "voxpy"}
        self._valid_density_types = {"wad", "lad", "pad"}
        self._valid_comparison_modes = {
            "same_density_type_same_model_type",
            "same_density_type_different_model_type", 
            "different_density_type_same_model_type",
            "different_density_type_different_model_type"
        }
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="VoxPlot: Advanced voxel-based forest structure analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py --config config.yaml
  python main.py --config config.yaml --output-dir custom_results
  python main.py --config config.yaml --comparison-mode same_density_type_different_model_type
            """
        )
        
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to YAML configuration file"
        )
        
        parser.add_argument(
            "--output-dir",
            type=str,
            help="Override output directory from config file"
        )
        
        parser.add_argument(
            "--comparison-mode",
            type=str,
            choices=self._valid_comparison_modes,
            help="Override comparison mode from config file"
        )
        
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without running analysis"
        )
        
        return parser.parse_args()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
        
        self.config = config
        self._validate_config()
        return self.config
    
    def _validate_config(self):
        """Validate configuration structure and values."""
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate analysis section
        if "analysis" not in self.config:
            raise ValueError("Missing 'analysis' section in configuration")
        
        analysis = self.config["analysis"]
        self._validate_analysis_config(analysis)
        
        # Validate models section
        if "models" not in self.config:
            raise ValueError("Missing 'models' section in configuration")
        
        models = self.config["models"]
        self._validate_models_config(models)
    
    def _validate_analysis_config(self, analysis: Dict[str, Any]):
        """Validate analysis configuration parameters."""
        required_params = ["crown_base_height", "voxel_size", "comparison_mode"]
        for param in required_params:
            if param not in analysis:
                raise ValueError(f"Missing required analysis parameter: {param}")
        
        # Validate numeric parameters
        if not isinstance(analysis["crown_base_height"], (int, float)) or analysis["crown_base_height"] < 0:
            raise ValueError("crown_base_height must be a non-negative number")
        
        if not isinstance(analysis["voxel_size"], (int, float)) or analysis["voxel_size"] <= 0:
            raise ValueError("voxel_size must be a positive number")
        
        if analysis.get("min_density", 0) < 0:
            raise ValueError("min_density must be non-negative")
        
        # Validate comparison mode
        if analysis["comparison_mode"] not in self._valid_comparison_modes:
            raise ValueError(f"Invalid comparison_mode. Must be one of: {self._valid_comparison_modes}")
        
        # Validate density types
        if "density_types" in analysis:
            for dtype in analysis["density_types"]:
                if dtype not in self._valid_density_types:
                    raise ValueError(f"Invalid density type: {dtype}")
    
    def _validate_models_config(self, models: Dict[str, Any]):
        """Validate models configuration."""
        if not models:
            raise ValueError("At least one model must be defined")
        
        for model_name, model_config in models.items():
            self._validate_single_model_config(model_name, model_config)
    
    def _validate_single_model_config(self, model_name: str, model_config: Dict[str, Any]):
        """Validate a single model configuration."""
        if not isinstance(model_config, dict):
            raise ValueError(f"Model '{model_name}' configuration must be a dictionary")
        
        # Validate model_type
        if "model_type" not in model_config:
            raise ValueError(f"Model '{model_name}' missing model_type")
        
        model_type = model_config["model_type"]
        if model_type not in self._valid_model_types:
            raise ValueError(f"Model '{model_name}' has invalid model_type: {model_type}")
        
        # Validate file_paths
        if "file_paths" not in model_config:
            raise ValueError(f"Model '{model_name}' missing file_paths")
        
        file_paths = model_config["file_paths"]
        if not isinstance(file_paths, dict) or not file_paths:
            raise ValueError(f"Model '{model_name}' file_paths must be a non-empty dictionary")
        
        # Special validation for VoxPy models
        if model_type == "voxpy":
            self._validate_voxpy_file_paths(model_name, file_paths)
        else:
            # Validate standard density types in file_paths
            for density_type, file_path in file_paths.items():
                self._validate_file_path_entry(model_name, density_type, file_path, model_type)
    
    def _validate_voxpy_file_paths(self, model_name: str, file_paths: Dict[str, str]):
        """Validate VoxPy model file paths (can have either individual files OR combined file)."""
        has_combined = "combined" in file_paths
        has_individual = any(dt in file_paths for dt in self._valid_density_types)
        
        if has_combined and has_individual:
            raise ValueError(f"Model '{model_name}' cannot have both 'combined' file and individual density files. Choose one approach.")
        
        if not has_combined and not has_individual:
            raise ValueError(f"Model '{model_name}' must have either a 'combined' file or individual density files.")
        
        # Validate each file path entry
        for density_type, file_path in file_paths.items():
            if has_combined and density_type == "combined":
                # Allow 'combined' for VoxPy models
                self._validate_file_path_entry(model_name, density_type, file_path, "voxpy", required=True)
            elif density_type in self._valid_density_types:
                # Standard density types (optional files)
                self._validate_file_path_entry(model_name, density_type, file_path, "voxpy", required=False)
            elif density_type == "use_combined_file":
                # Allow configuration option for combined file mode
                if not isinstance(file_path, bool):
                    raise ValueError(f"Model '{model_name}' use_combined_file option must be true or false")
            else:
                raise ValueError(f"Model '{model_name}' has invalid density type: {density_type}")
    
    def _validate_file_path_entry(self, model_name: str, density_type: str, file_path: str, 
                                 model_type: str, required: bool = False):
        """Validate a single file path entry."""
        if not isinstance(file_path, str) or not file_path.strip():
            if required:
                raise ValueError(f"Model '{model_name}' has invalid file path for {density_type}")
            return  # Skip validation for optional empty paths
        
        # Check if file exists (make this optional for non-critical files)
        path = Path(file_path)
        if not path.exists():
            if required or model_type == "voxpy" and density_type == "combined":
                raise FileNotFoundError(f"File not found for model '{model_name}', density '{density_type}': {file_path}")
            else:
                # Log warning but don't fail validation
                print(f"Warning: File not found for model '{model_name}', density '{density_type}': {file_path}")
                print(f"This file will be skipped during analysis.")
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration parameters."""
        return self.config.get("analysis", {})
    
    def get_models_config(self) -> Dict[str, Any]:
        """Get models configuration."""
        return self.config.get("models", {})
    
    def get_output_dir(self) -> str:
        """Get output directory from configuration."""
        return self.config.get("analysis", {}).get("output_dir", "voxplot_results")
    
    def get_comparison_mode(self) -> str:
        """Get comparison mode from configuration."""
        return self.config.get("analysis", {}).get("comparison_mode", "same_density_type_different_model_type")
    
    def get_density_types(self) -> List[str]:
        """Get list of density types to analyze."""
        return self.config.get("analysis", {}).get("density_types", ["lad"])
    
    def override_config(self, **overrides):
        """Override configuration parameters with command line arguments."""
        analysis = self.config.setdefault("analysis", {})
        
        for key, value in overrides.items():
            if value is not None:
                if key == "output_dir":
                    analysis["output_dir"] = value
                elif key == "comparison_mode":
                    analysis["comparison_mode"] = value
                else:
                    analysis[key] = value
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        
        analysis = self.get_analysis_config()
        print(f"Crown base height: {analysis.get('crown_base_height', 'N/A')} m")
        print(f"Voxel size: {analysis.get('voxel_size', 'N/A')} m")
        print(f"Minimum density: {analysis.get('min_density', 0)}")
        print(f"Comparison mode: {analysis.get('comparison_mode', 'N/A')}")
        print(f"Density types: {', '.join(analysis.get('density_types', []))}")
        print(f"Output directory: {analysis.get('output_dir', 'N/A')}")
        
        models = self.get_models_config()
        print(f"\nModels to analyze: {len(models)}")
        for model_name, model_config in models.items():
            print(f"  - {model_name} ({model_config.get('model_type', 'unknown')})")
            file_paths = model_config.get('file_paths', {})
            for density_type in file_paths:
                print(f"    {density_type}: ✓")
        
        print("="*60 + "\n")


def create_example_config() -> str:
    """Create an example configuration file content."""
    example_config = """
# VoxPlot Configuration File
# This file defines the analysis parameters and data sources for voxel-based forest structure analysis

analysis:
  # Height above ground level to consider as crown base (meters)
  crown_base_height: 0.7
  
  # Voxel size for analysis (meters)
  voxel_size: 0.25
  
  # Minimum density threshold for analysis
  min_density: 0.05
  
  # Comparison mode for analysis
  # Options: same_density_type_same_model_type, same_density_type_different_model_type,
  #          different_density_type_same_model_type, different_density_type_different_model_type
  comparison_mode: "same_density_type_different_model_type"
  
  # Density types to analyze
  density_types: ["lad", "wad", "pad"]
  
  # Output directory for results
  output_dir: "voxplot_results"
  
  # Additional visualization parameters
  visualization:
    figsize_large: [18, 20]
    figsize_medium: [15, 12] 
    figsize_small: [12, 8]
    dpi: 300
    color_scale_max: 40.0

models:
  # AmapVox TLS model
  AmapVox_TLS:
    model_type: "amapvox"
    file_paths:
      pad: "/path/to/amapvox_tls_pad.csv"
      wad: "/path/to/amapvox_tls_wad.csv"
      lad: "/path/to/amapvox_tls_lad.csv"
  
  # AmapVox ULS model  
  AmapVox_ULS:
    model_type: "amapvox"
    file_paths:
      pad: "/path/to/amapvox_uls_pad.csv"
      wad: "/path/to/amapvox_uls_wad.csv"
  
  # VoxLAD TLS model
  VoxLAD_TLS:
    model_type: "voxlad"
    file_paths:
      pad: "/path/to/voxlad_tls_pad.txt"
      wad: "/path/to/voxlad_tls_wad.txt"
      lad: "/path/to/voxlad_tls_lad.txt"
  
  # VoxPy combined model
  VoxPy_Combined:
    model_type: "voxpy"
    file_paths:
      combined: "/path/to/voxpy_combined.csv"  # Contains all density types in one file
"""
    return example_config.strip()