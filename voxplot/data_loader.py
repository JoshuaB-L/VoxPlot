#!/usr/bin/env python3
"""
Data loading module for VoxPlot analysis.

This module handles loading and preprocessing of voxel-based forest structure data
from different sources (AmapVox, VoxLAD, VoxPy) and formats (WAD, LAD, PAD).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

from utils import (
    validate_dataframe, normalize_height, validate_file_path,
    get_density_column_name, format_model_name
)


class DataLoader:
    """Handles loading and preprocessing of voxel data from multiple sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._supported_model_types = {"amapvox", "voxlad", "voxpy"}
        self._supported_density_types = {"wad", "lad", "pad"}
    
    def load_model_data(self, model_name: str, model_config: Dict) -> Dict[str, pd.DataFrame]:
        """
        Load all data for a given model configuration.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration dictionary
        
        Returns:
            Dictionary mapping density types to DataFrames
        """
        model_type = model_config.get("model_type")
        file_paths = model_config.get("file_paths", {})
        
        self.logger.info(f"Loading data for model: {model_name} (type: {model_type})")
        
        datasets = {}
        
        # Check for VoxPy combined file mode
        if model_type == "voxpy":
            use_combined = file_paths.get("use_combined_file", False)
            has_combined_path = "combined" in file_paths
            
            if use_combined and has_combined_path:
                # Use combined file mode
                try:
                    datasets = self._load_voxpy_combined(model_name, file_paths["combined"])
                except Exception as e:
                    self.logger.error(f"Failed to load VoxPy combined file for {model_name}: {e}")
            elif use_combined and not has_combined_path:
                self.logger.error(f"Model {model_name} is set to use combined file but no 'combined' path provided")
            else:
                # Use individual files mode
                datasets = self._load_individual_files(model_name, model_type, file_paths)
        else:
            # Handle individual density type files for non-VoxPy models
            datasets = self._load_individual_files(model_name, model_type, file_paths)
        
        if not datasets:
            self.logger.warning(f"No datasets successfully loaded for model: {model_name}")
        else:
            loaded_types = list(datasets.keys())
            self.logger.info(f"Model {model_name} loaded with density types: {loaded_types}")
        
        return datasets
    
    def _load_individual_files(self, model_name: str, model_type: str, file_paths: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Load individual density type files."""
        datasets = {}
        
        for density_type, file_path in file_paths.items():
            # Skip configuration options
            if density_type in ["use_combined_file", "combined"]:
                continue
                
            # Skip if file_path is empty or None
            if not file_path or not str(file_path).strip():
                self.logger.info(f"Skipping {model_name}_{density_type}: No file path specified")
                continue
            
            # Check if file exists before attempting to load
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                self.logger.warning(f"Skipping {model_name}_{density_type}: File not found - {file_path}")
                continue
            
            try:
                df = self._load_single_file(model_name, model_type, density_type, file_path)
                if df is not None and not df.empty:
                    datasets[density_type] = df
                    self.logger.info(f"Successfully loaded {len(df)} records for {model_name}_{density_type}")
                else:
                    self.logger.warning(f"No data loaded for {model_name}_{density_type}")
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}_{density_type}: {e}")
                continue
        
        return datasets
    
    def _load_single_file(self, model_name: str, model_type: str, 
                         density_type: str, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a single data file based on model type and density type.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (amapvox, voxlad, voxpy)
            density_type: Type of density (wad, lad, pad)
            file_path: Path to the data file
        
        Returns:
            Loaded and preprocessed DataFrame
        """
        # Validate file path (not required, so missing files return None)
        file_path_obj = validate_file_path(file_path, model_name, density_type, required=False)
        if file_path_obj is None:
            return None
        
        # Load based on model type
        if model_type == "voxlad":
            df = self._load_voxlad_data(file_path_obj, density_type)
        elif model_type == "amapvox":
            df = self._load_amapvox_data(file_path_obj, density_type)
        elif model_type == "voxpy":
            df = self._load_voxpy_data(file_path_obj, density_type)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        if df is not None and not df.empty:
            # Add metadata
            df = self._add_metadata(df, model_name, model_type, density_type)
            
            # Normalize height
            df = normalize_height(df)
            
            # Validate final DataFrame
            self._validate_loaded_data(df, model_name, density_type)
        
        return df
    
    def _load_voxlad_data(self, file_path: Path, density_type: str) -> Optional[pd.DataFrame]:
        """
        Load VoxLAD format data with flexible scan detection.
        
        VoxLAD format (space-delimited):
        x y z density_1 percent_explored_1 density_2 percent_explored_2 ... best_scan
        
        The number of scans is detected automatically from the file structure.
        """
        self.logger.debug(f"Loading VoxLAD file: {file_path.name}")
        
        try:
            # First, read a sample line to determine the number of columns
            with open(file_path, 'r') as f:
                # Skip any potential header lines and get the first data line
                lines = f.readlines()
                
            # Find the first line with numeric data
            sample_line = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and len(line.split()) > 4:
                    # Check if this looks like a data line (starts with numbers)
                    parts = line.split()
                    try:
                        float(parts[0])  # Try to convert first element to float
                        float(parts[1])  # Try to convert second element to float
                        sample_line = line
                        break
                    except (ValueError, IndexError):
                        continue
            
            if sample_line is None:
                self.logger.error(f"Could not find valid data line in VoxLAD file: {file_path}")
                return None
            
            # Determine number of columns from sample line
            num_columns = len(sample_line.split())
            self.logger.debug(f"Detected {num_columns} columns in VoxLAD file")
            
            # Calculate number of scans: (num_columns - 4) / 2
            # Format: x, y, z, [density_1, percent_1, density_2, percent_2, ...], best_scan
            if num_columns < 6:  # Minimum: x, y, z, density_1, percent_1, best_scan
                self.logger.error(f"VoxLAD file has too few columns ({num_columns}). Expected at least 6.")
                return None
            
            num_scan_pairs = (num_columns - 4) // 2
            if (num_columns - 4) % 2 != 0:
                self.logger.error(f"VoxLAD file has invalid column structure. Expected paired density/percent columns.")
                return None
            
            self.logger.info(f"Detected {num_scan_pairs} scans in VoxLAD file")
            
            # Build column names dynamically
            column_names = ['x', 'y', 'z']
            for i in range(1, num_scan_pairs + 1):
                column_names.extend([f'{density_type}_{i}', f'percent_explored_{i}'])
            column_names.append('best_scan')
            
            self.logger.debug(f"Using column names: {column_names}")
            
            # Read the file with determined structure
            # Try without skipping rows first (no header)
            try:
                df = pd.read_csv(file_path, delim_whitespace=True, names=column_names, header=None)
                
                # Check if first row looks like header (contains non-numeric data)
                first_row_is_header = False
                try:
                    # Try to convert first row values to float
                    for i, col in enumerate(['x', 'y', 'z']):
                        float(df.iloc[0, i])
                except (ValueError, TypeError):
                    first_row_is_header = True
                
                if first_row_is_header:
                    # Re-read skipping the header row
                    df = pd.read_csv(file_path, delim_whitespace=True, names=column_names, skiprows=1)
                    
            except Exception:
                # Fallback: try skipping first row
                df = pd.read_csv(file_path, delim_whitespace=True, names=column_names, skiprows=1)
            
            # Validate that we have the expected columns
            required_cols = ['x', 'y', 'z', 'best_scan']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns in VoxLAD file: {missing_cols}")
                return None
            
            # Clean the data - remove rows with NaN in critical columns
            initial_len = len(df)
            df = df.dropna(subset=['x', 'y', 'z', 'best_scan'])
            
            if len(df) < initial_len:
                self.logger.warning(f"Removed {initial_len - len(df)} rows with NaN values in critical columns")
            
            if len(df) == 0:
                self.logger.error("No valid data rows remaining after cleaning")
                return None
            
            # Ensure best_scan values are valid integers
            df['best_scan'] = pd.to_numeric(df['best_scan'], errors='coerce')
            df = df.dropna(subset=['best_scan'])
            df['best_scan'] = df['best_scan'].astype(int)
            
            # Validate best_scan values are within expected range
            valid_scan_range = range(1, num_scan_pairs + 1)
            invalid_scans = ~df['best_scan'].isin(valid_scan_range)
            if invalid_scans.any():
                self.logger.warning(f"Found {invalid_scans.sum()} rows with invalid best_scan values. Setting to 1.")
                df.loc[invalid_scans, 'best_scan'] = 1
            
            # Select the best density value based on best_scan column
            def get_best_density(row):
                try:
                    scan_num = int(row['best_scan'])
                    density_col = f'{density_type}_{scan_num}'
                    if density_col in df.columns:
                        value = row[density_col]
                        # Handle NaN values
                        if pd.isna(value):
                            # Try other scans if the best scan has NaN
                            for alt_scan in valid_scan_range:
                                alt_col = f'{density_type}_{alt_scan}'
                                if alt_col in df.columns and pd.notna(row[alt_col]):
                                    return row[alt_col]
                            return 0.0  # Default if all scans have NaN
                        return value
                    else:
                        # Fallback to first available density column
                        return row[f'{density_type}_1'] if f'{density_type}_1' in df.columns else 0.0
                except (ValueError, TypeError):
                    return 0.0
            
            df[density_type] = df.apply(get_best_density, axis=1)
            
            # Clean final density values
            df[density_type] = pd.to_numeric(df[density_type], errors='coerce').fillna(0.0)
            
            # Keep only required columns
            final_df = df[['x', 'y', 'z', density_type]].copy()
            
            # Final validation
            if len(final_df) == 0:
                self.logger.error("No valid data remaining after processing")
                return None
            
            self.logger.info(f"Successfully processed VoxLAD file with {len(final_df)} valid records")
            return final_df
            
        except Exception as e:
            self.logger.error(f"Error loading VoxLAD data from {file_path}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def _load_amapvox_data(self, file_path: Path, density_type: str) -> Optional[pd.DataFrame]:
        """
        Load AmapVox format data.
        
        AmapVox format (CSV):
        x,y,z,lad (or wad/pad)
        """
        self.logger.debug(f"Loading AmapVox file: {file_path.name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Check if the density column exists
            if density_type not in df.columns:
                self.logger.error(f"Density column '{density_type}' not found in {file_path}")
                return None
            
            # Ensure required spatial columns exist
            required_columns = ['x', 'y', 'z', density_type]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns {missing_columns} in {file_path}")
                return None
            
            return df[required_columns]
            
        except Exception as e:
            self.logger.error(f"Error loading AmapVox data from {file_path}: {e}")
            return None
    
    def _load_voxpy_data(self, file_path: Path, density_type: str) -> Optional[pd.DataFrame]:
        """
        Load VoxPy format data (simple format).
        
        VoxPy format (CSV):
        x,y,z,lad (or wad/pad) or x,y,z,voxel_points
        """
        self.logger.debug(f"Loading VoxPy file: {file_path.name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Handle different VoxPy column naming conventions
            if density_type in df.columns:
                required_columns = ['x', 'y', 'z', density_type]
            elif 'voxel_points' in df.columns and density_type == 'lad':
                # Some VoxPy files use 'voxel_points' instead of 'lad'
                df = df.rename(columns={'voxel_points': density_type})
                required_columns = ['x', 'y', 'z', density_type]
            else:
                self.logger.error(f"Density column '{density_type}' not found in {file_path}")
                return None
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns {missing_columns} in {file_path}")
                return None
            
            return df[required_columns]
            
        except Exception as e:
            self.logger.error(f"Error loading VoxPy data from {file_path}: {e}")
            return None
    
    def _load_voxpy_combined(self, model_name: str, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load VoxPy combined format data containing multiple density types.
        
        VoxPy combined format:
        x,y,z,density_type,density_value,voxel_points,scaling_factor,vertical_calibration,calibration_mode
        """
        file_path_obj = validate_file_path(file_path, model_name, "combined", required=True)
        if file_path_obj is None:
            return {}
            
        self.logger.debug(f"Loading VoxPy combined file: {file_path_obj.name}")
        
        datasets = {}
        
        try:
            df = pd.read_csv(file_path_obj)
            
            # Validate required columns
            required_columns = ['x', 'y', 'z', 'density_type', 'density_value']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns {missing_columns} in {file_path_obj}")
                return datasets
            
            # Get unique density types and filter for supported ones
            available_density_types = df['density_type'].unique()
            self.logger.info(f"Found density types in combined file: {list(available_density_types)}")
            
            # Split by density type
            for density_type in available_density_types:
                # Convert to lowercase and check if it's a supported density type
                density_type_lower = str(density_type).lower()
                if density_type_lower in self._supported_density_types:
                    density_df = df[df['density_type'] == density_type].copy()
                    
                    if len(density_df) == 0:
                        self.logger.warning(f"No data found for density type {density_type} in combined file")
                        continue
                    
                    # Rename density_value column to match density type
                    density_column = density_type_lower
                    density_df[density_column] = density_df['density_value']
                    
                    # Keep only required columns
                    final_df = density_df[['x', 'y', 'z', density_column]].copy()
                    
                    # Add metadata
                    final_df = self._add_metadata(final_df, model_name, "voxpy", density_column)
                    
                    # Normalize height
                    final_df = normalize_height(final_df)
                    
                    # Validate the data
                    try:
                        self._validate_loaded_data(final_df, model_name, density_column)
                        datasets[density_column] = final_df
                        self.logger.info(f"Successfully loaded {len(final_df)} records for {model_name}_{density_column}")
                    except Exception as e:
                        self.logger.warning(f"Validation failed for {model_name}_{density_column}: {e}")
                        continue
                else:
                    self.logger.warning(f"Unsupported density type '{density_type}' found in combined file, skipping")
            
            if not datasets:
                self.logger.warning(f"No valid density types found in combined file for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error loading VoxPy combined data from {file_path_obj}: {e}")
        
        return datasets
    
    def _add_metadata(self, df: pd.DataFrame, model_name: str, 
                     model_type: str, density_type: str) -> pd.DataFrame:
        """Add metadata columns to DataFrame."""
        df = df.copy()
        df['model_name'] = model_name
        df['model_type'] = model_type
        df['density_type'] = density_type
        df['display_name'] = format_model_name(model_name, model_type, density_type)
        return df
    
    def _validate_loaded_data(self, df: pd.DataFrame, model_name: str, density_type: str):
        """Validate loaded data meets requirements."""
        required_columns = ['x', 'y', 'z', density_type]
        data_description = f"{model_name}_{density_type}"
        
        validate_dataframe(df, required_columns, data_description)
        
        # Additional validation for spatial extent
        if len(df) == 0:
            raise ValueError(f"No data loaded for {data_description}")
        
        # Check for reasonable spatial extents
        for col in ['x', 'y', 'z']:
            col_range = df[col].max() - df[col].min()
            if col_range <= 0:
                self.logger.warning(f"Zero or negative range in {col} for {data_description}")
        
        # Check density values
        density_stats = df[density_type].describe()
        if density_stats['max'] <= 0:
            self.logger.warning(f"No positive density values found for {data_description}")


class DataProcessor:
    """Processes loaded data for analysis and comparison."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def organize_datasets_by_comparison_mode(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]], 
                                           comparison_mode: str) -> List[List[pd.DataFrame]]:
        """
        Organize datasets into comparison groups based on comparison mode.
        
        Args:
            all_datasets: Dictionary mapping model names to their density datasets
            comparison_mode: Type of comparison to perform
        
        Returns:
            List of dataset groups for comparison
        """
        self.logger.info(f"Organizing datasets for comparison mode: {comparison_mode}")
        
        if comparison_mode == "same_density_type_same_model_type":
            return self._group_same_density_same_model(all_datasets)
        elif comparison_mode == "same_density_type_different_model_type":
            return self._group_same_density_different_model(all_datasets)
        elif comparison_mode == "different_density_type_same_model_type":
            return self._group_different_density_same_model(all_datasets)
        elif comparison_mode == "different_density_type_different_model_type":
            return self._group_different_density_different_model(all_datasets)
        else:
            raise ValueError(f"Unsupported comparison mode: {comparison_mode}")
    
    def _group_same_density_same_model(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> List[List[pd.DataFrame]]:
        """Group datasets with same density type and same model type."""
        groups = []
        
        # Group by model type and density type
        model_density_groups = {}
        
        for model_name, datasets in all_datasets.items():
            for density_type, df in datasets.items():
                model_type = df['model_type'].iloc[0]
                group_key = f"{model_type}_{density_type}"
                
                if group_key not in model_density_groups:
                    model_density_groups[group_key] = []
                
                model_density_groups[group_key].append(df)
        
        # Only include groups with multiple datasets
        for group_datasets in model_density_groups.values():
            if len(group_datasets) > 1:
                groups.append(group_datasets)
        
        return groups
    
    def _group_same_density_different_model(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> List[List[pd.DataFrame]]:
        """Group datasets with same density type but different model types."""
        groups = []
        
        # Group by density type
        density_groups = {}
        
        for model_name, datasets in all_datasets.items():
            for density_type, df in datasets.items():
                if density_type not in density_groups:
                    density_groups[density_type] = []
                
                density_groups[density_type].append(df)
        
        # Only include groups with multiple datasets from different model types
        for density_type, group_datasets in density_groups.items():
            if len(group_datasets) > 1:
                # Check if they have different model types
                model_types = set(df['model_type'].iloc[0] for df in group_datasets)
                if len(model_types) > 1:
                    groups.append(group_datasets)
        
        return groups
    
    def _group_different_density_same_model(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> List[List[pd.DataFrame]]:
        """Group datasets with different density types but same model."""
        groups = []
        
        for model_name, datasets in all_datasets.items():
            if len(datasets) > 1:
                # Multiple density types for the same model
                groups.append(list(datasets.values()))
        
        return groups
    
    def _group_different_density_different_model(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> List[List[pd.DataFrame]]:
        """Group all datasets together for comprehensive comparison."""
        all_datasets_list = []
        
        for model_name, datasets in all_datasets.items():
            for density_type, df in datasets.items():
                all_datasets_list.append(df)
        
        if len(all_datasets_list) > 1:
            return [all_datasets_list]
        else:
            return []
    
    def create_dataset_summary(self, all_datasets: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Create a summary table of all loaded datasets."""
        summary_data = []
        
        for model_name, datasets in all_datasets.items():
            for density_type, df in datasets.items():
                summary_data.append({
                    'Model': model_name,
                    'Model_Type': df['model_type'].iloc[0],
                    'Density_Type': density_type.upper(),
                    'Records': len(df),
                    'X_Range': f"{df['x'].min():.2f} to {df['x'].max():.2f}",
                    'Y_Range': f"{df['y'].min():.2f} to {df['y'].max():.2f}",
                    'Z_Range': f"{df['z'].min():.2f} to {df['z'].max():.2f}",
                    'Density_Range': f"{df[density_type].min():.3f} to {df[density_type].max():.3f}",
                    'Mean_Density': f"{df[density_type].mean():.3f}",
                    'Nonzero_Count': len(df[df[density_type] > 0])
                })
        
        return pd.DataFrame(summary_data)