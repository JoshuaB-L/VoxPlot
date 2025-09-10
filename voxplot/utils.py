#!/usr/bin/env python3
"""
Enhanced utility functions for VoxPlot analysis with fixes for:
1. Equal layer division by voxel count instead of height
2. Improved data validation and processing
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import logging


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration for the application."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('voxplot')


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], data_type: str = "dataset") -> bool:
    """
    Validate that a DataFrame contains required columns and has valid data.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        data_type: Description of data type for error messages
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    if df is None or df.empty:
        raise ValueError(f"{data_type} is empty or None")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{data_type} missing required columns: {missing_columns}")
    
    # Check for non-finite values in numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in required_columns:
            if df[col].isna().any():
                raise ValueError(f"{data_type} contains NaN values in column '{col}'")
            if not np.isfinite(df[col]).all():
                raise ValueError(f"{data_type} contains infinite values in column '{col}'")
    
    return True


def normalize_height(df: pd.DataFrame, height_column: str = 'z') -> pd.DataFrame:
    """
    Normalize height values by subtracting the minimum height.
    
    Args:
        df: DataFrame containing height data
        height_column: Name of the height column
    
    Returns:
        DataFrame with normalized heights
    """
    df = df.copy()
    if height_column in df.columns:
        min_height = df[height_column].min()
        df[height_column] = df[height_column] - min_height
    return df


def filter_by_density_threshold(df: pd.DataFrame, density_column: str, threshold: float = 0.0) -> pd.DataFrame:
    """
    Filter DataFrame by density threshold.
    
    Args:
        df: DataFrame containing density data
        density_column: Name of the density column
        threshold: Minimum density threshold
    
    Returns:
        Filtered DataFrame
    """
    if density_column not in df.columns:
        raise ValueError(f"Density column '{density_column}' not found in DataFrame")
    
    return df[df[density_column] > threshold].copy()


def filter_by_height_range(df: pd.DataFrame, height_column: str = 'z', 
                          min_height: Optional[float] = None, 
                          max_height: Optional[float] = None) -> pd.DataFrame:
    """
    Filter DataFrame by height range.
    
    Args:
        df: DataFrame containing height data
        height_column: Name of the height column
        min_height: Minimum height threshold
        max_height: Maximum height threshold
    
    Returns:
        Filtered DataFrame
    """
    if height_column not in df.columns:
        raise ValueError(f"Height column '{height_column}' not found in DataFrame")
    
    filtered_df = df.copy()
    
    if min_height is not None:
        filtered_df = filtered_df[filtered_df[height_column] >= min_height]
    
    if max_height is not None:
        filtered_df = filtered_df[filtered_df[height_column] <= max_height]
    
    return filtered_df


def calculate_crown_layer_bounds_by_count(df: pd.DataFrame, height_column: str = 'z', 
                                        crown_base_height: float = 0.7, 
                                        num_layers: int = 3) -> Tuple[float, List[float]]:
    """
    Calculate crown layer boundaries based on equal voxel counts, not height.
    
    This ensures each layer contains approximately the same number of valid voxel points
    above the minimum threshold, which provides more balanced layer analysis.
    
    Args:
        df: DataFrame containing height data
        height_column: Name of the height column
        crown_base_height: Minimum height to consider as crown
        num_layers: Number of crown layers to create
    
    Returns:
        Tuple of (crown_height, layer_boundaries)
    """
    crown_df = df[df[height_column] >= crown_base_height]
    
    if len(crown_df) == 0:
        return 0.0, []
    
    z_min = crown_df[height_column].min()
    z_max = crown_df[height_column].max()
    crown_height = z_max - z_min
    
    if crown_height <= 0:
        return 0.0, []
    
    # Sort data by height to divide into equal count groups
    sorted_heights = np.sort(crown_df[height_column].values)
    n_voxels = len(sorted_heights)
    
    if n_voxels < num_layers:
        # Not enough voxels to create all layers, fall back to height-based division
        layer_height = crown_height / num_layers
        boundaries = [z_min + i * layer_height for i in range(1, num_layers)]
        return crown_height, boundaries
    
    # Calculate indices for equal count divisions
    voxels_per_layer = n_voxels // num_layers
    remainder = n_voxels % num_layers
    
    # Distribute remainder voxels to lower layers for more balanced distribution
    layer_sizes = [voxels_per_layer] * num_layers
    for i in range(remainder):
        layer_sizes[i] += 1
    
    # Calculate boundary heights based on voxel counts
    boundaries = []
    current_index = 0
    
    for i in range(num_layers - 1):  # num_layers - 1 boundaries
        current_index += layer_sizes[i]
        if current_index < len(sorted_heights):
            boundary_height = sorted_heights[current_index - 1]
            boundaries.append(boundary_height)
    
    return crown_height, boundaries


def calculate_crown_layer_bounds(df: pd.DataFrame, height_column: str = 'z', 
                               crown_base_height: float = 0.7, 
                               num_layers: int = 3) -> Tuple[float, List[float]]:
    """
    Calculate crown layer boundaries using the improved count-based method.
    
    This is the main function that should be used for layer boundary calculation.
    It delegates to the count-based method for more balanced layer analysis.
    """
    return calculate_crown_layer_bounds_by_count(df, height_column, crown_base_height, num_layers)


def create_spatial_grid(df: pd.DataFrame, voxel_size: float, 
                       x_column: str = 'x', y_column: str = 'y') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create spatial grid coordinates based on DataFrame extent and voxel size.
    
    Args:
        df: DataFrame containing spatial data
        voxel_size: Size of each voxel
        x_column: Name of x coordinate column
        y_column: Name of y coordinate column
    
    Returns:
        Tuple of (x_grid, y_grid) arrays
    """
    x_min, x_max = df[x_column].min(), df[x_column].max()
    y_min, y_max = df[y_column].min(), df[y_column].max()
    
    # Expand bounds to ensure all data points are included
    x_min = np.floor(x_min / voxel_size) * voxel_size
    x_max = np.ceil(x_max / voxel_size) * voxel_size
    y_min = np.floor(y_min / voxel_size) * voxel_size
    y_max = np.ceil(y_max / voxel_size) * voxel_size
    
    x_grid = np.arange(x_min, x_max + voxel_size, voxel_size)
    y_grid = np.arange(y_min, y_max + voxel_size, voxel_size)
    
    return x_grid, y_grid


def aggregate_to_grid(df: pd.DataFrame, voxel_size: float, 
                     density_column: str, aggregation_method: str = 'sum',
                     x_column: str = 'x', y_column: str = 'y') -> pd.DataFrame:
    """
    Aggregate data to spatial grid using specified method.
    
    Args:
        df: DataFrame containing spatial and density data
        voxel_size: Size of each voxel
        density_column: Name of density column to aggregate
        aggregation_method: Method to use for aggregation ('sum', 'mean', 'max')
        x_column: Name of x coordinate column
        y_column: Name of y coordinate column
    
    Returns:
        DataFrame with aggregated data on grid
    """
    if aggregation_method not in ['sum', 'mean', 'max']:
        raise ValueError("aggregation_method must be 'sum', 'mean', or 'max'")
    
    # Create grid coordinates
    df_grid = df.copy()
    df_grid['x_grid'] = np.round(df_grid[x_column] / voxel_size) * voxel_size
    df_grid['y_grid'] = np.round(df_grid[y_column] / voxel_size) * voxel_size
    
    # Aggregate by grid coordinates
    agg_func = {'sum': 'sum', 'mean': 'mean', 'max': 'max'}[aggregation_method]
    
    aggregated = df_grid.groupby(['x_grid', 'y_grid'])[density_column].agg(agg_func).reset_index()
    aggregated = aggregated.rename(columns={'x_grid': x_column, 'y_grid': y_column})
    
    return aggregated


def create_2d_raster(df: pd.DataFrame, voxel_size: float, 
                    density_column: str, x_column: str = 'x', y_column: str = 'y',
                    aggregation_method: str = 'sum') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 2D raster array from point data with improved handling of missing data.
    
    Args:
        df: DataFrame containing spatial and density data
        voxel_size: Size of each voxel
        density_column: Name of density column
        x_column: Name of x coordinate column
        y_column: Name of y coordinate column
        aggregation_method: Method to use for aggregation
    
    Returns:
        Tuple of (raster_array, x_coordinates, y_coordinates)
    """
    if len(df) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Filter out zero/negative values for cleaner visualization
    valid_df = df[df[density_column] > 0].copy()
    
    if len(valid_df) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Aggregate to grid
    grid_df = aggregate_to_grid(valid_df, voxel_size, density_column, aggregation_method, x_column, y_column)
    
    if len(grid_df) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Create coordinate arrays
    x_coords = np.sort(grid_df[x_column].unique())
    y_coords = np.sort(grid_df[y_column].unique())
    
    # Create raster array with zeros (will be masked in visualization)
    raster = np.zeros((len(y_coords), len(x_coords)))
    
    for _, row in grid_df.iterrows():
        x_idx = np.where(x_coords == row[x_column])[0][0]
        y_idx = np.where(y_coords == row[y_column])[0][0]
        raster[y_idx, x_idx] = row[density_column]
    
    return raster, x_coords, y_coords


def calculate_statistics(data: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a data array.
    
    Args:
        data: Data array to analyze
    
    Returns:
        Dictionary containing statistical measures
    """
    if len(data) == 0:
        return {
            'count': 0, 'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'q25': 0, 'q75': 0
        }
    
    # Remove non-finite values
    clean_data = data[np.isfinite(data)] if isinstance(data, np.ndarray) else data.dropna()
    
    if len(clean_data) == 0:
        return {
            'count': 0, 'mean': 0, 'median': 0, 'std': 0,
            'min': 0, 'max': 0, 'q25': 0, 'q75': 0
        }
    
    return {
        'count': len(clean_data),
        'mean': float(np.mean(clean_data)),
        'median': float(np.median(clean_data)),
        'std': float(np.std(clean_data)),
        'min': float(np.min(clean_data)),
        'max': float(np.max(clean_data)),
        'q25': float(np.percentile(clean_data, 25)),
        'q75': float(np.percentile(clean_data, 75))
    }


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary with robust error handling.
    
    Args:
        path: Directory path to create
    
    Returns:
        Path object for the directory
    
    Raises:
        OSError: If directory creation fails due to permissions or other issues
    """
    dir_path = Path(path)
    
    try:
        # If directory already exists and is valid, return it
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
        
        # If path exists but is not a directory, this is an error
        if dir_path.exists() and not dir_path.is_dir():
            raise OSError(f"Path exists but is not a directory: {dir_path}")
        
        # Try to create the directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Verify creation was successful
        if not dir_path.exists() or not dir_path.is_dir():
            raise OSError(f"Directory creation failed: {dir_path}")
            
        return dir_path
        
    except FileExistsError:
        # This can happen in race conditions - check if directory now exists
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
        else:
            raise OSError(f"File exists error but directory not found: {dir_path}")
            
    except PermissionError as e:
        raise OSError(f"Permission denied creating directory {dir_path}: {e}")
        
    except OSError as e:
        # Don't double-wrap OSError
        raise e
        
    except Exception as e:
        raise OSError(f"Unexpected error creating directory {dir_path}: {e}")


def get_density_column_name(model_type: str, density_type: str) -> str:
    """
    Get the appropriate density column name based on model type and density type.
    
    Args:
        model_type: Type of model (amapvox, voxlad, voxpy)
        density_type: Type of density (wad, lad, pad)
    
    Returns:
        Column name for the density values
    """
    if model_type == "voxpy" and density_type == "combined":
        return "density_value"
    elif model_type == "voxlad":
        return density_type  # VoxLAD uses generic column selection
    else:
        return density_type  # AmapVox and simple VoxPy use direct column names


def format_model_name(model_name: str, model_type: str, density_type: str) -> str:
    """
    Format model name for display purposes.
    
    Args:
        model_name: Base model name
        model_type: Type of model
        density_type: Type of density
    
    Returns:
        Formatted model name
    """
    if density_type:
        return f"{model_name}_{density_type.upper()}"
    return model_name


def validate_file_path(file_path: str, model_name: str, density_type: str, required: bool = True) -> Optional[Path]:
    """
    Validate that a file path exists and is accessible.
    
    Args:
        file_path: Path to validate
        model_name: Name of the model for error messages
        density_type: Type of density for error messages
        required: Whether the file is required (if False, missing files return None)
    
    Returns:
        Path object if valid, None if missing and not required
    
    Raises:
        FileNotFoundError: If file does not exist and is required
        ValueError: If path is invalid and is required
    """
    if not file_path or not isinstance(file_path, str) or not file_path.strip():
        if required:
            raise ValueError(f"Invalid file path for model '{model_name}', density '{density_type}'")
        return None
    
    path = Path(file_path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"File not found for model '{model_name}', density '{density_type}': {file_path}")
        return None
    
    if not path.is_file():
        if required:
            raise ValueError(f"Path is not a file for model '{model_name}', density '{density_type}': {file_path}")
        return None
    
    return path


def safe_divide(numerator: Union[float, np.ndarray], denominator: Union[float, np.ndarray], 
               default_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers or arrays, handling division by zero.
    
    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default_value: Value to return when denominator is zero
    
    Returns:
        Division result or default value
    """
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, default_value, dtype=float)
        valid_mask = denominator != 0
        if np.any(valid_mask):
            result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        return result
    else:
        return numerator / denominator if denominator != 0 else default_value


def create_equal_layer_masks(crown_df: pd.DataFrame, layer_boundaries: List[float], 
                           height_column: str = 'z') -> List[np.ndarray]:
    """
    Create boolean masks for crown layers with equal voxel counts.
    
    This function works with boundaries calculated by calculate_crown_layer_bounds_by_count
    to ensure each layer contains approximately equal numbers of voxels.
    
    Args:
        crown_df: DataFrame containing crown voxel data
        layer_boundaries: List of height boundaries from calculate_crown_layer_bounds_by_count
        height_column: Name of height column
    
    Returns:
        List of boolean masks for each layer [lower, middle, upper]
    """
    if len(crown_df) == 0:
        return [np.array([]), np.array([]), np.array([])]
    
    if len(layer_boundaries) < 2:
        # Fallback to simple thirds if boundaries are insufficient
        z_min = crown_df[height_column].min()
        z_max = crown_df[height_column].max()
        layer_height = (z_max - z_min) / 3
        lower_bound = z_min + layer_height
        middle_bound = z_min + 2 * layer_height
    else:
        lower_bound, middle_bound = layer_boundaries[:2]
    
    # Create masks based on calculated boundaries
    lower_mask = crown_df[height_column] <= lower_bound
    middle_mask = (crown_df[height_column] > lower_bound) & (crown_df[height_column] <= middle_bound)
    upper_mask = crown_df[height_column] > middle_bound
    
    return [lower_mask, middle_mask, upper_mask]


def validate_layer_division(crown_df: pd.DataFrame, layer_masks: List[np.ndarray], 
                          density_column: str, min_density: float = 0.05) -> Dict[str, int]:
    """
    Validate that layer division results in reasonably balanced layers.
    
    Args:
        crown_df: DataFrame containing crown voxel data
        layer_masks: List of boolean masks for each layer
        density_column: Name of density column
        min_density: Minimum density threshold
    
    Returns:
        Dictionary with layer statistics
    """
    layer_names = ['lower', 'middle', 'upper']
    layer_stats = {}
    
    total_valid_voxels = len(crown_df[crown_df[density_column] > min_density])
    
    for i, (name, mask) in enumerate(zip(layer_names, layer_masks)):
        if len(mask) > 0 and isinstance(mask, np.ndarray):
            layer_data = crown_df[mask]
            valid_voxels = len(layer_data[layer_data[density_column] > min_density])
            
            layer_stats[name] = {
                'total_voxels': len(layer_data),
                'valid_voxels': valid_voxels,
                'percentage': (valid_voxels / total_valid_voxels * 100) if total_valid_voxels > 0 else 0
            }
        else:
            layer_stats[name] = {
                'total_voxels': 0,
                'valid_voxels': 0,
                'percentage': 0
            }
    
    return layer_stats