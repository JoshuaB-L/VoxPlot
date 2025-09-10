#!/usr/bin/env python3
"""
Enhanced Visualization module for VoxPlot analysis.

This module provides publication-quality visualizations for voxel-based forest structure data,
optimized for Nature journal standards with improved typography, color palettes, and layout.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from datetime import datetime
import warnings

from utils import ensure_directory, calculate_statistics

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class PublicationStyleManager:
    """Manages publication-quality styling and color schemes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        viz_config = config.get('analysis', {}).get('visualization', {})
        
        # Typography settings
        typography = viz_config.get('typography', {})
        self.title_font = typography.get('title_font', 'Helvetica')
        self.body_font = typography.get('body_font', 'Helvetica')
        self.fallback_fonts = typography.get('fallback_fonts', ['Arial', 'DejaVu Sans'])
        
        self.title_fontsize = typography.get('title_fontsize', 24)
        self.subtitle_fontsize = typography.get('subtitle_fontsize', 18)
        self.label_fontsize = typography.get('label_fontsize', 14)
        self.tick_fontsize = typography.get('tick_fontsize', 11)
        self.annotation_fontsize = typography.get('annotation_fontsize', 12)
        
        self.title_weight = typography.get('title_weight', 'bold')
        self.subtitle_weight = typography.get('subtitle_weight', 'semibold')
        self.label_weight = typography.get('label_weight', 'normal')
        
        # Color settings
        colors = viz_config.get('colors', {})
        self.model_palette = colors.get('model_palette', [
            "#4A90E2", "#E74C3C", "#2ECC71", "#9B59B6", 
            "#F39C12", "#1ABC9C", "#34495E", "#E67E22"
        ])
        self.density_colormap = colors.get('density_colormap', 'plasma')
        self.figure_background = colors.get('figure_background', '#FFFFFF')
        self.plot_background = colors.get('plot_background', '#FFFFFF')
        self.grid_color = colors.get('grid_color', '#E5E5E5')
        self.text_color = colors.get('text_color', '#2C3E50')
        
        # Initialize matplotlib settings
        self._setup_matplotlib_style()
    
    def _setup_matplotlib_style(self):
        """Configure matplotlib for publication quality."""
        # Font configuration with fallbacks
        font_list = [self.title_font] + self.fallback_fonts
        plt.rcParams['font.family'] = font_list
        plt.rcParams['font.size'] = self.label_fontsize
        
        # Figure and axes styling
        plt.rcParams['figure.facecolor'] = self.figure_background
        plt.rcParams['axes.facecolor'] = self.plot_background
        plt.rcParams['axes.edgecolor'] = self.text_color
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['axes.labelcolor'] = self.text_color
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        
        # Grid styling
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.color'] = self.grid_color
        plt.rcParams['grid.linewidth'] = 0.8
        plt.rcParams['grid.alpha'] = 0.7
        
        # Text and tick styling
        plt.rcParams['text.color'] = self.text_color
        plt.rcParams['xtick.color'] = self.text_color
        plt.rcParams['ytick.color'] = self.text_color
        plt.rcParams['xtick.labelsize'] = self.tick_fontsize
        plt.rcParams['ytick.labelsize'] = self.tick_fontsize
        
        # Legend styling
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.fancybox'] = True
        plt.rcParams['legend.shadow'] = False
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['legend.edgecolor'] = self.grid_color
    
    def get_model_color(self, model_index: int) -> str:
        """Get consistent color for a model based on index."""
        return self.model_palette[model_index % len(self.model_palette)]
    
    def get_font_properties(self, font_type: str) -> Dict[str, Any]:
        """Get font properties for different text types."""
        properties = {
            'family': [self.body_font] + self.fallback_fonts,
            'color': self.text_color
        }
        
        if font_type == 'title':
            properties.update({
                'size': self.title_fontsize,
                'weight': self.title_weight
            })
        elif font_type == 'subtitle':
            properties.update({
                'size': self.subtitle_fontsize,
                'weight': self.subtitle_weight
            })
        elif font_type == 'label':
            properties.update({
                'size': self.label_fontsize,
                'weight': self.label_weight
            })
        elif font_type == 'annotation':
            properties.update({
                'size': self.annotation_fontsize,
                'weight': self.label_weight
            })
        elif font_type == 'tick':
            properties.update({
                'size': self.tick_fontsize,
                'weight': self.label_weight
            })
        
        return properties


class ForestStructureVisualizer:
    """Creates publication-quality visualizations for forest structure analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize style manager
        self.style_manager = PublicationStyleManager(config)
        
        # Visualization settings
        viz_config = config.get('analysis', {}).get('visualization', {})
        
        # Figure sizes
        self.figsize_large = tuple(viz_config.get('figsize_large', [20, 24]))
        self.figsize_medium = tuple(viz_config.get('figsize_medium', [16, 12]))
        self.figsize_small = tuple(viz_config.get('figsize_small', [12, 9]))
        self.figsize_table = tuple(viz_config.get('figsize_table', [16, 10]))
        
        # DPI and quality settings
        self.dpi = viz_config.get('dpi', 600)
        
        # Layout settings
        layout = viz_config.get('layout', {})
        self.crown_layout = layout.get('crown_layers', {})
        self.three_d_layout = layout.get('three_d', {})
        self.dist_layout = layout.get('distributions', {})
        self.table_layout = layout.get('tables', {})
        
        # Color scale settings
        colors = viz_config.get('colors', {})
        self.color_scale_min = colors.get('color_scale_min', 0.0)
        self.color_scale_max = colors.get('color_scale_max', 40.0)
        
        # Output mode
        self.plot_mode = viz_config.get('plot_mode', 'combined_plots')
        
        self.logger.info("Publication-quality visualizer initialized")
    
    def create_crown_layer_comparison(self, comparison_results: Dict[str, Any], 
                                    density_type: str) -> plt.Figure:
        """Create enhanced crown layer comparison with fixed text positioning."""
        model_names = comparison_results['model_names']
        layer_analyses = comparison_results['layer_analyses']
        
        if not model_names or not layer_analyses:
            return self._create_empty_figure("No data available for visualization")
        
        n_models = len(model_names)
        n_models = min(n_models, len(layer_analyses))
        
        if n_models == 0:
            return self._create_empty_figure("No valid model data available")
        
        # Create figure with proper spacing
        fig = plt.figure(figsize=self.figsize_large, facecolor=self.style_manager.figure_background)
        
        # Improved grid spacing to prevent overlap
        height_ratios = [1, 1, 1, 1, 0.8]
        width_ratios = [1] * n_models
        
        gs = gridspec.GridSpec(
            5, n_models, 
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=self.crown_layout.get('subplot_spacing_vertical', 0.35),
            wspace=self.crown_layout.get('subplot_spacing_horizontal', 0.20),
            top=0.92,  # Leave space for title
            bottom=0.08,  # Leave space for labels
            left=0.08,  # Leave space for row labels
            right=0.90   # Leave space for colorbar
        )
        
        # Set main title with proper positioning
        density_label = self._get_density_label(density_type)
        title_props = self.style_manager.get_font_properties('title')
        fig.suptitle(f'Crown Layer Comparison - {density_label}', 
                    fontsize=title_props['size'], 
                    fontweight=title_props['weight'],
                    fontfamily=title_props['family'],
                    color=title_props['color'],
                    y=0.96)
        
        # Row information with improved spacing
        row_info = [
            (0, 'Whole\nCrown', 'whole'),
            (1, 'Upper\nCrown', 'upper'), 
            (2, 'Middle\nCrown', 'middle'),
            (3, 'Lower\nCrown', 'lower'),
            (4, 'Vertical Slice\n(ZX)', 'zx_slice')
        ]
        
        # Determine color scale
        vmin, vmax = self._determine_color_scale(layer_analyses, density_type)
        
        # Create subplots with improved text positioning
        for row_idx, row_label, layer_key in row_info:
            # Add row label with better positioning
            row_props = self.style_manager.get_font_properties('subtitle')
            fig.text(0.02, 0.8 - row_idx * 0.16, row_label, 
                    ha='center', va='center', 
                    fontsize=row_props['size'], 
                    fontweight=row_props['weight'],
                    fontfamily=row_props['family'],
                    color=row_props['color'],
                    rotation=90)
            
            for col_idx in range(n_models):
                if col_idx >= len(layer_analyses):
                    continue
                    
                analysis = layer_analyses[col_idx]
                model_name = model_names[col_idx] if col_idx < len(model_names) else f"Model_{col_idx+1}"
                
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.set_facecolor(self.style_manager.plot_background)
                
                # Add model name title for top row only with better positioning
                if row_idx == 0:
                    title_props = self.style_manager.get_font_properties('subtitle')
                    ax.text(0.5, 1.08, model_name, ha='center', va='bottom',
                           transform=ax.transAxes, 
                           fontsize=title_props['size'], 
                           fontweight=title_props['weight'],
                           fontfamily=title_props['family'],
                           color=title_props['color'])
                
                if layer_key == 'zx_slice':
                    self._plot_vertical_slice_enhanced(ax, analysis, density_type)
                else:
                    self._plot_crown_layer_raster_enhanced(ax, analysis, layer_key, density_type, vmin, vmax)
                
                # Set axis labels with proper font
                label_props = self.style_manager.get_font_properties('label')
                ax.set_xlabel('X (m)', fontdict=label_props)
                ax.set_ylabel('Y (m)' if layer_key != 'zx_slice' else 'Z (m)', fontdict=label_props)
                
                # Style ticks
                tick_props = self.style_manager.get_font_properties('tick')
                ax.tick_params(axis='both', which='major', 
                             labelsize=tick_props['size'],
                             colors=tick_props['color'])
        
        # Add enhanced colorbar
        self._add_enhanced_colorbar(fig, vmin, vmax, density_label)
        
        return fig
    
    def _plot_crown_layer_raster_enhanced(self, ax, analysis, layer_key, density_type, vmin, vmax):
        """Enhanced crown layer raster plotting with fixed text positioning."""
        raster_data = analysis['rasters'].get(layer_key)
        metrics = analysis['metrics'].get(layer_key, {})
        
        if raster_data and raster_data['array'].size > 0:
            # Plot raster with enhanced colormap
            im = ax.imshow(raster_data['array'], origin='lower', 
                         cmap=self.style_manager.density_colormap,
                         vmin=vmin, vmax=vmax, extent=raster_data['extent'])
            
            # Add metrics text with better positioning and styling
            total_area = metrics.get('total_area', 0)
            mean_ai = metrics.get('mean_area_index', 0)
            
            if density_type.lower() == 'lad':
                metrics_text = f"LA: {total_area:.1f} m²\nLAI: {mean_ai:.2f}"
            elif density_type.lower() == 'wad':
                metrics_text = f"WA: {total_area:.1f} m²\nWAI: {mean_ai:.2f}"
            elif density_type.lower() == 'pad':
                metrics_text = f"PA: {total_area:.1f} m²\nPAI: {mean_ai:.2f}"
            else:
                metrics_text = f"Area: {total_area:.1f} m²\nAI: {mean_ai:.2f}"
            
            # Enhanced text box styling
            annotation_props = self.style_manager.get_font_properties('annotation')
            bbox_props = dict(
                boxstyle="round,pad=0.4", 
                facecolor='white', 
                edgecolor=self.style_manager.grid_color,
                alpha=0.95,
                linewidth=1.2
            )
            
            ax.text(0.5, 1.02, metrics_text, ha='center', va='bottom',
                   transform=ax.transAxes, 
                   fontsize=annotation_props['size'],
                   fontweight=annotation_props['weight'],
                   fontfamily=annotation_props['family'],
                   color=annotation_props['color'],
                   bbox=bbox_props)
        else:
            self._plot_no_data_message(ax)
    
    def _plot_vertical_slice_enhanced(self, ax, analysis, density_type):
        """Enhanced vertical slice plotting."""
        zx_slice = analysis.get('zx_slice')
        if zx_slice and zx_slice['array'].size > 0:
            # Dynamic color scale for vertical slices
            slice_data = zx_slice['array'][zx_slice['array'] > 0]
            if len(slice_data) > 0:
                slice_vmax = np.percentile(slice_data, 95)
                slice_vmax = max(slice_vmax, 0.1)
            else:
                slice_vmax = 1.0
            
            im = ax.imshow(zx_slice['array'], origin='lower', 
                         cmap=self.style_manager.density_colormap,
                         vmin=0, vmax=slice_vmax, extent=zx_slice['extent'])
            
            # Enhanced boundary lines
            boundaries = analysis.get('layer_boundaries', [])
            boundary_color = self.crown_layout.get('boundary_line_color', '#E74C3C')
            boundary_style = self.crown_layout.get('boundary_line_style', '--')
            boundary_width = self.crown_layout.get('boundary_line_width', 2.0)
            boundary_alpha = self.crown_layout.get('boundary_line_alpha', 0.8)
            
            for boundary in boundaries:
                ax.axhline(y=boundary, color=boundary_color, 
                          linestyle=boundary_style, alpha=boundary_alpha, 
                          linewidth=boundary_width)
        else:
            self._plot_no_data_message(ax)
    
    def _plot_no_data_message(self, ax):
        """Plot consistent 'no data' message."""
        annotation_props = self.style_manager.get_font_properties('annotation')
        bbox_props = dict(
            boxstyle="round,pad=0.5", 
            facecolor='#F8F9FA', 
            edgecolor=self.style_manager.grid_color,
            alpha=0.8
        )
        
        ax.text(0.5, 0.5, 'No data\navailable', ha='center', va='center',
               transform=ax.transAxes, 
               fontsize=annotation_props['size'],
               fontweight=annotation_props['weight'],
               fontfamily=annotation_props['family'],
               color=annotation_props['color'],
               bbox=bbox_props)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set white background instead of default color-filled background
        ax.set_facecolor('white')
    
    def _add_enhanced_colorbar(self, fig, vmin, vmax, density_label):
        """Add enhanced colorbar with proper positioning."""
        colorbar_pos = self.crown_layout.get('colorbar_position', [0.94, 0.15, 0.015, 0.7])
        cbar_ax = fig.add_axes(colorbar_pos)
        
        sm = plt.cm.ScalarMappable(
            cmap=self.style_manager.density_colormap, 
            norm=Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=cbar_ax)
        
        # Style colorbar
        label_props = self.style_manager.get_font_properties('label')
        cbar.set_label(f'{density_label} (m²/m²)', 
                      fontdict=label_props,
                      labelpad=self.crown_layout.get('colorbar_label_pad', 20))
        
        tick_props = self.style_manager.get_font_properties('tick')
        cbar.ax.tick_params(labelsize=tick_props['size'], 
                           colors=tick_props['color'])
    
    def create_enhanced_3d_visualization(self, datasets: List[pd.DataFrame], 
                                       density_type: str,
                                       crown_base_height: float = 0.7,
                                       density_threshold: float = 0.1) -> plt.Figure:
        """Create publication-quality 3D voxel visualization."""
        if not datasets:
            return self._create_empty_figure("No datasets available for 3D visualization")
        
        n_models = len(datasets)
        
        # Calculate grid layout
        if n_models == 1:
            grid_rows, grid_cols = 1, 1
            fig_width, fig_height = 12, 10
        elif n_models == 2:
            grid_rows, grid_cols = 1, 2
            fig_width, fig_height = 20, 10
        elif n_models <= 4:
            grid_rows, grid_cols = 2, 2
            fig_width, fig_height = 20, 16
        else:
            grid_rows = int(np.ceil(np.sqrt(n_models)))
            grid_cols = int(np.ceil(n_models / grid_rows))
            fig_width, fig_height = grid_cols * 8, grid_rows * 8
        
        fig = plt.figure(figsize=(fig_width, fig_height), 
                        facecolor=self.style_manager.figure_background)
        
        # Enhanced title
        density_label = self._get_density_label(density_type)
        title_props = self.style_manager.get_font_properties('title')
        fig.suptitle(f'3D Voxel Structure - {density_label}', 
                    fontsize=title_props['size'], 
                    fontweight=title_props['weight'],
                    fontfamily=title_props['family'],
                    color=title_props['color'],
                    y=0.95)
        
        # Color normalization
        all_density_values = []
        for df in datasets:
            crown_df = df[(df['z'] >= crown_base_height) & (df[density_type] > density_threshold)]
            if len(crown_df) > 0:
                all_density_values.extend(crown_df[density_type].values)
        
        if all_density_values:
            norm = Normalize(vmin=0, vmax=np.percentile(all_density_values, 95))
        else:
            norm = Normalize(vmin=0, vmax=1)
        
        cmap = plt.cm.get_cmap(self.style_manager.density_colormap)
        
        for i, df in enumerate(datasets):
            if i >= grid_rows * grid_cols:
                break
                
            model_name = df['display_name'].iloc[0]
            
            # Filter and sample data
            crown_df = df[(df['z'] >= crown_base_height) & (df[density_type] > density_threshold)]
            
            if len(crown_df) == 0:
                continue
            
            # Sample for performance
            max_voxels = self.three_d_layout.get('max_voxels_per_model', 8000)
            if len(crown_df) > max_voxels:
                crown_df = crown_df.sample(max_voxels, random_state=42)
            
            # Create 3D subplot
            ax = fig.add_subplot(grid_rows, grid_cols, i + 1, projection='3d')
            ax.set_facecolor(self.style_manager.plot_background)
            
            # Enhanced 3D cube plotting
            self._plot_enhanced_3d_cubes(ax, crown_df, density_type, cmap, norm)
            
            # Style the 3D plot
            title_props = self.style_manager.get_font_properties('subtitle')
            ax.set_title(model_name, 
                        fontsize=title_props['size'], 
                        fontweight=title_props['weight'],
                        fontfamily=title_props['family'],
                        color=title_props['color'],
                        pad=20)
            
            # Set axis labels
            label_props = self.style_manager.get_font_properties('label')
            ax.set_xlabel('X (m)', fontdict=label_props)
            ax.set_ylabel('Y (m)', fontdict=label_props)
            ax.set_zlabel('Z (m)', fontdict=label_props)
            
            # Set viewing angle
            elev = self.three_d_layout.get('viewing_angle_elevation', 20)
            azim = self.three_d_layout.get('viewing_angle_azimuth', 45)
            ax.view_init(elev=elev, azim=azim)
            
            # Equal aspect ratio
            self._set_equal_3d_aspect(ax, crown_df)
            
            # Style grid and ticks
            ax.grid(True, alpha=0.3)
            tick_props = self.style_manager.get_font_properties('tick')
            ax.tick_params(axis='both', which='major', labelsize=tick_props['size'])
        
        # Add colorbar with proper positioning to avoid overlap
        if n_models > 0:
            # Position colorbar to avoid overlap
            if grid_cols == 1:
                cbar_pos = [0.92, 0.15, 0.02, 0.7]
            else:
                cbar_pos = [0.92, 0.15, 0.015, 0.7]
            
            cbar_ax = fig.add_axes(cbar_pos)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            
            label_props = self.style_manager.get_font_properties('label')
            cbar.set_label(f'{density_label} Density (m²/m³)', fontdict=label_props)
            
            tick_props = self.style_manager.get_font_properties('tick')
            cbar.ax.tick_params(labelsize=tick_props['size'])
        
        plt.tight_layout()
        return fig
    
    def _plot_enhanced_3d_cubes(self, ax, crown_df, density_type, cmap, norm):
        """Plot enhanced 3D cubes with better visibility."""
        voxel_size = 0.25  # Could be made configurable
        cube_size_factor = self.three_d_layout.get('cube_size_factor', 0.9)
        cube_alpha = self.three_d_layout.get('cube_alpha', 0.7)
        edge_width = self.three_d_layout.get('cube_edge_width', 0.3)
        edge_color = self.three_d_layout.get('cube_edge_color', '#34495E')
        
        cube_size = voxel_size * cube_size_factor
        half_size = cube_size / 2
        
        # Plot cubes for each voxel
        for _, row in crown_df.iterrows():
            x, y, z = row['x'], row['y'], row['z']
            density = row[density_type]
            
            # Cube vertices
            vertices = np.array([
                [x - half_size, y - half_size, z - half_size],
                [x + half_size, y - half_size, z - half_size],
                [x + half_size, y + half_size, z - half_size],
                [x - half_size, y + half_size, z - half_size],
                [x - half_size, y - half_size, z + half_size],
                [x + half_size, y - half_size, z + half_size],
                [x + half_size, y + half_size, z + half_size],
                [x - half_size, y + half_size, z + half_size]
            ])
            
            # Cube faces
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            
            color = cmap(norm(density))
            
            # Add cube
            cube_collection = Poly3DCollection(faces, facecolors=color, alpha=cube_alpha,
                                             linewidths=edge_width, edgecolors=edge_color)
            ax.add_collection3d(cube_collection)
    
    def create_enhanced_distribution_comparison(self, datasets: List[pd.DataFrame], 
                                              density_type: str) -> plt.Figure:
        """Create publication-quality distribution comparison with fixed text overlap."""
        if not datasets:
            return self._create_empty_figure("No datasets available for distribution comparison")
        
        # Create figure with better proportions
        fig = plt.figure(figsize=self.figsize_large, facecolor=self.style_manager.figure_background)
        
        # Improved grid layout to prevent overlap
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1.2, 0.8], width_ratios=[1, 1, 1],
                              hspace=0.35, wspace=0.30, top=0.92, bottom=0.08, left=0.08, right=0.95)
        
        density_label = self._get_density_label(density_type)
        title_props = self.style_manager.get_font_properties('title')
        fig.suptitle(f'{density_label} Distribution Analysis', 
                    fontsize=title_props['size'], 
                    fontweight=title_props['weight'],
                    fontfamily=title_props['family'],
                    color=title_props['color'],
                    y=0.96)
        
        # Process data with outlier removal
        processed_data = self._process_distribution_data(datasets, density_type)
        
        if not processed_data['all_density_data']:
            return self._create_empty_figure("No valid density data for distribution comparison")
        
        # Get consistent colors for models
        model_colors = [self.style_manager.get_model_color(i) for i in range(len(processed_data['model_names']))]
        
        # 1. Histogram with KDE (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_distribution_histogram(ax1, processed_data, density_label, model_colors)
        
        # 2. Cumulative distribution (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_cumulative_distribution(ax2, processed_data, density_label, model_colors)
        
        # 3. Box plot (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_distribution_boxplot(ax3, processed_data, density_label, model_colors)
        
        # 4. Violin plot (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_distribution_violin(ax4, processed_data, density_label, model_colors)
        
        # 5. Density ridgeline plot (middle center)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_density_ridgeline(ax5, processed_data, density_label, model_colors)
        
        # 6. Quantile comparison (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_quantile_comparison(ax6, processed_data, density_label, model_colors)
        
        # 7. Statistics table (bottom, spanning all columns)
        ax7 = fig.add_subplot(gs[2, :])
        self._create_enhanced_statistics_table(ax7, processed_data, density_label)
        
        return fig
    
    def _process_distribution_data(self, datasets, density_type):
        """Process distribution data with outlier removal."""
        all_density_data = []
        model_names = []
        
        remove_outliers = self.dist_layout.get('remove_outliers', True)
        lower_percentile = self.dist_layout.get('outlier_percentile_lower', 2.0)
        upper_percentile = self.dist_layout.get('outlier_percentile_upper', 98.0)
        
        for df in datasets:
            model_name = df['display_name'].iloc[0]
            nonzero_density = df[df[density_type] > 0][density_type]
            
            if len(nonzero_density) > 0:
                if remove_outliers:
                    lower_bound = np.percentile(nonzero_density, lower_percentile)
                    upper_bound = np.percentile(nonzero_density, upper_percentile)
                    clean_density = nonzero_density[
                        (nonzero_density >= lower_bound) & (nonzero_density <= upper_bound)
                    ]
                else:
                    clean_density = nonzero_density
                    
                if len(clean_density) > 0:
                    all_density_data.append(clean_density)
                    model_names.append(model_name)
        
        return {
            'all_density_data': all_density_data,
            'model_names': model_names
        }
    
    def _plot_distribution_histogram(self, ax, processed_data, density_label, model_colors):
        """Plot enhanced histogram with KDE overlay."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        bins = self.dist_layout.get('histogram_bins', 35)
        alpha = self.style_manager.config.get('analysis', {}).get('visualization', {}).get('colors', {}).get('fill_alpha', 0.6)
        
        for i, (data, name, color) in enumerate(zip(all_density_data, model_names, model_colors)):
            # Plot histogram
            ax.hist(data, bins=bins, alpha=alpha, color=color, label=name, 
                   density=True, edgecolor='white', linewidth=0.5)
            
            # Add KDE overlay
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_range, kde(x_range), color=color, linewidth=2, alpha=0.8)
            except ImportError:
                pass  # Skip KDE if scipy not available
        
        # Style the plot
        label_props = self.style_manager.get_font_properties('label')
        ax.set_xlabel(f'{density_label} Density (m²/m³)', fontdict=label_props)
        ax.set_ylabel('Probability Density', fontdict=label_props)
        
        title_props = self.style_manager.get_font_properties('subtitle')
        ax.set_title('Distribution Comparison', fontdict=title_props)
        
        ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    def _plot_cumulative_distribution(self, ax, processed_data, density_label, model_colors):
        """Plot cumulative distribution function."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        for i, (data, name, color) in enumerate(zip(all_density_data, model_names, model_colors)):
            sorted_data = np.sort(data)
            y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, y_values, color=color, linewidth=2.5, label=name, alpha=0.8)
        
        # Style the plot
        label_props = self.style_manager.get_font_properties('label')
        ax.set_xlabel(f'{density_label} Density (m²/m³)', fontdict=label_props)
        ax.set_ylabel('Cumulative Probability', fontdict=label_props)
        
        title_props = self.style_manager.get_font_properties('subtitle')
        ax.set_title('Cumulative Distribution', fontdict=title_props)
        
        ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_distribution_boxplot(self, ax, processed_data, density_label, model_colors):
        """Plot enhanced box plot comparison."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        # Create box plot
        bp = ax.boxplot(all_density_data, labels=model_names, patch_artist=True,
                       notch=self.dist_layout.get('box_notch', True),
                       whis=self.dist_layout.get('whisker_length', 1.5),
                       showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.6))
        
        # Apply colors
        for patch, color in zip(bp['boxes'], model_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('#2C3E50')
            patch.set_linewidth(1.2)
        
        # Style whiskers and medians
        for whisker in bp['whiskers']:
            whisker.set_color('#2C3E50')
            whisker.set_linewidth(1.5)
        
        for median in bp['medians']:
            median.set_color('#2C3E50')
            median.set_linewidth(2)
        
        # Style the plot
        label_props = self.style_manager.get_font_properties('label')
        ax.set_ylabel(f'{density_label} Density (m²/m³)', fontdict=label_props)
        
        title_props = self.style_manager.get_font_properties('subtitle')
        ax.set_title('Box Plot Comparison', fontdict=title_props)
        
        # Rotate x-axis labels if needed
        tick_props = self.style_manager.get_font_properties('tick')
        ax.tick_params(axis='x', rotation=45, labelsize=tick_props['size'])
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_distribution_violin(self, ax, processed_data, density_label, model_colors):
        """Plot violin plot comparison."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        # Create violin plot
        parts = ax.violinplot([data for data in all_density_data], 
                             positions=range(len(model_names)),
                             showmeans=True, showmedians=True)
        
        # Apply colors
        for i, (pc, color) in enumerate(zip(parts['bodies'], model_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('#2C3E50')
            pc.set_linewidth(1.2)
        
        # Style other violin elements
        if 'cmeans' in parts:
            parts['cmeans'].set_color('#2C3E50')
            parts['cmeans'].set_linewidth(2)
        if 'cmedians' in parts:
            parts['cmedians'].set_color('#E74C3C')
            parts['cmedians'].set_linewidth(2)
        
        # Set labels and styling
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names)
        
        label_props = self.style_manager.get_font_properties('label')
        ax.set_ylabel(f'{density_label} Density (m²/m³)', fontdict=label_props)
        
        title_props = self.style_manager.get_font_properties('subtitle')
        ax.set_title('Violin Plot Comparison', fontdict=title_props)
        
        tick_props = self.style_manager.get_font_properties('tick')
        ax.tick_params(axis='x', rotation=45, labelsize=tick_props['size'])
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_density_ridgeline(self, ax, processed_data, density_label, model_colors):
        """Plot density ridgeline plot."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        n_models = len(model_names)
        
        try:
            from scipy.stats import gaussian_kde
            
            for i, (data, name, color) in enumerate(zip(all_density_data, model_names, model_colors)):
                # Calculate KDE
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                y_density = kde(x_range)
                
                # Normalize and offset for ridgeline effect
                y_density_norm = y_density / y_density.max() * 0.8  # Scale height
                y_offset = (n_models - i - 1) * 1.2  # Vertical offset
                
                # Plot filled density curve
                ax.fill_between(x_range, y_offset, y_offset + y_density_norm, 
                               alpha=0.7, color=color, label=name)
                ax.plot(x_range, y_offset + y_density_norm, color=color, linewidth=2)
                
                # Add model name
                ax.text(x_range[0], y_offset + 0.4, name, 
                       fontsize=self.style_manager.tick_fontsize,
                       verticalalignment='center')
        
        except ImportError:
            # Fallback to simple histogram overlay if scipy not available
            for i, (data, name, color) in enumerate(zip(all_density_data, model_names, model_colors)):
                counts, bins = np.histogram(data, bins=30, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                y_offset = (n_models - i - 1) * 1.2
                
                ax.fill_between(bin_centers, y_offset, y_offset + counts/counts.max()*0.8,
                               alpha=0.7, color=color, label=name, step='mid')
        
        # Style the plot
        label_props = self.style_manager.get_font_properties('label')
        ax.set_xlabel(f'{density_label} Density (m²/m³)', fontdict=label_props)
        ax.set_ylabel('Models', fontdict=label_props)
        
        title_props = self.style_manager.get_font_properties('subtitle')
        ax.set_title('Density Ridgeline', fontdict=title_props)
        
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')

    def _plot_quantile_comparison(self, ax, processed_data, density_label, model_colors):
        """Plot quantile-quantile comparison."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        if len(all_density_data) >= 2:
            # Use first dataset as reference
            reference_data = np.sort(all_density_data[0])
            ref_quantiles = np.linspace(0, 1, len(reference_data))
            
            for i, (data, name, color) in enumerate(zip(all_density_data[1:], model_names[1:], model_colors[1:])):
                sorted_data = np.sort(data)
                # Interpolate to common quantiles
                common_quantiles = np.linspace(0, 1, min(len(reference_data), len(sorted_data)))
                ref_interp = np.interp(common_quantiles, ref_quantiles, reference_data)
                data_interp = np.interp(common_quantiles, np.linspace(0, 1, len(sorted_data)), sorted_data)
                
                ax.scatter(ref_interp, data_interp, alpha=0.6, color=color, 
                          label=f'{name} vs {model_names[0]}', s=20)
            
            # Add diagonal reference line
            max_val = max([data.max() for data in all_density_data])
            min_val = min([data.min() for data in all_density_data])
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
            
            # Style the plot
            label_props = self.style_manager.get_font_properties('label')
            ax.set_xlabel(f'{model_names[0]} Quantiles', fontdict=label_props)
            ax.set_ylabel('Other Model Quantiles', fontdict=label_props)
            
            title_props = self.style_manager.get_font_properties('subtitle')
            ax.set_title('Q-Q Plot Comparison', fontdict=title_props)
            
            ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        else:
            # Not enough data for Q-Q plot
            ax.text(0.5, 0.5, 'Insufficient data\nfor Q-Q plot', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Q-Q Plot Comparison')

    def _create_enhanced_statistics_table(self, ax, processed_data, density_label):
        """Create enhanced statistics table."""
        all_density_data = processed_data['all_density_data']
        model_names = processed_data['model_names']
        
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate statistics
        stats_data = []
        for data, name in zip(all_density_data, model_names):
            stats = calculate_statistics(data)
            stats_data.append([
                name,
                f"{stats['count']:,}",
                f"{stats['mean']:.3f}",
                f"{stats['median']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['q25']:.3f}",
                f"{stats['q75']:.3f}"
            ])
        
        headers = ['Model', 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75']
        
        # Create table
        table = ax.table(
            cellText=stats_data,
            colLabels=headers,
            loc='center',
            cellLoc='center'
        )
        
        # Apply enhanced styling
        table.auto_set_font_size(False)
        
        # Get table styling configuration
        table_config = self.table_layout
        cell_fontsize = table_config.get('cell_fontsize', self.style_manager.tick_fontsize)
        header_fontsize = table_config.get('header_fontsize', self.style_manager.label_fontsize)
        
        table.set_fontsize(cell_fontsize)
        table.scale(1.2, table_config.get('row_height', 1.8))
        
        # Style header row
        header_color = table_config.get('header_color', '#34495E')
        header_text_color = table_config.get('header_text_color', '#FFFFFF')
        
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(header_color)
            cell.set_text_props(weight='bold', color=header_text_color, fontsize=header_fontsize)
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
        
        # Style data rows with alternating colors
        alternating_colors = table_config.get('alternating_colors', ['#F8F9FA', '#FFFFFF'])
        border_color = table_config.get('border_color', '#BDC3C7')
        
        for i in range(1, len(stats_data) + 1):
            row_color = alternating_colors[(i-1) % len(alternating_colors)]
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor(row_color)
                cell.set_edgecolor(border_color)
                cell.set_linewidth(1)
                
                # Bold model names
                if j == 0:
                    cell.set_text_props(weight='bold', fontsize=cell_fontsize)
        
        # Set title
        title_props = self.style_manager.get_font_properties('subtitle')
        ax.set_title(f'{density_label} Statistics Summary', 
                    fontdict=title_props, pad=20)

    def create_enhanced_table_visualization(self, comparison_results: Dict[str, Any], 
                                          density_type: str) -> plt.Figure:
        """Create enhanced table visualization with export functionality."""
        model_names = comparison_results['model_names']
        layer_metrics = comparison_results['layer_metrics']
        
        if not model_names or not layer_metrics:
            return self._create_empty_figure("No metrics data available for table")
        
        fig = plt.figure(figsize=self.figsize_table, 
                        facecolor=self.style_manager.figure_background)
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        # Create enhanced table data
        table_data = self._prepare_enhanced_table_data(layer_metrics, model_names, density_type)
        
        # Create the table with enhanced styling
        table = ax.table(
            cellText=table_data['data'],
            colLabels=table_data['headers'],
            loc='center',
            cellLoc='center'
        )
        
        # Apply enhanced table styling
        self._style_enhanced_table(table, len(model_names))
        
        # Set title
        density_label = self._get_density_label(density_type)
        title_props = self.style_manager.get_font_properties('title')
        fig.suptitle(f'Crown Layer Metrics Summary - {density_label}', 
                    fontsize=title_props['size'], 
                    fontweight=title_props['weight'],
                    fontfamily=title_props['family'],
                    color=title_props['color'],
                    y=0.95)
        
        plt.tight_layout()
        return fig
    
    def _prepare_enhanced_table_data(self, layer_metrics, model_names, density_type):
        """Prepare enhanced table data for export and visualization."""
        table_data = []
        layer_names = ['Whole', 'Upper', 'Middle', 'Lower']
        layer_keys = ['whole', 'upper', 'middle', 'lower']
        
        # Create headers
        headers = ['Layer', 'Metric'] + model_names
        
        # Determine metric names based on density type
        if density_type.lower() == 'lad':
            area_name = 'Leaf Area'
            index_name = 'LAI'
        elif density_type.lower() == 'wad':
            area_name = 'Wood Area'
            index_name = 'WAI'
        elif density_type.lower() == 'pad':
            area_name = 'Plant Area'
            index_name = 'PAI'
        else:
            area_name = f'{density_type.upper()} Area'
            index_name = f'{density_type.upper()}I'
        
        # Add data rows
        for layer_name, layer_key in zip(layer_names, layer_keys):
            # Area Index row
            ai_row = [layer_name, f'{index_name} (m²/m²)']
            for metrics in layer_metrics:
                ai_value = metrics[layer_key].get('mean_area_index', 0)
                ai_row.append(f"{ai_value:.3f}")
            table_data.append(ai_row)
            
            # Total Area row
            area_row = ['', f'{area_name} (m²)']
            for metrics in layer_metrics:
                area_value = metrics[layer_key].get('total_area', 0)
                area_row.append(f"{area_value:.1f}")
            table_data.append(area_row)
            
            # Voxel Count row
            count_row = ['', 'Voxel Count']
            for metrics in layer_metrics:
                count_value = metrics[layer_key].get('voxel_count', 0)
                count_row.append(f"{count_value:,}")
            table_data.append(count_row)
            
            # Add separator row (empty) between layers except for the last
            if layer_name != layer_names[-1]:
                separator_row = [''] * len(headers)
                table_data.append(separator_row)
        
        return {
            'headers': headers,
            'data': table_data
        }

    def _style_enhanced_table(self, table, n_models):
        """Apply enhanced styling to table visualization."""
        table.auto_set_font_size(False)
        
        # Get styling configuration
        table_config = self.table_layout
        cell_fontsize = table_config.get('cell_fontsize', 11)
        header_fontsize = table_config.get('header_fontsize', 13)
        cell_padding = table_config.get('cell_padding', 0.08)
        row_height = table_config.get('row_height', 1.8)
        
        table.set_fontsize(cell_fontsize)
        table.scale(1 + cell_padding, row_height)
        
        # Colors
        header_color = table_config.get('header_color', '#34495E')
        header_text_color = table_config.get('header_text_color', '#FFFFFF')
        alternating_colors = table_config.get('alternating_colors', ['#F8F9FA', '#FFFFFF'])
        border_color = table_config.get('border_color', '#BDC3C7')
        border_width = table_config.get('border_width', 1.5)
        
        # Get table dimensions
        n_cols = 2 + n_models  # Layer, Metric, + model columns
        n_rows = len(table._cells) // n_cols
        
        # Style header row
        for j in range(n_cols):
            if (0, j) in table._cells:
                cell = table[(0, j)]
                cell.set_facecolor(header_color)
                cell.set_text_props(weight='bold', color=header_text_color, 
                                   fontsize=header_fontsize)
                cell.set_edgecolor('white')
                cell.set_linewidth(border_width)
        
        # Style data rows
        for i in range(1, n_rows):
            if (i, 0) in table._cells:
                # Determine row color
                row_color = alternating_colors[(i-1) % len(alternating_colors)]
                
                for j in range(n_cols):
                    if (i, j) in table._cells:
                        cell = table[(i, j)]
                        cell.set_facecolor(row_color)
                        cell.set_edgecolor(border_color)
                        cell.set_linewidth(1)
                        
                        # Bold layer names and metric names
                        if j in [0, 1]:
                            cell.set_text_props(weight='bold', fontsize=cell_fontsize)
                        
                        # Highlight metric values
                        if j > 1:
                            cell.set_text_props(fontsize=cell_fontsize)

    def export_table_data(self, comparison_results: Dict[str, Any], 
                         density_type: str, output_dir: Path) -> Dict[str, Path]:
        """Export table data in multiple formats."""
        model_names = comparison_results['model_names']
        layer_metrics = comparison_results['layer_metrics']
        
        if not model_names or not layer_metrics:
            return {}
        
        # Prepare table data
        table_data = self._prepare_enhanced_table_data(layer_metrics, model_names, density_type)
        
        # Create DataFrame
        df = pd.DataFrame(table_data['data'], columns=table_data['headers'])
        
        # Export files
        export_config = self.config.get('analysis', {}).get('visualization', {}).get('export', {}).get('table_export', {})
        formats = export_config.get('formats', ['csv', 'xlsx', 'txt'])
        
        exported_files = {}
        
        for fmt in formats:
            try:
                if fmt == 'csv':
                    filepath = output_dir / f"crown_metrics_{density_type}.csv"
                    separator = export_config.get('csv_separator', ',')
                    df.to_csv(filepath, index=False, sep=separator)
                    exported_files['csv'] = filepath
                    
                elif fmt == 'xlsx':
                    filepath = output_dir / f"crown_metrics_{density_type}.xlsx"
                    df.to_excel(filepath, index=False, sheet_name=f'{density_type.upper()}_Metrics')
                    exported_files['xlsx'] = filepath
                    
                elif fmt == 'txt':
                    filepath = output_dir / f"crown_metrics_{density_type}.txt"
                    delimiter = export_config.get('txt_delimiter', '\t')
                    df.to_csv(filepath, index=False, sep=delimiter)
                    exported_files['txt'] = filepath
                    
                self.logger.info(f"Exported table data to: {filepath}")
                
            except Exception as e:
                self.logger.error(f"Failed to export table in {fmt} format: {e}")
        
        return exported_files
    
    def create_vertical_profile_comparison(self, profile_data: Dict[str, Any],
                                         density_type: str) -> plt.Figure:
        """Create vertical profile comparison plot."""
        model_names = profile_data['model_names']
        profiles = profile_data['profiles']
        
        if not model_names or not profiles:
            return self._create_empty_figure("No profile data available")
        
        fig, ax = plt.subplots(figsize=self.figsize_small, facecolor=self.style_manager.figure_background)
        ax.set_facecolor(self.style_manager.plot_background)
        
        # Get consistent colors for models
        colors = [self.style_manager.get_model_color(i) for i in range(len(model_names))]
        line_styles = ['-', '--', '-.', ':', '-', '--']
        
        max_height = 0
        
        for i, (model_name, profile) in enumerate(zip(model_names, profiles)):
            heights = profile.get('heights', np.array([]))
            densities = profile.get('densities', np.array([]))
            
            if len(heights) > 0 and len(densities) > 0:
                color = colors[i % len(colors)]
                linestyle = line_styles[i % len(line_styles)]
                
                ax.plot(densities, heights, label=model_name, color=color,
                       linestyle=linestyle, linewidth=2, marker='o', markersize=3)
                
                max_height = max(max_height, heights.max())
        
        density_label = self._get_density_label(density_type)
        
        label_props = self.style_manager.get_font_properties('label')
        ax.set_xlabel(f'{density_label} Density (m²/m³)', fontdict=label_props)
        ax.set_ylabel('Height (m)', fontdict=label_props)
        
        title_props = self.style_manager.get_font_properties('title')
        ax.set_title(f'Vertical {density_label} Profile Comparison', fontdict=title_props)
        
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9)
        
        if max_height > 0:
            ax.set_ylim(0, max_height)
        
        plt.tight_layout()
        return fig

    def save_figure_with_options(self, fig: plt.Figure, output_dir: Path, 
                               base_filename: str, analysis_type: str = "general") -> List[Path]:
        """Save figure with multiple format options and single plot mode support."""
        export_config = self.config.get('analysis', {}).get('visualization', {}).get('export', {})
        
        formats = export_config.get('formats', ['png'])
        saved_files = []
        
        # Determine output directory based on plot mode
        if self.plot_mode == "single_plots":
            single_plot_config = export_config.get('single_plots', {})
            if single_plot_config.get('create_subfolders', True):
                subfolder_structure = single_plot_config.get('subfolder_structure', {})
                subfolder_name = subfolder_structure.get(analysis_type, analysis_type)
                final_output_dir = ensure_directory(output_dir / subfolder_name)
            else:
                final_output_dir = output_dir
        else:
            final_output_dir = output_dir
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for fmt in formats:
            try:
                # Create filename with timestamp if in single plots mode
                if self.plot_mode == "single_plots":
                    filename = f"{base_filename}_{timestamp}.{fmt}"
                else:
                    filename = f"{base_filename}.{fmt}"
                
                filepath = final_output_dir / filename
                
                # Get format-specific DPI
                if fmt == 'png':
                    dpi = export_config.get('png_dpi', 600)
                elif fmt == 'pdf':
                    dpi = export_config.get('pdf_dpi', 600)
                elif fmt == 'svg':
                    dpi = export_config.get('svg_dpi', 600)
                else:
                    dpi = self.dpi
                
                # Save with high quality settings
                fig.savefig(
                    filepath,
                    dpi=dpi,
                    bbox_inches=export_config.get('bbox_inches', 'tight'),
                    pad_inches=export_config.get('pad_inches', 0.2),
                    facecolor=export_config.get('facecolor', '#FFFFFF'),
                    edgecolor=export_config.get('edgecolor', 'none'),
                    format=fmt
                )
                
                saved_files.append(filepath)
                self.logger.info(f"Saved {fmt.upper()} figure: {filepath}")
                
            except Exception as e:
                self.logger.error(f"Failed to save figure in {fmt} format: {e}")
        
        return saved_files
    
    # Helper methods
    def _determine_color_scale(self, layer_analyses: List[Dict], density_type: str) -> Tuple[float, float]:
        """Determine appropriate color scale with enhanced logic."""
        all_values = []
        
        for analysis in layer_analyses:
            for layer_name in ['whole', 'upper', 'middle', 'lower']:
                raster = analysis['rasters'].get(layer_name)
                if raster and raster['array'].size > 0:
                    positive_values = raster['array'][raster['array'] > 0]
                    if len(positive_values) > 0:
                        all_values.extend(positive_values.flatten())
        
        if not all_values:
            return self.color_scale_min, 1.0
        
        vmin = self.color_scale_min
        
        # Dynamic maximum based on data distribution
        if len(all_values) > 100:
            vmax = np.percentile(all_values, 99)
        else:
            vmax = np.percentile(all_values, 95)
        
        # Ensure reasonable range
        vmax = max(vmax, 1.0)
        vmax = min(vmax, self.color_scale_max)
        
        return vmin, vmax
    
    def _get_density_label(self, density_type: str) -> str:
        """Get proper label for density type."""
        labels = {
            'lad': 'Leaf Area',
            'wad': 'Wood Area', 
            'pad': 'Plant Area'
        }
        return labels.get(density_type.lower(), density_type.upper())
    
    def _create_empty_figure(self, message: str) -> plt.Figure:
        """Create empty figure with styled message."""
        fig, ax = plt.subplots(figsize=self.figsize_small, 
                              facecolor=self.style_manager.figure_background)
        ax.set_facecolor(self.style_manager.plot_background)
        
        annotation_props = self.style_manager.get_font_properties('annotation')
        ax.text(0.5, 0.5, message, ha='center', va='center',
                transform=ax.transAxes, fontdict=annotation_props)
        ax.axis('off')
        return fig
    
    def _set_equal_3d_aspect(self, ax, data):
        """Set equal aspect ratio for 3D plot."""
        x_range = data['x'].max() - data['x'].min()
        y_range = data['y'].max() - data['y'].min()
        z_range = data['z'].max() - data['z'].min()
        max_range = max(x_range, y_range, z_range)
        
        mid_x = (data['x'].max() + data['x'].min()) / 2
        mid_y = (data['y'].max() + data['y'].min()) / 2
        mid_z = (data['z'].max() + data['z'].min()) / 2
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


class EnhancedResultsManager:
    """Enhanced results manager with single plot support and table export."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize output directory
        self.output_dir = ensure_directory(output_dir)
        
        # Get plot mode
        viz_config = config.get('analysis', {}).get('visualization', {})
        self.plot_mode = viz_config.get('plot_mode', 'combined_plots')
        
        # Create subdirectories
        self.figures_dir = self._create_subdirectory("figures")
        self.data_dir = self._create_subdirectory("data") 
        self.reports_dir = self._create_subdirectory("reports")
        self.tables_dir = self._create_subdirectory("tables")
        
        self.logger.info(f"Enhanced results manager initialized - Mode: {self.plot_mode}")
    
    def _create_subdirectory(self, subdir_name: str) -> Path:
        """Create subdirectory with error handling."""
        try:
            subdir_path = self.output_dir / subdir_name
            return ensure_directory(subdir_path)
        except Exception as e:
            self.logger.error(f"Failed to create subdirectory '{subdir_name}': {e}")
            raise
    
    def save_enhanced_comparison_results(self, comparison_results: Dict[str, Any], 
                                       density_type: str, comparison_mode: str):
        """Save enhanced comparison results with all visualization types."""
        self.logger.info(f"Saving enhanced results for {density_type} comparison ({comparison_mode})")
        
        visualizer = ForestStructureVisualizer(self.config)
        
        # Crown layer comparison
        crown_fig = visualizer.create_crown_layer_comparison(comparison_results, density_type)
        base_filename = f"crown_layers_{density_type}_{comparison_mode}"
        visualizer.save_figure_with_options(crown_fig, self.figures_dir, base_filename, "crown_layers")
        plt.close(crown_fig)
        
        # Enhanced table visualization
        table_fig = visualizer.create_enhanced_table_visualization(comparison_results, density_type)
        table_filename = f"crown_metrics_table_{density_type}_{comparison_mode}"
        visualizer.save_figure_with_options(table_fig, self.figures_dir, table_filename, "metrics_tables")
        plt.close(table_fig)
        
        # Export table data in multiple formats
        exported_tables = visualizer.export_table_data(comparison_results, density_type, self.tables_dir)
        
        # Save metrics summary
        summary_df = comparison_results['summary']
        summary_path = self.data_dir / f"metrics_summary_{density_type}_{comparison_mode}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Enhanced comparison results saved for {density_type}")
    
    def save_enhanced_3d_results(self, datasets: List[pd.DataFrame], 
                               density_type: str, comparison_mode: str):
        """Save enhanced 3D visualization results."""
        visualizer = ForestStructureVisualizer(self.config)
        
        voxel_fig = visualizer.create_enhanced_3d_visualization(datasets, density_type)
        filename = f"3d_voxels_{density_type}_{comparison_mode}"
        visualizer.save_figure_with_options(voxel_fig, self.figures_dir, filename, "three_d_plots")
        plt.close(voxel_fig)
    
    def save_enhanced_distribution_results(self, datasets: List[pd.DataFrame], 
                                         density_type: str, comparison_mode: str):
        """Save enhanced distribution analysis results."""
        visualizer = ForestStructureVisualizer(self.config)
        
        dist_fig = visualizer.create_enhanced_distribution_comparison(datasets, density_type)
        filename = f"distributions_{density_type}_{comparison_mode}"
        visualizer.save_figure_with_options(dist_fig, self.figures_dir, filename, "distributions")
        plt.close(dist_fig)
    
    def save_profile_results(self, profile_data: Dict[str, Any], 
                           density_type: str, comparison_mode: str):
        """Save vertical profile comparison results."""
        visualizer = ForestStructureVisualizer(self.config)
        
        profile_fig = visualizer.create_vertical_profile_comparison(profile_data, density_type)
        filename = f"vertical_profiles_{density_type}_{comparison_mode}"
        visualizer.save_figure_with_options(profile_fig, self.figures_dir, filename, "vertical_profiles")
        plt.close(profile_fig)
    
    def create_enhanced_summary_report(self, all_results: Dict[str, Any], 
                                     dataset_summary: pd.DataFrame):
        """Create enhanced summary report."""
        report_path = self.reports_dir / "enhanced_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("VoxPlot Enhanced Analysis Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration details
            f.write("Analysis Configuration:\n")
            f.write("-" * 25 + "\n")
            analysis_config = self.config.get('analysis', {})
            viz_config = analysis_config.get('visualization', {})
            
            f.write(f"Plot Mode: {viz_config.get('plot_mode', 'combined_plots')}\n")
            f.write(f"Crown base height: {analysis_config.get('crown_base_height', 'N/A')} m\n")
            f.write(f"Voxel size: {analysis_config.get('voxel_size', 'N/A')} m\n")
            f.write(f"Minimum density: {analysis_config.get('min_density', 'N/A')}\n")
            f.write(f"Comparison mode: {analysis_config.get('comparison_mode', 'N/A')}\n")
            f.write(f"Output DPI: {viz_config.get('dpi', 'N/A')}\n")
            
            # Dataset summary
            f.write(f"\nDataset Summary ({len(dataset_summary)} datasets):\n")
            f.write("-" * 35 + "\n")
            for _, row in dataset_summary.iterrows():
                f.write(f"{row['Model']} ({row['Model_Type']}) - {row['Density_Type']}:\n")
                f.write(f"  Records: {row['Records']:,}\n")
                f.write(f"  Density Range: {row['Density_Range']}\n")
                f.write(f"  Mean Density: {row['Mean_Density']}\n\n")
            
            # Output structure
            f.write("Generated Outputs:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Figures: {self.figures_dir}\n")
            f.write(f"Data: {self.data_dir}\n")
            f.write(f"Tables: {self.tables_dir}\n")
            f.write(f"Reports: {self.reports_dir}\n")
            
            if self.plot_mode == "single_plots":
                f.write("\nSingle Plot Mode Active:\n")
                f.write("- Individual plots saved in themed subfolders\n")
                f.write("- Multiple export formats available\n")
        
        self.logger.info(f"Enhanced summary report saved: {report_path}")
        
    def create_summary_report(self, all_results: Dict[str, Any], 
                            dataset_summary: pd.DataFrame):
        """Create standard summary report for backward compatibility."""
        return self.create_enhanced_summary_report(all_results, dataset_summary)