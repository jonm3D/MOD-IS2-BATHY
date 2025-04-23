import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)
def detect_peaks(density, grid, prominence=0.0, height=0.0, rel_height_maj=0.9):
    """
    Simple peak detection function that works directly with density and grid data.
    
    Parameters:
    -----------
    density : array-like
        KDE density values
    grid : array-like
        Grid points corresponding to density values
    prominence : float, optional
        Minimum prominence for peak detection as fraction of max height (default: 0.0)
    height : float, optional
        Minimum height for peak detection (default: 0.0)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'peak_info_df': DataFrame with peak information
        - 'density_data': Dict with 'grid' and 'density' values
    """
    # Define empty DataFrame with correct columns as fallback
    empty_df = pd.DataFrame(columns=[
        'index', 'location', 'heights', 'prominence', 
        'fwhm', 'sigma_est', 'empirical_auc', 'auc', 'prominence_ratio'
    ])
    
    # Check inputs
    if len(density) != len(grid):
        logger.error(f"Density and grid must have same length: {len(density)} vs {len(grid)}")
        return {'peak_info_df': empty_df, 'density_data': {'grid': grid, 'density': density}}
    
    # Check for valid density values
    if len(density) == 0 or np.all(np.isnan(density)):
        logger.warning("Invalid density values (empty or all NaN)")
        return {'peak_info_df': empty_df, 'density_data': {'grid': grid, 'density': density}}
    
    max_height = np.max(density)
    if max_height <= 0 or np.isnan(max_height):
        logger.warning("Invalid density values (all zeros/negative or NaN)")
        return {'peak_info_df': empty_df, 'density_data': {'grid': grid, 'density': density}}
    
    # Find peaks with minimum prominence
    try:
        peak_indices, properties = find_peaks(
            density, 
            prominence=prominence,
            height=height,
        )
    except Exception as e:
        logger.error(f"Error during peak detection: {e}")
        return {'peak_info_df': empty_df, 'density_data': {'grid': grid, 'density': density}}
    
    # If no peaks found, return empty result
    if len(peak_indices) == 0:
        logger.info("No peaks detected with the specified prominence threshold")
        return {'peak_info_df': empty_df, 'density_data': {'grid': grid, 'density': density}}
    
    # Calculate peak widths (FWHM)
    try:
        # Get average grid spacing
        grid_spacing = np.mean(np.diff(grid)) if len(grid) > 1 else 0.1
        
        # Calculate width at half-maximum
        results_half = peak_widths(density, peak_indices, rel_height=0.5)
        fwhm = results_half[0] * grid_spacing  # Convert to actual width using grid spacing
        sigma_est = fwhm / 2.355  # Convert FWHM to sigma (assuming Gaussian shape)

        # intersection points
        results_maj = peak_widths(density, peak_indices, rel_height=rel_height_maj)
        # left_ips = results_half[2]
        # right_ips = results_half[3]
        left_ips = results_maj[2]
        right_ips = results_maj[3]
        left_positions = np.interp(left_ips, np.arange(len(grid)), grid)
        right_positions = np.interp(right_ips, np.arange(len(grid)), grid)

    except Exception as e:
        logger.warning(f"Error in peak width calculation: {e}")
        fwhm = np.full_like(peak_indices, np.nan, dtype=float)
        sigma_est = np.full_like(peak_indices, 0.1, dtype=float)  # Default sigma
    
    # Calculate AUC (area under curve) for each peak using prominence and sigma
    auc_est = properties['prominences'] * sigma_est * np.sqrt(2 * np.pi)
    
    # Calculate empirical AUC by numerical integration between left and right intersection points
    empirical_auc = []
    for i, peak_idx in enumerate(peak_indices):
        try:
            # Convert float indices to integer indices for slicing, ensuring they're within bounds
            left_idx = max(0, int(np.floor(left_ips[i])))
            right_idx = min(len(density) - 1, int(np.ceil(right_ips[i])))
            
            # If left and right indices are the same, expand slightly
            if left_idx == right_idx:
                left_idx = max(0, left_idx - 1)
                right_idx = min(len(density) - 1, right_idx + 1)
            
            # Extract the density values within these bounds
            segment_density = density[left_idx:right_idx+1]
            segment_grid = grid[left_idx:right_idx+1]
            
            # Calculate empirical AUC using trapezoidal rule
            emp_auc = np.trapz(segment_density, segment_grid)
            empirical_auc.append(emp_auc)
            
        except Exception as e:
            logger.warning(f"Error calculating empirical AUC for peak {i}: {e}")
            empirical_auc.append(np.nan)

    # Create peak info DataFrame
    try:
        peak_info_df = pd.DataFrame({
            'index': peak_indices,
            'location': grid[peak_indices],
            'heights': density[peak_indices],
            'prominence': properties['prominences'],
            'fwhm': fwhm,
            'sigma_est': sigma_est,
            'auc': auc_est,
            'empirical_auc': empirical_auc,
            'prominence_ratio': properties['prominences'] / density[peak_indices],
            'left_ips': left_ips,  # Original indices
            'right_ips': right_ips,  # Original indices
            'left_position': left_positions,  # New grid-based positions
            'right_position': right_positions,  # New grid-based positions
        })
        
        # Sort peaks by prominence (highest first)
        peak_info_df = peak_info_df.sort_values('prominence', ascending=False)
        peak_info_df = peak_info_df.reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error creating peak info DataFrame: {e}")
        return {'peak_info_df': empty_df, 'density_data': {'grid': grid, 'density': density}}
    
    # Return peak information and density data
    return {
        'peak_info_df': peak_info_df,
        'density_data': {'grid': grid, 'density': density}
    }
    
def plot_peaks_debug(data, density_data, peak_info_df, output_path, 
                   invert_y_axis=False, peak_threshold=None, z_range=(None, None)):
    """
    Creates a debug plot for peak analysis with data scatter, density plot, and peak info table.
    
    Parameters:
    -----------
    data : array-like
        The raw data values (z-coordinates, elevations, etc.)
    density_data : dict
        Dictionary with density data containing:
        - 'grid' or 'bin_centers': grid points for KDE evaluation
        - 'density' or 'hist': density values from KDE
    peak_info_df : pandas.DataFrame
        DataFrame with peak information
    output_path : str
        Path to save the debug plot
    invert_y_axis : bool, optional
        Whether to invert the y-axis (e.g., for depth data) (default: False)
    peak_threshold : float, optional
        Threshold value used for peak detection. If provided, a horizontal line
        will be drawn on the density plot to visualize the threshold (default: None)
    """
    # Get density data from dict - support both old and new naming conventions
    density = density_data.get('density', density_data.get('hist', np.array([])))
    grid = density_data.get('grid', density_data.get('bin_centers', np.array([])))
    
    if len(density) == 0 or len(grid) == 0:
        logger.error("No density data provided")
        return
    
    # Create figure with gridspec for custom layout
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 2])
    
    # Set plot titles
    fig.suptitle("Peak Detection Analysis", fontsize=14, fontweight='bold')
    
    # Data plot (left)
    ax_data = plt.subplot(gs[0])
    ax_density = plt.subplot(gs[1], sharey=ax_data)
    
    # Plot raw data points
    ax_data.scatter(range(len(data)), data, s=3, alpha=0.7, color='black', label='Data Points')

    # Calculate spacing for density bars
    grid_spacing = np.mean(np.diff(grid)) if len(grid) > 1 else 0.1
    
    # Vertical density plot (right)
    ax_density.barh(grid, density, height=grid_spacing, alpha=0.7, color='gray', label='Density')
    
    # Add peak threshold line if provided
    if peak_threshold is not None:
        ax_density.axvline(x=peak_threshold, color='red', linestyle='--', 
                          linewidth=1.5, label=f'Threshold: {peak_threshold:.4f}')
        ax_density.legend(loc='upper right', fontsize=8)

    # for the first peak
    for _, row in peak_info_df[:1].iterrows():

            # Add shaded region from left to right position if available
            if 'left_position' in row and 'right_position' in row and not np.isnan(row['left_position']) and not np.isnan(row['right_position']):
                # Create shaded rectangle from left to right position
                # The width of the rectangle should be some fraction of the peak height
                rect_width = row['heights'] * 0.1 
                
                # Add a semi-transparent rectangle
                rect = plt.Rectangle(
                    (0, row['left_position']),  # (x, y) of bottom left corner
                    rect_width,  # width
                    row['right_position'] - row['left_position'],  # height
                    color='blue',
                    alpha=0.5,
                    label='Peak width' if _ == 0 else ""  # Only add label for the first rectangle
                )
                ax_density.add_patch(rect)
    
    # Set y-axis limits to the range of the data
    y_min = np.min(data) if z_range[0] is None else z_range[0]
    y_max = np.max(data) if z_range[1] is None else z_range[1]
    
    # Add a small padding to the limits (5% on each side)
    y_padding = 0.05 * (y_max - y_min)
    ax_data.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Invert y-axis if requested (e.g., for depth data)
    if invert_y_axis:
        ax_data.invert_yaxis()

    # Set y axis lim for both
    
    # Set labels 
    ax_data.set_xlabel('Point Index')
    ax_data.set_ylabel('Value')
    ax_data.grid(True, alpha=0.3)
    
    ax_density.set_xlabel('Density')
    ax_density.grid(True, alpha=0.3)
    
    # Use the shared y-axis ticks
    plt.setp(ax_density.get_yticklabels(), visible=True)
    
    # Add peak info table if peaks were found
    if not peak_info_df.empty:
        _add_peak_info_table(fig, peak_info_df[:10])  # Show only top 10 peaks`]
    
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.9)  # Make room for the text table and suptitle
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def _add_peak_info_table(fig, peak_info_df):
    """
    Helper function to add a peak information table to the figure.
    """
    # Add a formatted summary table at the bottom
    table_text = "Peak Summary:\n"
    header = f"{'Peak':<5} {'μ':<10} {'σ':<10} {'Height':<10} {'Prominence':<12} {'AUC':<10} {'Emp.AUC':<10} {"Left":<10} {'Right':<10}\n"
    separator = f"{'-'*5:<5} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10} {'-'*12:<12} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10} {'-'*10:<10}\n"
    table_text += header + separator
    
    for i, (_, peak) in enumerate(peak_info_df.iterrows()):
        # Format each value with consistent spacing and decimal precision
        peak_num = f"{i+1:<5}"
        loc = f"{peak['location']:.4f}".ljust(10)
        sigma = f"{peak['sigma_est']:.4f}".ljust(10)
        height = f"{peak['heights']:.4f}".ljust(10)
        prom = f"{peak['prominence']:.4f}".ljust(12)
        auc = f"{peak.get('auc', np.nan):.4f}".ljust(10) if not np.isnan(peak.get('auc', np.nan)) else "N/A".ljust(10)
        emp_auc = f"{peak.get('empirical_auc', np.nan):.4f}".ljust(10) if not np.isnan(peak.get('empirical_auc', np.nan)) else "N/A".ljust(10)
        left = f"{peak.get('left_position', np.nan):.4f}".ljust(10) if not np.isnan(peak.get('left_position', np.nan)) else "N/A".ljust(10)
        right = f"{peak.get('right_position', np.nan):.4f}".ljust(10) if not np.isnan(peak.get('right_position', np.nan)) else "N/A".ljust(10)
        
        table_text += f"{peak_num} {loc} {sigma} {height} {prom} {auc} {emp_auc} {left} {right}\n"
    
    fig.text(0.5, 0.01, table_text, fontsize=8, va='top', ha='center', family='monospace')