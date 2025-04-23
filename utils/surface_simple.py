import numpy as np
import pandas as pd
from scipy.stats import kstest
import logging
import os

logger = logging.getLogger(__name__)

def is_uniform(data, alpha=0.05):
    """
    Determines if a given 1D array of data comes from a uniform distribution.

    Parameters:
    - data (array-like): The 1D array of data to be tested.
    - alpha (float): Significance level for the hypothesis test. Default is 0.05.

    Returns:
    - bool: True if data is likely from a uniform distribution, False otherwise.
    """
    # Normalize to 0, 1 if not already
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    ks_stat, p_value = kstest(data, 'uniform')

    # Return True if we fail to reject the null hypothesis (p-value > alpha)
    return p_value > alpha

def simple_surface_model(chunk, grid_points=1000, kde_method='silverman', min_data_points=50, debug=False, debug_dir=None):
    """
    Process a chunk of data to model the water surface using KDE.
    
    Parameters:
    ----------
    chunk : numpy.ndarray
        Array with shape (n, 4) containing:
        - x-coordinates (along-track distance in meters)
        - z-coordinates (elevation in meters, referenced to EGM08 geoid)
        - indices (point indices)
        - chunk_id (identifier for the chunk, same value for all points in a chunk)
    grid_points : int, optional
        Number of points in the evaluation grid for KDE (default: 1000)
    min_prominence : float, optional
        Minimum prominence required for peak detection as fraction of max height (default: 0.05)

    kde_method : str, optional
        Method to use for KDE bandwidth selection
    min_data_points : int, optional
        Minimum number of data points required to process the chunk (default: 50)
    debug : bool, optional
        Enable debug mode with additional plotting
    debug_dir : str, optional
        Directory for debug output files
    
    Returns:
    -------
    pd.DataFrame
        DataFrame with all points and their classifications
    """
    from utils.density import custom_kde
    from utils.peaks import detect_peaks
    
    # breakpoint()
    # Extract coordinates, indices and chunk_id
    x_chunk = chunk[:, 0]  
    z_chunk = chunk[:, 1]  # Elevation data
    i_chunk = chunk[:, 2]  # Indices (point indices)
    atl03_cnf = chunk[:, 3]  # Classification flags (ATL03)
    chunk_id = chunk[:, 4]  # Chunk ID (same for all points in a chunk)

    # Initialize empty DataFrame with proper columns for the case when we can't model the surface
    empty_result = pd.DataFrame({
        'ph_index': i_chunk,
        'chunk_id': chunk_id,
        'surf_sigma': np.nan,
        'surf_mu': np.nan,
        'surf_A': np.nan,
        'kde_bandwidth': np.nan,
        'surf_left': np.nan,
        'surf_right': np.nan
    })

    noise_result = empty_result.copy()
    noise_result['classification'] = 1  # All points classified as noise

    # Check for minimum number of data points
    if len(z_chunk) < min_data_points:
        logger.info(f"Insufficient data points: {len(z_chunk)} < {min_data_points}")
        return empty_result
        
    # Require at least 3 distinct elevation values to model the surface
    if len(np.unique(z_chunk)) < 3:
        logger.info(f"Insufficient unique values: {len(np.unique(z_chunk))} < 3")
        return empty_result

    # Require non-uniform distribution of elevations
    if is_uniform(z_chunk):
        logger.info("Data appears uniformly distributed, likely noise")
        return noise_result

    # Require at least 10 points with atl03_cnf >= 3
    if np.sum(atl03_cnf >= 3) < 10:
        logger.info(f"Insufficient points with atl03_cnf >= 3: {np.sum(atl03_cnf >= 3)} < 10")
        return empty_result
    
    # Generate KDE for elevation data
    try:
        # Determine range for KDE evaluation
        z_min, z_max = np.min(z_chunk), np.max(z_chunk)
        
        # Add padding to the range for better edge handling
        padding = (z_max - z_min) * 0.1
        grid_min = z_min - padding
        grid_max = z_max + padding
        
        # Use custom_kde for density estimation
        grid, density, bw, success, method_used = custom_kde(
            z_chunk, 
            method=kde_method,
            grid_points=grid_points,
            grid_min=grid_min,
            grid_max=grid_max
        )
        
        if not success:
            logger.warning(f"KDE calculation failed using {method_used}")
            return empty_result
            
        # Detect peaks using the density
        peaks_result = detect_peaks(
            density=density,
            grid=grid,
            rel_height_maj=0.9,
        )
    except Exception as e:
        logger.error(f"Error in KDE or peak detection: {str(e)}")
        return empty_result
    
    # Extract peak information
    peak_info_df = peaks_result['peak_info_df']
    
    # If no peaks were found, return empty result
    if peak_info_df.empty:
        logger.info("No peaks detected")
        return empty_result
        
    # Generate debug plot if debug is enabled and we have a directory
    if debug and debug_dir is not None:
        try:
            # Create a unique filename for this chunk
            chunk_id_val = int(chunk_id[0]) if len(chunk_id) > 0 else 0
            debug_filename = f"surface_peaks_chunk_{chunk_id_val}.png"
            debug_path = os.path.join(debug_dir, debug_filename)
            
            # Import the plotting function
            from utils.peaks import plot_peaks_debug
            
            # Get the surface peak for highlighting
            surface_peak_location = None
            if len(peak_info_df) > 0:
                # Sort peaks by prominence and get the top one
                peak_info_sorted = peak_info_df.sort_values('prominence', ascending=False)
                surface_peak_location = peak_info_sorted.iloc[0]['location']
            
            # Generate the plot
            plot_peaks_debug(
                data=z_chunk,
                density_data=peaks_result['density_data'],
                peak_info_df=peak_info_df,
                output_path=debug_path,
                invert_y_axis=False,
                z_range=(-10, 10)
            )

        except Exception as e:
            logger.warning(f"Error creating debug plot for surface: {str(e)}")
    
    # Sort peaks by prominence (should already be sorted, but just to be sure)
    # peak_info_df = peak_info_df.sort_values('prominence', ascending=False)

    # remove peaks with fwhm > 5m (likely noise/poor surface fits)
    peak_info_df = peak_info_df[peak_info_df['fwhm'] < 5.0]
    # peak_info_df = peak_info_df.sort_values('empirical_auc', ascending=False)

    # Select surface peak 
    # If multiple peaks, check if second peak might be the surface
    if len(peak_info_df) > 1:
        surface_peak = peak_info_df.iloc[0]
        second_peak = peak_info_df.iloc[1]
        
        # Check if second peak is higher elevation and has significant prominence
        if (second_peak['location'] > (surface_peak['location']) and 
            second_peak['prominence'] > 0.5 * surface_peak['prominence']):
            surface_peak = second_peak
    else:
        surface_peak = peak_info_df.iloc[0]

    mu_surf_prior = surface_peak['location']
    sigma_surf_prior = surface_peak['sigma_est']
    A_surf_prior = surface_peak['heights']

    # Create result DataFrame with all points and their classifications
    result_df = pd.DataFrame({
        'ph_index': i_chunk,
        'chunk_id': chunk_id.astype(np.int8),
        'surf_sigma': sigma_surf_prior,
        'surf_mu': mu_surf_prior,
        'surf_A': A_surf_prior,
        'surf_left': surface_peak['left_position'],
        'surf_right': surface_peak['right_position'],
        'kde_bandwidth': bw  
    })

    return result_df