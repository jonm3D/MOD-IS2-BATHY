import numpy as np
import pandas as pd
import os
from utils.peaks import detect_peaks, plot_peaks_debug
from utils.density import custom_kde
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)

def resample_1d(data, threshold=0.2):
    if len(data) == 0:
        return data
    
    # Sort the data
    sorted_data = np.sort(data)
    
    # Keep the first point
    resampled = [sorted_data[0]]
    last_kept = sorted_data[0]
    
    # Iterate through sorted data and keep points that are at least threshold away
    for point in sorted_data[1:]:
        if abs(point - last_kept) >= threshold:
            resampled.append(point)
            last_kept = point
    
    return np.array(resampled)

def process_nonsurface_data(chunk, min_data_points=25, bathy_snr_min=3, debug=False, debug_dir=None, nighttime=False):

    x_chunk = chunk['x'].values  # Along-track distance (meters)
    z_chunk = chunk['y'].values  # Elevation (meters, EGM08)
    i_chunk = chunk['original_index'].values  # Point indices
    subsurf_mask = chunk['subsurf_mask'].values.astype(bool)  # Subsurface mask (boolean array)
    chunk_id = chunk['chunk_id'].values  # Chunk identifier
        
    # Get atmospheric and subsurface data
    atmos_mask = ~subsurf_mask  # Invert subsurface mask to get atmospheric data
    x_atmos = x_chunk[atmos_mask]
    z_atmos = z_chunk[atmos_mask]

    x_subsurf = x_chunk[subsurf_mask]
    z_subsurf = z_chunk[subsurf_mask]
    i_subsurf = i_chunk[subsurf_mask]
    
    depth_pseudo = z_subsurf

    # # resample depth_pseudo
    # if nighttime:
    # depth_pseudo = resample_1d(depth_pseudo, threshold=0.05)

    # if nighttime, we want to increase the number of data in the 'atmospheric returns' to match the overall subsurf data density
    if len(z_atmos) < len(depth_pseudo):
        atmos_range = (0, np.ptp(depth_pseudo))
        n_needed = len(depth_pseudo) - len(z_atmos)
        z_atmos = np.concatenate((z_atmos, np.random.uniform(low=atmos_range[0], high=atmos_range[1], size=n_needed)))
        x_atmos = np.concatenate((x_atmos, np.random.uniform(low=np.min(x_chunk), high=np.max(x_chunk), size=n_needed)))
    
    n_resampled_total = len(z_atmos) + len(depth_pseudo)
    
    # elif len(z_atmos) > len(z_subsurf):
    #     # randomly draw z_subsurf points from z_atmos
    #     z_atmos = np.random.choice(z_atmos, size=len(z_subsurf), replace=False)
    #     x_atmos = np.random.choice(x_atmos, size=len(z_subsurf), replace=False)
        

    # Empty dataframes (also null outputs)
    subsurf_result = pd.DataFrame(columns=['ph_index', 'chunk_id', 'bathy_mu', 'bathy_sigma', 'kde_bandwidth', 'kde_method'])

    # Set up classification dataframe 
    bathy_score_df = pd.DataFrame({
        'z': z_subsurf,
        'i': i_subsurf,
        'chunk_id': chunk_id[0]*np.ones_like(z_subsurf),
        'score': np.zeros_like(z_subsurf),
    })

    # Estimate bandwidth using the entire column
    if len(z_chunk) > min_data_points:
        _, _, bathy_bw, _, _ = custom_kde(
            depth_pseudo, 
            method='isj',
        )

        # Compute subsurface density (mirror about 0 depth to avoid edge effect)
        bathy_grid, bathy_density, _, bw_success, bw_method = custom_kde(
            depth_pseudo, 
            method=bathy_bw,
            mirror=(np.floor(np.min(depth_pseudo)), 0),
            grid_max=0,
            # grid_min=np.floor(np.min(depth_pseudo)),
        )

        atmos_grid, atmos_density, _, _, _ = custom_kde(
            z_atmos, 
            method=bathy_bw,  
            grid_min=0,
            grid_max=np.ceil(np.max(z_atmos)),
            mirror=(0, np.ceil(np.max(z_atmos))),
        )

        # print(atmos_density[:10], atmos_density[-10:])
        
        # rescale density to match original quantities before peak finding
        atmos_density = atmos_density * len(z_atmos) / n_resampled_total
        bathy_density = bathy_density * len(depth_pseudo) / n_resampled_total

        # bathy_density = np.log(bathy_density + 1)  
        # atmos_density = np.log(atmos_density + 1)

        atmos_peaks = detect_peaks(
            density=atmos_density,
            grid=atmos_grid,
        )

        # Bathymetry peak finding
        bathy_peaks = detect_peaks(
            density=bathy_density,
            grid=bathy_grid,
            prominence=0,
            height=0,
        )

        atmos_peak_info_df = atmos_peaks['peak_info_df']
        atmos_peaks['peak_info_df']['pass'] = False # For plotting / consistency

        bathy_peak_info_df = bathy_peaks['peak_info_df']

        if not nighttime:
            if atmos_peak_info_df.shape[0] > 0:
                height_threshold = atmos_peak_info_df['heights'].median() * bathy_snr_min
            else:
                height_threshold = 0

        else:
            height_threshold = 0 #bathy_peak_info_df['heights'].median() * bathy_snr_min
        
        z_score_threshold = 3

        atmos_filter_pass = (bathy_peak_info_df['heights'] > height_threshold)

        # combine atmos+subsurf prominence values to z score stats
        prom_all = np.concatenate((atmos_peak_info_df['prominence'].values, bathy_peak_info_df['prominence'].values))
        # prom_all = atmos_peak_info_df['prominence'].values
        prom_mad = np.median(np.abs(prom_all - np.median(prom_all))) # prom_mad = np.median(np.abs(bathy_peak_info_df['prominence'] - np.median(bathy_peak_info_df['prominence'])))
        bathy_peak_info_df['prom_z'] = 0.6745 * (bathy_peak_info_df['prominence'] - np.median(prom_all)) / prom_mad

        auc_all = np.concatenate((atmos_peak_info_df['empirical_auc'].values, bathy_peak_info_df['empirical_auc'].values))
        # auc_all = atmos_peak_info_df['empirical_auc'].values
        auc_mad = np.median(np.abs(auc_all - np.median(auc_all))) # auc_mad = np.median(np.abs(bathy_peak_info_df['empirical_auc'] - np.median(bathy_peak_info_df['empirical_auc'])))

        bathy_peak_info_df['auc_z'] = 0.6745 * (bathy_peak_info_df['empirical_auc'] - np.median(auc_all)) / auc_mad
        positive_bathy_flag = (bathy_peak_info_df['prom_z'] > z_score_threshold) & (bathy_peak_info_df['auc_z'] > z_score_threshold)
        bathy_peak_info_df['pass'] = (positive_bathy_flag) & (atmos_filter_pass) 

        # Update compute scores for any passing peaks
        for i, peak in bathy_peak_info_df.iterrows():
            if peak['pass']:
                # Compute Gaussian score for each point in the subsurface data
                # subsurf_scores = gaussian_score(
                #     bathy_score_df['z'],
                #     mu=peak['location'],
                #     sigma=peak['sigma_est'],
                # )

                subsurf_scores = uniform_score(
                    bathy_score_df['z'],
                    left_pos=peak['left_position'],
                    right_pos=peak['right_position'],
                )

                # Update the score in the dataframe
                bathy_score_df.loc[:, 'score'] += subsurf_scores

        # Clip any scores above 1.0 (ie. peaks within peaks)
        bathy_score_df['score'] = np.clip(bathy_score_df['score'], 0, 1)

        if debug and debug_dir is not None:
            try:
                nonsurf_debug_path = os.path.join(debug_dir, f"nonsurface_peaks_chunk_{chunk_id[0]}.png")
                plot_unified_debug(
                    atmos_x=x_atmos,
                    atmos_z=z_atmos,
                    atmos_density_data={'grid': atmos_grid, 'density': atmos_density},
                    atmos_peak_info_df=atmos_peaks.get('peak_info_df', pd.DataFrame())[:10],
                    subsurf_x=x_subsurf,
                    subsurf_z=z_subsurf,
                    subsurf_density_data={'grid': bathy_grid, 'density': bathy_density},
                    subsurf_peak_info_df=bathy_peak_info_df[:10],
                    output_path=nonsurf_debug_path,
                    # peak_threshold=snr_threshold,
                    bw_method=bw_method,
                    bw_size=bathy_bw,
                    # bathy_score_df=bathy_score_df,
                )
            except Exception as e:
                logger.warning(f"Error creating debug plots: {str(e)}")
        
        # Process and format results 
        subsurf_result = pd.DataFrame({
            # 'ph_index': i_subsurf,
            'chunk_id': chunk_id[0],
            'bathy_mu': bathy_peak_info_df['location'].values if not bathy_peak_info_df.empty else np.nan,
            'bathy_sigma': bathy_peak_info_df['sigma_est'].values if not bathy_peak_info_df.empty else np.nan,
            'kde_bandwidth': bathy_bw,
            'kde_method': str(bw_method)
        }, index=[0] if bathy_peak_info_df.shape[0] <= 1 else None)

    return {
        'subsurf_result': subsurf_result,
        'bathy_score_df': bathy_score_df,
    }

def plot_unified_debug(atmos_x, atmos_z, atmos_density_data, atmos_peak_info_df,
                      subsurf_x, subsurf_z, subsurf_density_data, subsurf_peak_info_df,
                      output_path, peak_threshold=None,
                      bw_method=None, bw_size=None,
                      bathy_score_df=None):  # Add parameter for scores

    # Get density data from dict
    bathy_density = subsurf_density_data.get('density', np.array([]))
    bathy_grid = subsurf_density_data.get('grid', np.array([]))
    atmos_density = atmos_density_data.get('density', np.array([]))
    atmos_grid = atmos_density_data.get('grid', np.array([]))

    y_max = np.abs([np.max(atmos_z), np.max(subsurf_z)]).max()
    
    if len(atmos_density) == 0 or len(atmos_grid) == 0:
        logger.error("No density data provided")
        return
    
    # Create figure with gridspec for custom layout - now 2x3 with third column for tables
    fig = plt.figure(figsize=(15, 6))  # Wider to accommodate tables
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 1.5])  # Added third column for tables
    fig.suptitle("Peak Detection Analysis", fontsize=14, fontweight='bold')

    # Plot atmospheric data
    ax_atmos_data = plt.subplot(gs[0, 0])
    ax_atmos_density = plt.subplot(gs[0, 1], sharey=ax_atmos_data) 
    ax_atmos_table = plt.subplot(gs[0, 2])  # New axis for atmospheric table
    
    ax_atmos_data.scatter(atmos_x, atmos_z, s=3, alpha=0.7, color='black', label='Data Points')
    atmos_grid_spacing = np.mean(np.diff(atmos_grid)) if len(atmos_grid) > 1 else 0.1
    ax_atmos_density.barh(atmos_grid, atmos_density, height=atmos_grid_spacing, alpha=0.7, color='gray', label='Density')

    # Atmospheric plot formatting
    ax_atmos_data.set_ylim(0, y_max)
    ax_atmos_data.set_xlabel('x')
    ax_atmos_data.set_ylabel('z')
    ax_atmos_data.grid(True, alpha=0.3)
    ax_atmos_density.set_xlabel('Density')
    ax_atmos_density.grid(True, alpha=0.3)
    ax_atmos_density.set_title(f'Atmospheric Density\nBW: {bw_size:.3f} ({bw_method})')

    # bathy plots
    ax_bathy_data = plt.subplot(gs[1, 0])
    ax_bathy_density = plt.subplot(gs[1, 1], sharey=ax_bathy_data, sharex=ax_atmos_density)
    ax_bathy_table = plt.subplot(gs[1, 2])  # New axis for bathymetry table

    # Plot subsurface data with color according to scores if provided
    if bathy_score_df is not None and len(bathy_score_df) == len(subsurf_z):
        scatter = ax_bathy_data.scatter(
            subsurf_x, subsurf_z, 
            s=3, alpha=0.7, 
            c=bathy_score_df['score'].values,  # Color by score
            cmap='viridis',    # Use viridis colormap
            label='Data Points'
        )
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax_bathy_data)
        cbar.set_label('Bathy Score')
    else:
        # Original behavior if no scores provided
        ax_bathy_data.scatter(subsurf_x, subsurf_z, s=3, alpha=0.7, color='black', label='Data Points')

    bathy_grid_spacing = np.mean(np.diff(bathy_grid)) if len(bathy_grid) > 1 else 0.1
    ax_bathy_density.barh(bathy_grid, bathy_density, height=bathy_grid_spacing, alpha=0.7, color='gray', label='Density')

    # Plot points with error bars for peaks that pass
    passing_peaks = subsurf_peak_info_df[subsurf_peak_info_df['pass'] == True]
    if not passing_peaks.empty:
        # For each peak that passes, plot a point at the peak location and add visual elements
        for _, row in passing_peaks.iterrows():
            # Plot a point at the peak location
            ax_bathy_density.plot(
                row['heights'], 
                row['location'], 
                'ro',  # Red dot
                markersize=6,
                label='Peak' if _ == 0 else ""  # Only add label for the first point
            )
            
            # # Add the sigma-based error bars (keep these as they were)
            # ax_bathy_density.errorbar(
            #     row['heights'],  # x-position (using peak height as x-coordinate)
            #     row['location'],  # y-position (peak location)
            #     yerr=3*row['sigma_est'],  # error bar height is 3*sigma
            #     fmt='none',  # No marker
            #     color='red',
            #     capsize=4,  # Size of cap on error bars
            #     alpha=0.7,
            #     label='±3σ range' if _ == 0 else ""  # Only add label for the first error bar
            # )
            
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
                ax_bathy_density.add_patch(rect)
    
    # Bathy plot formatting
    y_min = np.min(subsurf_z)
    ax_bathy_data.set_ylim(y_min, 0)
    ax_bathy_data.set_xlabel('x')
    ax_bathy_data.set_ylabel('z')
    ax_bathy_data.grid(True, alpha=0.3)
    ax_bathy_density.set_xlabel('Density')
    ax_bathy_density.set_title(f'Bathymetry Density\nBW: {bw_size:.3f} ({bw_method})')
    
    if peak_threshold is not None:
        ax_atmos_density.axvline(x=peak_threshold, color='red', linestyle='--', 
                          linewidth=1.5, label=f'Threshold: {peak_threshold:.4f}')
        ax_atmos_density.legend(loc='upper right', fontsize=8)

        ax_bathy_density.axvline(x=peak_threshold, color='red', linestyle='--', 
                          linewidth=1.5, label=f'Threshold: {peak_threshold:.4f}')
        ax_bathy_density.legend(loc='upper right', fontsize=8)
    elif not passing_peaks.empty:
        # If we have passing peaks but no threshold, still show the legend
        ax_bathy_density.legend(loc='upper right', fontsize=8)

    # Add peak info tables
    add_peak_info_table(ax_atmos_table, atmos_peak_info_df.head(25), "Atmospheric Peaks")
    add_peak_info_table(ax_bathy_table, subsurf_peak_info_df.head(25), "Bathymetry Peaks")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

def add_peak_info_table(ax, peak_info_df, title):
    """
    Helper function to add a peak information table to a specific axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis where the table will be placed
    peak_info_df : pandas.DataFrame
        DataFrame containing peak information
    title : str
        Title for the table
    """
    # Turn off axis
    ax.axis('off')
    ax.set_title(title)
    
    # If the DataFrame is empty, show a message
    if peak_info_df.empty:
        ax.text(0.5, 0.5, "No peaks detected", ha='center', va='center')
        return
    
    # Create table data
    columns = ['Peak', 'μ', 'σ', 'Prominence', 'z(Prom)', 'AUC', 'z(AUC)', 'Pass']
    
    # Prepare data for the table
    table_data = []
    for i, (_, peak) in enumerate(peak_info_df.iterrows()):
        peak_num = i + 1
        loc = f"{peak['location']:.2f}"
        sigma = f"{peak['sigma_est']:.2f}"
        prom = f"{peak.get('prominence', np.nan):.4f}" if not np.isnan(peak.get('prominence', np.nan)) else "N/A"
        prom_z = f"{peak.get('prom_z', np.nan):.4f}" if not np.isnan(peak.get('prom_z', np.nan)) else "N/A"
        auc = f"{peak.get('empirical_auc', np.nan):.4f}" if not np.isnan(peak.get('empirical_auc', np.nan)) else "N/A"
        auc_z = f"{peak.get('auc_z', np.nan):.4f}" if not np.isnan(peak.get('auc_z', np.nan)) else "N/A"
        passed = "Yes" if peak.get('pass', False) else "No"
        
        table_data.append([peak_num, loc, sigma, prom, prom_z, auc, auc_z, passed])
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.1, 0.15, 0.15, 0.225, 0.225, 0.225, 0.225, 0.1]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)  # Adjust row height
    
    # Style the header
    for i, col in enumerate(columns):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#e6e6e6')

def gaussian_score(x, mu, sigma):
    """
    Assign scores to 1D data points using a Gaussian distribution, 
    with score=1 at mu and score=0 beyond 5 sigma.
    
    Parameters:
    -----------
    x : array-like
        The 1D data points to score
    mu : float
        The mean of the Gaussian distribution
    sigma : float
        The standard deviation of the Gaussian distribution
    
    Returns:
    --------
    scores : array-like
        Scores between 0 and 1 with Gaussian weighting
    """
    # Compute the Gaussian values directly
    gaussian = np.exp(-0.5 * ((x - mu) / sigma)**2)
    
    # Normalize so the peak at mu is 1.0
    gaussian = gaussian / np.exp(-0.5 * 0)  # exp(-0.5 * 0) = 1, but being explicit
    
    # Create cutoff at 5 sigma
    z_score = np.abs(x - mu) / sigma
    scores = np.where(z_score <= 5, gaussian, 0)
    
    return scores

def uniform_score(x, left_pos, right_pos):
    """
    Assign uniform scores to data points within a specified range.
    
    Parameters:
    -----------
    x : array-like
        The 1D data points to score
    left_pos : float
        The left boundary of the scoring range
    right_pos : float
        The right boundary of the scoring range
    
    Returns:
    --------
    scores : array-like
        Scores between 0 and 1 within the specified range, 0 outside
    """
    # Create uniform scores between left_pos and right_pos
    scores = np.where((x >= left_pos) & (x <= right_pos), 1.0, 0.0)
    
    return scores