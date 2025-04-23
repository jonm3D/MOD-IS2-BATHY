#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.n_chunk import n_chunk, mask_n_chunk
from utils.x_chunk import x_chunk
from utils.refraction import correction
from utils.surface_processing import surface_model
from utils.bathy_processing import process_nonsurface_data
import warnings
import logging
import matplotlib.pyplot as plt
import os

def process_profile(data, **config):
    """
    Process a single ICESat-2 profile with all configuration parameters passed as kwargs.
    
    Args:
        data: The profile data to process
        **config: Configuration parameters as keyword arguments
        
    Returns:
        dict: Results of processing, including the processed data
    """
    # Extract parameters from config dict with defaults
    method_surface_bins = config.get('method_surface_bins', 'adaptive')
    n_ph_surface_bins = config.get('n_ph_surface_bins', 100)
    x_m_surface_bins = config.get('x_m_surface_bins', 100)
    overlap_surface = config.get('overlap_surface', 0.5)
    max_gap_bins_m = config.get('max_gap_bins_m', 1000)
    surf_sigma_cutoff = config.get('surf_sigma_cutoff', 2.0)
    method_nonsurf_bins = config.get('method_nonsurf_bins', 'adaptive')
    n_ph_nonsurf_bins = config.get('n_ph_nonsurf_bins', 50)
    x_m_nonsurf_bins = config.get('x_m_nonsurf_bins', 100)
    overlap_nonsurf = config.get('overlap_nonsurf', 0.5)
    n_nodes = config.get('n_nodes', 5)
    kde_surf_method = config.get('kde_surf_method', 'silverman')  # Uncommented
    kde_surf_grid_points = config.get('kde_surf_grid_points', 1000)
    debug = config.get('debug', False)
    debug_dir = config.get('debug_dir', None)
    min_data_surface = config.get('min_data_surface', 50)
    min_data_nonsurface = config.get('min_data_nonsurface', 25)
    bathy_snr_min = config.get('bathy_snr_min', 3)
    force_serial = config.get('serial', False)  # Added to extract serial mode flag
    
    logger = logging.getLogger(__name__)
    
    try:
        ############################## Setup ##############################
        data = data.copy(deep=True) 

        # Remove data over dry land
        if 'water_mask' in data.columns:
            data = data[data.water_mask == 1]
            if len(data) == 0:
                logger.warning("No water points found after water mask filtering")
                # return data with no rows
                return {
                    'data': pd.DataFrame(columns=[data.columns + 'classification' + 'confidence']),
                }
        
        # Sort by along-track distance
        data = data.sort_values(by='x_ph')
        
        # Create unique photon index for the remaining data
        data['ph_index'] = np.arange(len(data))
        data = data.reset_index(drop=True)
        
        # Initialize classification storage
        data_classification = pd.Series(0, index=data.index, dtype=int)
        data_confidence = pd.Series(0, index=data.index, dtype=float)
        
        if data.solar_elevation.mean() < 0:
            nighttime = True
        else:
            nighttime = False

        ############################## Surface Modeling ##############################
        x_ = data.x_ph.values
        z_ = data.z_ph.values
        inds_ = data.index.values  
        atl03_cnf_ = data.atl03_cnf.values 
        additional_surf_chunk_data = {
            'atl03_cnf': atl03_cnf_,
        }

        # Break the data into chunks for processing
        if method_surface_bins == 'n':
            surface_chunks = n_chunk(x0=x_, y0=z_, points_per_chunk=n_ph_surface_bins, 
                                    indices=inds_, overlap=overlap_surface, max_gap_size=max_gap_bins_m,
                                    additional_data=additional_surf_chunk_data)
        elif method_surface_bins == 'x':
            surface_chunks = x_chunk(x0=x_, y0=z_,
                                    bin_size=x_m_surface_bins, 
                                    indices=inds_, overlap=overlap_surface, max_gap_size=max_gap_bins_m,
                                    additional_data=additional_surf_chunk_data)
        else:
            raise ValueError(f"method_surface_bins = {method_surface_bins} not recognized. Use 'n' or 'x'.")

        if not surface_chunks:
            logger.warning("No valid surface chunks found")
            data.loc[:, 'classification'] = data_classification
            data.loc[:, 'confidence'] = data_confidence
            return {
                'data': data,
            }        

        def process_chunk(chunk):
            return surface_model(
                chunk.values, 
                grid_points=kde_surf_grid_points, 
                kde_method=kde_surf_method,
                min_data_points=min_data_surface,
                debug=debug,
                debug_dir=debug_dir
            )
            
        if force_serial:
            results = [process_chunk(chunk) for chunk in surface_chunks]
        else:
            try:
                with ProcessPool(n_nodes) as pool:
                    results = pool.map(process_chunk, surface_chunks)
            except Exception as e:
                logger.error(f"Error in parallel processing of surface chunks: {str(e)}")
                # Fall back to serial processing
                results = [process_chunk(chunk) for chunk in surface_chunks]
        
        # Combine all results into a single DataFrame
        if not results:
            logger.warning("No results from surface modeling")
            data.loc[:, 'classification'] = data_classification
            data.loc[:, 'confidence'] = data_confidence
            return {
                'data': data,
            }

        combined_df = pd.concat(results)
        
        # Retain relevant columns for surface modeling (others should already be in data)
        df_surf = combined_df.loc[:, ['ph_index', 'surf_mu', 'surf_sigma', 'surf_A', 'kde_bandwidth', 'surf_left', 'surf_right']]

        # Average the surface model for duplicate photons from overlapping processing chunks
        df_surf = df_surf.groupby('ph_index').mean()
        
        # Merge surf_mu and surf_sigma into data
        data = data.merge(df_surf, left_on='ph_index', right_index=True, how='left')

        # Surface classification logic
        # Photons with surf_mu = nan are no-signal
        data_classification.loc[data.surf_mu.isna()] = 1
        
        # Noise (evaluated) = 1 (above N sigma)
        data_classification.loc[((data.z_ph - data.surf_mu) > (surf_sigma_cutoff * data.surf_sigma)) | (data.surf_mu.isna())] = 1
        # Noise = above surf right position
        # data_classification.loc[(data.z_ph > data.surf_right) & (~data.surf_mu.isna())] = 1
        
        # Water surface = 2
        data_classification.loc[(np.abs(data.z_ph - data.surf_mu) < surf_sigma_cutoff * data.surf_sigma) & (~data.surf_mu.isna())] = 2
        # within right/left surface positions
        # data_classification.loc[(data.z_ph < data.surf_right) & (data.z_ph > data.surf_left) & (~data.surf_mu.isna())] = 2
        
        # Subaqueous unclassified = 3
        data_classification.loc[(data_classification == 0) & (data.z_ph < data.surf_mu)] = 3
        # below surface left
        # data_classification.loc[(data.z_ph < data.surf_left) & (~data.surf_mu.isna())] = 3

        ############################## Bathymetry Modeling ##############################


        data_atmos_mask = (
            (data.z_ph > data.surf_mu + surf_sigma_cutoff*data.surf_sigma) &  
            (~data.surf_mu.isna())
        )

        data_subsurf_mask = (
            (data.z_ph < data.surf_mu - surf_sigma_cutoff*data.surf_sigma) &  
            (~data.surf_mu.isna())
        )

        nonsurface_mask = (data_atmos_mask | data_subsurf_mask)
        df_nonsurface = data[nonsurface_mask]

        atmos_mask = df_nonsurface.z_ph > df_nonsurface.surf_mu + surf_sigma_cutoff * df_nonsurface.surf_sigma
        subsurf_mask = df_nonsurface.z_ph < df_nonsurface.surf_mu - surf_sigma_cutoff * df_nonsurface.surf_sigma

        # Pseudo-z = height/depth relative to the surface model + x * sigma boundaries
        df_nonsurface.loc[atmos_mask, 'z_pseudo'] = (
            df_nonsurface[atmos_mask].z_ph - 
            df_nonsurface[atmos_mask].surf_mu -
            surf_sigma_cutoff * df_nonsurface[atmos_mask].surf_sigma
        )

        df_nonsurface.loc[subsurf_mask, 'z_pseudo'] = (
            df_nonsurface[subsurf_mask].z_ph - 
            df_nonsurface[subsurf_mask].surf_mu +
            surf_sigma_cutoff * df_nonsurface[subsurf_mask].surf_sigma
        )

        # # use left and right surface values instead
        # data_atmos_mask = (
        #     (data.z_ph > data.surf_right) &  
        #     (~data.surf_mu.isna())
        # )
        # data_subsurf_mask = (
        #     (data.z_ph < data.surf_left) &  
        #     (~data.surf_mu.isna())
        # )
        # nonsurface_mask = (data_atmos_mask | data_subsurf_mask)
        # df_nonsurface = data[nonsurface_mask]

        # atmos_mask = df_nonsurface.z_ph > df_nonsurface.surf_right
        # subsurf_mask = df_nonsurface.z_ph < df_nonsurface.surf_left

        # # Pseudo-z = height/depth relative to the surface model + boundaries
        # df_nonsurface.loc[atmos_mask, 'z_pseudo'] = (
        #     df_nonsurface[atmos_mask].z_ph - 
        #     df_nonsurface[atmos_mask].surf_right
        # )
        # df_nonsurface.loc[subsurf_mask, 'z_pseudo'] = (
        #     df_nonsurface[subsurf_mask].z_ph - 
        #     df_nonsurface[subsurf_mask].surf_left
        # )


        x_nonsurface = df_nonsurface.x_ph.values
        z_nonsurface = df_nonsurface.z_pseudo.values
        inds_nonsurface = df_nonsurface.index.values

        additional_chunk_data = {
            'subsurf_mask': subsurf_mask}
        
        # breakpoint()
        # print(subsurf_mask)

        # Break the non-surface data into chunks for processing
        try:
            if method_nonsurf_bins == "n":
                # nonsurface_chunks = n_chunk(
                #     x0=x_nonsurface, 
                #     y0=z_nonsurface, 
                #     points_per_chunk=n_ph_nonsurf_bins, 
                #     indices=inds_nonsurface, 
                #     overlap=overlap_nonsurf, 
                #     max_gap_size=max_gap_bins_m,
                #     additional_data=additional_chunk_data,
                #     mask=subsurf_mask.values
                # )
                # def mask_n_chunk(x0, y0, points_per_chunk, mask=None, indices=None, overlap=0.0, additional_data=None):

                nonsurface_chunks = mask_n_chunk(
                    x0=x_nonsurface, 
                    y0=z_nonsurface, 
                    points_per_chunk=n_ph_nonsurf_bins, 
                    mask=subsurf_mask.values,
                    indices=inds_nonsurface, 
                    overlap=overlap_nonsurf, 
                    max_gap_size=max_gap_bins_m,
                    additional_data=additional_chunk_data
                )

            elif method_nonsurf_bins == "x":
                nonsurface_chunks = x_chunk(
                    x0=x_nonsurface, 
                    y0=z_nonsurface,
                    bin_size=x_m_nonsurf_bins, 
                    indices=inds_nonsurface, 
                    overlap=overlap_nonsurf, 
                    max_gap_size=max_gap_bins_m,
                    additional_data=additional_chunk_data
                )
            else:
                raise ValueError(f"method_nonsurf_bins = {method_nonsurf_bins} not recognized. Use 'n' or 'x'.")
                
            if not nonsurface_chunks:
                logger.warning("No valid non-surface chunks found")
                data.loc[:, 'classification'] = data_classification
                data.loc[:, 'confidence'] = data_confidence
                return {
                    'data': data,
                }

            # # Debug plot all chunks
            # if debug:
            #     for i, chunk in enumerate(nonsurface_chunks):
            #         fig, ax = plt.subplots()
            #         ax.scatter(chunk.x, chunk.y, c='b', marker='o', s=1)
            #         ax.set_title(f"Chunk {i}")
            #         ax.set_xlabel("x (m)")
            #         ax.set_ylabel("z (m)")

            #         # add annotation with True/False counts for subsurf_mask
            #         subsurf_count = chunk.subsurf_mask.sum()
            #         atmos_count = len(chunk) - subsurf_count
            #         ax.annotate(f"Subsurf: {subsurf_count}\nAtmos: {atmos_count}", xy=(0.05, 0.95), 
            #                     xycoords='axes fraction', fontsize=10, ha='left', va='top',
            #                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

            #         plt.savefig(os.path.join(debug_dir, f"chunk_{i}.png"))
            #         plt.close(fig)
            
            # Process each non-surface chunk
            def process_nonsurface_chunk(chunk):
                return process_nonsurface_data(
                    chunk,
                    min_data_points=min_data_nonsurface,
                    bathy_snr_min=bathy_snr_min,
                    debug=debug,
                    debug_dir=debug_dir,
                    nighttime=nighttime,
                )
            
            # Process all non-surface chunks - Use serial or parallel based on force_serial
            if force_serial:
                nonsurface_results = [process_nonsurface_chunk(chunk) for chunk in nonsurface_chunks]
            else:
                try:
                    with ProcessPool(n_nodes) as pool:
                        nonsurface_results = pool.map(process_nonsurface_chunk, nonsurface_chunks)
                        
                except Exception as e:
                    logger.error(f"Error in parallel processing of non-surface chunks: {str(e)}")
                    nonsurface_results = [process_nonsurface_chunk(chunk) for chunk in nonsurface_chunks]
                    
            subsurf_results = [r['bathy_score_df'] for r in nonsurface_results]
            df_subsurf_signal = pd.concat(subsurf_results)

            # groupby 'i', compute mean score
            subsurf_score = df_subsurf_signal.groupby('i').mean().loc[:, ['score']]
            subsurf_count = df_subsurf_signal.groupby('i').count().iloc[:, 0]
            subsurf_count = subsurf_count.rename('count')

            # # where count < 3, set score = 0
            # # these have not been classified enough times to assign any true score
            subsurf_score.loc[subsurf_count < 3, 'score'] = 0

            # assign scores to data_confidence
            data_confidence.loc[subsurf_score.index] = subsurf_score['score']

            # set all evaluated points to subsurf_noise = 4
            data_classification.loc[subsurf_score.index] = 4

            # classify photons with majority votes as bathymetry
            bathy_inds = subsurf_score[subsurf_score.score >= 0.5].index
            data_classification.loc[bathy_inds] = 5
        
            ###### Refraction Correction and Depth Calculation ######
            df_subsurf = df_nonsurface[subsurf_mask].copy()

            df_subsurf['depth_uncorr'] = (df_subsurf.surf_mu - df_subsurf.z_ph)
            
            # Correct depth using refraction correction if ref_azimuth and ref_elev are present
            if 'ref_azimuth' in df_subsurf.columns and 'ref_elev' in df_subsurf.columns:
                try:
                    _, _, dZ = correction(0.0, -df_subsurf["depth_uncorr"].values, 
                                         df_subsurf['ref_azimuth'].values, df_subsurf['ref_elev'].values)
                    df_subsurf["depth"] = -(-df_subsurf["depth_uncorr"] + dZ)
                except Exception as e:
                    logger.warning(f"Refraction correction failed: {str(e)}")
                    df_subsurf["depth"] = df_subsurf["depth_uncorr"]
            else:
                df_subsurf["depth"] = df_subsurf["depth_uncorr"]
                logger.warning("No refraction correction applied, ref_azimuth and ref_elev not present in columns")
            
            # Convert depth to abs z
            df_subsurf.loc[:, 'z_ph_refr'] = df_subsurf.surf_mu - df_subsurf.depth

            # merge depth and depth_uncorr into data
            data = data.merge(
                df_subsurf[['ph_index', 'depth', 'depth_uncorr', 'z_ph_refr']],
                left_on='ph_index',
                right_on='ph_index',
                how='left'
            )

            ###### END of processing, return results
            data.loc[:, 'classification'] = data_classification
            data.loc[:, 'confidence'] = data_confidence
            return {
                'data': data,
            }
            
        except Exception as e:
            logger.error(f"Error in non-surface processing: {str(e)}")
            data.loc[:, 'classification'] = data_classification
            data.loc[:, 'confidence'] = data_confidence
            return {
                'data': data,
            }
        
    except Exception as e:
        logger.error(f"Fatal error in process_profile: {str(e)}")
        return None