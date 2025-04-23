#!/usr/bin/env python3
"""
Batch processor for ICESat-2 bathymetry data.
Processes all profiles in a parquet file according to provided configuration.
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import geopandas as gpd
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import logging
import traceback

pd.options.mode.chained_assignment = None  # default='warn'

# Import utilities
from utils.preprocessor import Preprocessor
from utils.profile import process_profile
from utils.plotting import create_combined_plot, create_bathymetry_map
from utils.cmaps import cmap_classification
from utils.progess_monitor import ProgressMonitor

def setup_logging(output_dir):
    """Set up simplified logging configuration for critical updates only"""
    log_file = os.path.join(output_dir, "processing.log")
    
    # Create a simplified formatter
    basic_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler with basic formatter
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(basic_formatter)
    
    # Create console handler with the same basic formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(basic_formatter)
    
    # Configure root logger with WARNING level by default
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    root_logger.handlers = []  # Clear any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get main logger and set it to INFO for important process updates
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    return logger

def process_single_profile(profile_key, data, config, output_dir, debug_dir, debug=False):
    try:
        # Format output filename
        profile_date = data.time.iloc[0].strftime("%Y%m%d")
        key_str = f"{profile_date}_{profile_key[0]:04d}_{profile_key[1]:02d}_{profile_key[2]:01d}"
        output_parquet_path = os.path.join(output_dir, f"{key_str}.parquet")
        debug_plot_path = os.path.join(debug_dir, f"{key_str}.html")
        
        # Create profile-specific debug directory if debug is enabled
        profile_debug_dir = None
        if debug:
            profile_debug_dir = os.path.join(debug_dir, f"debug_{key_str}")
            os.makedirs(profile_debug_dir, exist_ok=True)
        
        # Pass the config dictionary directly to process_profile
        # Add any runtime parameters that aren't in the config file
        processing_kwargs = config.copy()
        processing_kwargs.update({
            'debug': debug,
            'debug_dir': profile_debug_dir,
            'serial': config.get('serial', False)
        })
        
        # Call process_profile with the data and expanded config dict
        results = process_profile(data, **processing_kwargs)

        # Handle results
        if results is None:
            raise ValueError("Error in profile processing (None Result)")

        # Save processed data
        data = results['data']
        data.to_parquet(output_parquet_path, index=False)

        # Create plot if enabled in config
        if config.get('make_plots', True):
            cmap, legend_labels = cmap_classification()
            fig_combined = create_combined_plot(data, cmap, legend_labels)
            fig_combined.write_html(debug_plot_path, include_plotlyjs='cdn')
        
        return key_str, True
    
    except Exception as e:
        # Simplified error handling with minimal logging
        error_message = f"Error processing profile {profile_key}: {str(e)}"
        return profile_key, error_message
        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process all profiles in an ICESat-2 data file.')
    parser.add_argument('input_file', nargs='?', help='Path to the input parquet file')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output-dir', help='Output directory (defaults to same as input with timestamp)')
    parser.add_argument('--profile-limit', type=int, help='Limit number of profiles to process (for testing)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional plotting')
    parser.add_argument('--serial', action='store_true', help='Run processing in serial mode (no parallel processing)')
    parser.add_argument('--rgt', type=int, help='Process only profiles with specific RGT value')
    parser.add_argument('--cc', type=int, help='Process only profiles with specific CC value')
    parser.add_argument('--spot', type=int, help='Process only profiles with specific SPOT value')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    # Override config with command line arguments where provided
    # Handle input file - command line takes precedence
    input_file = args.input_file or config.get('input_file')
    if not input_file:
        parser.error("input_file must be provided either as command line argument or in config file")
    
    # Handle output directory - command line takes precedence, then config, then default
    output_dir = args.output_dir or config.get('output_dir')
    if not output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        request_name = config.get('request_name', 'batch')
        output_dir = os.path.join(os.path.dirname(input_file), f"{request_name}_{timestamp}")
    
    # Handle other arguments - command line takes precedence over config
    debug = args.debug or config.get('debug', False)
    serial = args.serial or config.get('serial', False)
    rgt = args.rgt if args.rgt is not None else config.get('rgt')
    cc = args.cc if args.cc is not None else config.get('cc')
    spot = args.spot if args.spot is not None else config.get('spot')
    profile_limit = args.profile_limit if args.profile_limit is not None else config.get('profile_limit')
    
    # Update config with final values
    config['debug'] = debug
    config['serial'] = serial
    config['rgt'] = rgt
    config['cc'] = cc
    config['spot'] = spot
    config['profile_limit'] = profile_limit
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Set up simplified logging
    logger = setup_logging(output_dir)
    logger.info(f"Processing {input_file}")
    
    # Log if RGT, CC, or SPOT filter is being applied
    if rgt is not None:
        logger.info(f"Filtering for RGT: {rgt}")
    if cc is not None:
        logger.info(f"Filtering for CC: {cc}")
    if spot is not None:
        logger.info(f"Filtering for SPOT: {spot}")
    
    # Save the configuration used (with all runtime values)
    config_dest = os.path.join(output_dir, "runtime_config.yaml")
    with open(config_dest, 'w') as file:
        yaml.dump(config, file)
    
    try:
        # Load input data
        logger.info("Loading input data")
        input_data = gpd.read_parquet(input_file)
        input_data.time = pd.to_datetime(input_data.time)
        
        # Preprocess and separate into beams/profiles
        logger.info("Separating profiles")
        data_by_profile = Preprocessor(input_data).separate_beams()
        profiles_to_process = list(data_by_profile.keys())
        
        # Filter by RGT if specified
        if rgt is not None:
            profiles_to_process = [key for key in profiles_to_process if key[0] == rgt]
            logger.info(f"Found {len(profiles_to_process)} profiles with RGT {rgt}")
        
        # Filter by CC if specified
        if cc is not None:
            profiles_to_process = [key for key in profiles_to_process if key[1] == cc]
            logger.info(f"Found {len(profiles_to_process)} profiles with CC {cc}")
        
        # Filter by SPOT if specified
        if spot is not None:
            profiles_to_process = [key for key in profiles_to_process if key[2] == spot]
            logger.info(f"Found {len(profiles_to_process)} profiles with SPOT {spot}")
        
        # Limit profiles if requested
        if profile_limit:
            profiles_to_process = profiles_to_process[:profile_limit]
        
        logger.info(f"Processing {len(profiles_to_process)} profiles")
        
        # Check if there are no profiles to process
        if not profiles_to_process:
            logger.warning("No profiles found matching the specified criteria")
            return
        
        # Create progress monitor
        progress = ProgressMonitor(
            total_items=len(profiles_to_process),
            output_dir=output_dir,
            update_interval=2
        )
        
        # Process all profiles - serial or parallel based on args
        results = []
        
        if serial:
            # SERIAL PROCESSING
            logger.info("Using serial processing")
            for profile_key in profiles_to_process:
                progress.item_started()
                result = process_single_profile(
                    profile_key, 
                    data_by_profile[profile_key],
                    config,
                    output_dir,
                    debug_dir,
                    debug
                )
                results.append(result)
                success = not isinstance(result[1], str)
                progress.item_completed(success=success)
                
                if not success:
                    logger.error(f"Failed: {profile_key}")
        else:
            # PARALLEL PROCESSING
            logger.info(f"Using parallel processing ({config['n_nodes']} nodes)")
            with concurrent.futures.ProcessPoolExecutor(max_workers=config['n_nodes']) as executor:
                future_to_profile = {}
                
                for profile_key in profiles_to_process:
                    progress.item_started()
                    future = executor.submit(
                        process_single_profile, 
                        profile_key, 
                        data_by_profile[profile_key],
                        config,
                        output_dir,
                        debug_dir,
                        debug
                    )
                    future_to_profile[future] = profile_key
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_profile):
                    profile_key = future_to_profile[future]
                    try:
                        result = future.result()
                        results.append(result)
                        success = not isinstance(result[1], str)
                        progress.item_completed(success=success)
                        
                        if not success:
                            logger.error(f"Failed: {profile_key}")
                    except Exception as e:
                        logger.error(f"Exception: {profile_key}")
                        progress.item_completed(success=False)
        
        # Finalize progress when done
        progress.finalize()

        # Process output files and generate summary
        logger.info("Generating summary")
        parquet_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                        if f.endswith('.parquet') and not f.startswith('._') and not f.startswith('processing_summary')]

        if parquet_files:
            # Process output files and create summary
            bathy_gdfs = []
            summary_data = []
            
            for parquet_file in tqdm(parquet_files, desc="Writing results"):
                filename = os.path.basename(parquet_file)
                profile_id = filename.replace('.parquet', '')
                
                try:
                    # Load the processed data file
                    gdf = gpd.read_parquet(parquet_file)
                    
                    # Count points for summary
                    total_points = len(gdf)
                    subsurf_points = sum(gdf.classification >=4)
                    bathy_points = sum(gdf.classification == 5)
                    bathy_percent = round(100 * bathy_points / subsurf_points, 2) if subsurf_points > 0 else 0
                    
                    # Add to summary
                    summary_data.append({
                        'filename': filename,
                        'total_points': total_points,
                        'bathymetry_points': bathy_points,
                        'bathy_percentage': bathy_percent
                    })
                    
                    # Extract bathymetry data
                    bathy_data = gdf[gdf.classification == 5].copy()
                    if not bathy_data.empty:
                        bathy_data['profile_id'] = profile_id
                        bathy_gdfs.append(bathy_data)
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}")
                    summary_data.append({
                        'filename': filename,
                        'total_points': 'ERROR',
                        'bathymetry_points': 'ERROR',
                        'bathy_percentage': 'ERROR'
                    })
            
            # Save summary CSV
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, "summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            # Create and save combined bathymetry data if any exists
            if bathy_gdfs:
                try:
                    # Combine all bathymetry data
                    combined_bathy = pd.concat(bathy_gdfs, ignore_index=True)
                    
                    # Ensure it's a proper GeoDataFrame
                    if not isinstance(combined_bathy, gpd.GeoDataFrame):
                        combined_bathy = gpd.GeoDataFrame(combined_bathy, geometry='geometry')
                    
                    # Set CRS if missing
                    if combined_bathy.crs is None:
                        for gdf in bathy_gdfs:
                            if gdf.crs is not None:
                                combined_bathy.set_crs(gdf.crs, inplace=True)
                                break
                        if combined_bathy.crs is None:
                            combined_bathy.set_crs("EPSG:4326", inplace=True)
                    
                    # Save as GeoPackage
                    gpkg_path = os.path.join(output_dir, "bathymetry.gpkg")
                    combined_bathy.to_file(gpkg_path, driver="GPKG")

                    try:
                        map_created = create_bathymetry_map(output_dir)
                        if map_created:
                            logger.info("Bathymetry map created successfully")
                        else:
                            logger.warning("Failed to create bathymetry map")
                    except Exception as e:
                        logger.error(f"Error during map creation: {str(e)}")
                    
                    logger.info(f"Saved {len(combined_bathy)} bathymetry points")
                except Exception as e:
                    logger.error("Failed to save combined bathymetry data")
        
        # Final summary
        success_count = sum(1 for r in results if r[1] is True)
        logger.info(f"Complete: {success_count}/{len(profiles_to_process)} profiles successful")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)