#!/usr/bin/env python3
"""
ICESat-2 Data Processor for Bathymetry Requests

This script processes ICESat-2 data for specified areas of interest (AOI) and 
combines it with GEBCO bathymetry data and water masks for analysis.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import rioxarray
import xarray as xr
import threading
import concurrent.futures
import json
import datetime
from shapely.geometry import box
from utils.oo import sliderule_data_loader

# Configuration for all parameters
CONFIG = {
    # Required paths
    "aoi": "local/Oahu/Oahu.geojson",  # Path to the AOI geojson file
    "water_mask": "local/Oahu/DSWx/OPERA_L3_DSWX-HLS_V1_1.0-OAHU_merged_clipped.tif",  # Path to the water mask GeoTIFF
    "output_dir": "local/Oahu/ICESat2",  # Directory to store output files
    
    # Optional parameters
    "request_name": None,  # Name for this request (defaults to AOI filename if not provided)
    "aoi_title": None,  # Title string for the AOI (default: same as request name)
    
    # ICESat-2 query parameters
    "start_date": "2023-01-01", # ICESat-2 launch : Sept 2018
    "end_date": "2024-01-01",
    "min_confidence": 0,  # minimum signal_conf value
    "max_samples": int(1e5),  # max points to show in overhead view
    
    # Processing parameters
    "chunk_size": 5,  # chunk size in km
    "buffer_percent": 0,  # buffer percentage for chunks
    "max_workers": 8,  # maximum number of concurrent workers
    
    # Filtering parameters
    "min_elevation": -100,  # minimum photon elevation to include
    "max_elevation": 100,  # maximum photon elevation to include
    
    # GEBCO parameter
    "gebco_dir": "/Volumes/T7/gebco_2024_sub_ice_topo_geotiff"
}


def chunk_polygon(polygon_gdf, chunk_size_km=5, buffer_percent=0):
    """
    Subdivide a polygon GeoDataFrame into smaller chunks.
    
    Parameters:
    -----------
    polygon_gdf : GeoDataFrame
        A GeoDataFrame containing one or more polygons to be subdivided.
        Should have a valid CRS.
    chunk_size_km : float
        The approximate size of each chunk in kilometers.
    buffer_percent : float
        Optional buffer percentage (0-100) to add overlap between chunks.
        
    Returns:
    --------
    GeoDataFrame
        A GeoDataFrame containing the subdivided polygons, with original
        attributes copied to each chunk and an additional 'chunk_id' column.
    """
    # Ensure input is a GeoDataFrame with a valid CRS
    if not isinstance(polygon_gdf, gpd.GeoDataFrame):
        if isinstance(polygon_gdf, gpd.GeoSeries):
            polygon_gdf = gpd.GeoDataFrame(geometry=polygon_gdf)
        else:
            raise ValueError("Input must be a GeoDataFrame or GeoSeries")
    
    if polygon_gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS")
    
    # Convert to UTM for accurate distance calculations
    orig_crs = polygon_gdf.crs
    utm_gdf = polygon_gdf.to_crs(polygon_gdf.estimate_utm_crs())
    
    # Initialize an empty list to store all chunks
    all_chunks = []
    chunk_id = 0
    
    # Process each polygon in the GeoDataFrame
    for idx, row in utm_gdf.iterrows():
        geom = row.geometry
        
        # Get the bounds of the geometry in UTM coordinates (meters)
        minx, miny, maxx, maxy = geom.bounds
        
        # Calculate the number of cells in each direction
        # Convert km to meters for the grid
        cell_size = chunk_size_km * 1000
        
        # Calculate number of cells in x and y directions
        nx = int(np.ceil((maxx - minx) / cell_size))
        ny = int(np.ceil((maxy - miny) / cell_size))
        
        # Ensure at least one cell
        nx = max(1, nx)
        ny = max(1, ny)
        
        # Calculate the actual cell size to exactly cover the extent
        cell_size_x = (maxx - minx) / nx
        cell_size_y = (maxy - miny) / ny
        
        print(f"Subdividing polygon into {nx}x{ny} = {nx*ny} chunks")
        
        # Create grid cells
        for i in range(nx):
            for j in range(ny):
                # Calculate cell boundaries
                cell_minx = minx + i * cell_size_x
                cell_miny = miny + j * cell_size_y
                cell_maxx = minx + (i + 1) * cell_size_x
                cell_maxy = miny + (j + 1) * cell_size_y
                
                # Apply buffer if requested (as a percentage of cell size)
                if buffer_percent > 0:
                    buffer_x = (cell_size_x * buffer_percent) / 100.0
                    buffer_y = (cell_size_y * buffer_percent) / 100.0
                    cell_minx = max(minx, cell_minx - buffer_x)
                    cell_miny = max(miny, cell_miny - buffer_y)
                    cell_maxx = min(maxx, cell_maxx + buffer_x)
                    cell_maxy = min(maxy, cell_maxy + buffer_y)
                
                # Create a box for this cell
                cell = box(cell_minx, cell_miny, cell_maxx, cell_maxy)
                
                # Intersect with the original geometry to handle irregular shapes
                if cell.intersects(geom):
                    chunk = cell.intersection(geom)
                    
                    # Skip tiny slivers
                    if chunk.area < 1:  # Skip areas smaller than 1 square meter
                        continue
                    
                    # Copy attributes from the original polygon
                    chunk_attrs = row.to_dict()
                    chunk_attrs.pop('geometry')  # Remove the geometry as we'll replace it
                    
                    # Add chunk metadata
                    chunk_attrs['chunk_id'] = chunk_id
                    chunk_attrs['chunk_i'] = i
                    chunk_attrs['chunk_j'] = j
                    chunk_attrs['original_id'] = idx
                    
                    # Add to our collection
                    chunk_attrs['geometry'] = chunk
                    all_chunks.append(chunk_attrs)
                    chunk_id += 1
    
    # Create a new GeoDataFrame from all chunks
    if all_chunks:
        chunks_gdf = gpd.GeoDataFrame(all_chunks, crs=utm_gdf.crs)
        
        # Convert back to the original CRS
        chunks_gdf = chunks_gdf.to_crs(orig_crs)
        return chunks_gdf
    else:
        # Return an empty GeoDataFrame with the same structure
        empty_gdf = gpd.GeoDataFrame([], crs=orig_crs)
        for col in polygon_gdf.columns:
            if col != 'geometry':
                empty_gdf[col] = []
        empty_gdf['chunk_id'] = []
        empty_gdf['chunk_i'] = []
        empty_gdf['chunk_j'] = []
        empty_gdf['original_id'] = []
        return empty_gdf


def process_icesat2_data(run_timestamp):
    """Process ICESat-2 data for the given AOI."""
    # Set up request name if not provided
    if CONFIG["request_name"] is None:
        CONFIG["request_name"] = os.path.splitext(os.path.basename(CONFIG["aoi"]))[0]
    
    # Replace spaces with underscores in request name
    CONFIG["request_name"] = CONFIG["request_name"].replace(" ", "_")
    
    # Set up AOI title if not provided
    if CONFIG["aoi_title"] is None:
        CONFIG["aoi_title"] = CONFIG["request_name"]
    
    print(f"Processing request: {CONFIG['request_name']}")
    
    # Create timestamped output directory
    request_dir = os.path.join(CONFIG["output_dir"], CONFIG["request_name"])
    if not os.path.exists(request_dir):
        os.makedirs(request_dir)
    
    # Create a timestamped processing directory
    processing_dir = os.path.join(request_dir, f"{CONFIG["request_name"]}_{run_timestamp}")
    if not os.path.exists(processing_dir):
        os.makedirs(processing_dir)
    
    # Create chunks directory
    chunks_dir = os.path.join(processing_dir, "chunks")
    if not os.path.exists(chunks_dir):
        os.makedirs(chunks_dir)
    
    # Save the configuration
    config_to_save = CONFIG.copy()
    config_to_save["processing_timestamp"] = run_timestamp
    
    config_path = os.path.join(processing_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_to_save, f, indent=2)
    
    print(f"Config saved to: {config_path}")
    
    # Define the output parquet path
    output_parquet_path = os.path.join(processing_dir, f"{CONFIG['request_name']}_icesat2_atl03.parquet")
    
    # Load and format the AOI dataframe
    aoi = gpd.read_file(CONFIG["aoi"])
    aoi["start_date"] = pd.Timestamp(CONFIG["start_date"])
    aoi["end_date"] = pd.Timestamp(CONFIG["end_date"])
    aoi["request_name"] = CONFIG["request_name"]
    aoi["path_to_spatial_boundary"] = CONFIG["aoi"]
    aoi["start_date"] = aoi["start_date"].dt.tz_localize('UTC')
    aoi["end_date"] = aoi["end_date"].dt.tz_localize('UTC')
    
    # Get the GEBCO filename based on the AOI
    aoi_utm = aoi.to_crs(aoi.estimate_utm_crs())
    aoi_centroid = aoi_utm.centroid.to_crs(4326)
    x_centroid = aoi_centroid.x.values[0]
    y_centroid = aoi_centroid.y.values[0]

    ns_str = "n90.0_s0.0" if y_centroid > 0 else "n0.0_s-90.0"
    w_str = int(np.floor(x_centroid/90) * 90)
    e_str = int(np.ceil(x_centroid/90) * 90)
    gebco_fname = f"gebco_2024_sub_ice_{ns_str}_w{w_str:.1f}_e{e_str:.1f}.tif"
    print(f"GEBCO filename: {gebco_fname}")

    # Path to GEBCO file
    path_to_gebco = os.path.join(CONFIG["gebco_dir"], gebco_fname)
    
    # Convert to bounding box if needed
    bounds = aoi.total_bounds
    aoi_bbox = box(*bounds)
    
    # Replace geometry with bounding box
    aoi.geometry = [aoi_bbox]
    
    # Make sure we're in UTM, then buffer by specified amount
    # aoi = aoi.to_crs(aoi.estimate_utm_crs())
    # aoi = aoi.buffer(CONFIG["chunk_size"] * 1000 * (1 + CONFIG["buffer_percent"]), join_style=2)
    
    # Convert back to WGS84
    # aoi = aoi.to_crs(4326)
    
    # Create chunks of the AOI
    aoi_chunks = chunk_polygon(aoi, chunk_size_km=CONFIG["chunk_size"], buffer_percent=CONFIG["buffer_percent"])
    
    # Save the chunks for reference
    chunks_path = os.path.join(chunks_dir, "aoi_chunks.gpkg")
    aoi_chunks.to_file(chunks_path, driver="GPKG")
    
    # Define function to process a single chunk
    def process_chunk(chunk_idx):
        chunk_row = aoi_chunks.iloc[chunk_idx]
        chunk_id = chunk_row['chunk_id']
        
        # Create a filename for this chunk's results
        chunk_path = os.path.join(chunks_dir, f"chunk_{chunk_id}.parquet")
        
        print(f"Processing chunk {chunk_id} ({chunk_idx+1}/{aoi_chunks.shape[0]})")
        
        # Create a GeoDataFrame for this chunk
        chunk_gdf = gpd.GeoDataFrame([chunk_row], geometry='geometry', crs=aoi.crs)
        
        try:
            # Create a separate loader instance for thread safety
            chunk_loader = sliderule_data_loader.SlideRuleDataLoader(verbose=False)
            
            # Load data for this chunk
            chunk_data = chunk_loader.load_data(
                region=chunk_gdf,
                start_date=CONFIG["start_date"],
                end_date=CONFIG["end_date"],
                min_conf=CONFIG["min_confidence"],
            )
            
            if chunk_data is not None and len(chunk_data) > 0:
                print(f"  Chunk {chunk_id}: Found {len(chunk_data)} points")
                
                # Add chunk ID to the data
                chunk_data['chunk_id'] = chunk_id
                
                # Save chunk data
                chunk_data.to_parquet(chunk_path)
            else:
                print(f"  Chunk {chunk_id}: No data found")
                # Save empty GeoDataFrame to mark chunk as processed
                empty_gdf = gpd.GeoDataFrame(geometry=[], crs=aoi.crs)
                empty_gdf['chunk_id'] = []
                empty_gdf.to_parquet(chunk_path)
            
            return chunk_path
                
        except Exception as e:
            print(f"  Error processing chunk {chunk_id}: {e}")
            return None
    
    # Process chunks in parallel using ThreadPoolExecutor
    # Determine number of workers
    max_workers = min(aoi_chunks.shape[0], CONFIG["max_workers"])
    print(f"Processing {aoi_chunks.shape[0]} chunks with {max_workers} parallel workers")
    
    chunk_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_chunk, idx): idx 
            for idx in range(len(aoi_chunks))
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                chunk_path = future.result()
                if chunk_path:
                    chunk_paths.append(chunk_path)
            except Exception as e:
                print(f"Chunk {aoi_chunks.iloc[chunk_idx]['chunk_id']} generated an exception: {e}")
    
    # Now merge all the chunk files
    print(f"Merging {len(chunk_paths)} chunk datasets")
    
    # Initialize an empty list to store chunk data
    gdf_parts = []
    
    # Load and merge all chunk files
    for chunk_path in chunk_paths:
        try:
            if os.path.exists(chunk_path):
                chunk_gdf = gpd.read_parquet(chunk_path)
                if len(chunk_gdf) > 0:  # Only add non-empty datasets
                    gdf_parts.append(chunk_gdf)
        except Exception as e:
            print(f"  Error reading {chunk_path}: {e}")
    
    # Merge all GeoDataFrames
    if gdf_parts:
        gdf = pd.concat(gdf_parts)
        print(f"Total data points after merging all chunks: {len(gdf)}")
        
        # Clip to only data within the aoi
        gdf = gpd.clip(gdf, aoi.to_crs(gdf.crs))

        # Remove duplicate points from overlapping chunks if needed
        if CONFIG["buffer_percent"] > 0:
            print("Removing potential duplicate points from chunk overlaps...")
            # Use drop_duplicates on geometry to remove exact duplicates
            original_count = len(gdf)
            gdf = gdf.drop_duplicates(subset=['geometry'])
            print(f"  Removed {original_count - len(gdf)} duplicate points")
    else:
        print("No data found in any chunk")
        gdf = gpd.GeoDataFrame(geometry=[], crs=aoi.crs)
    
    # Apply elevation range filtering
    if CONFIG["min_elevation"] is not None:
        gdf = gdf[gdf.z_ph > CONFIG["min_elevation"]]
        
    if CONFIG["max_elevation"] is not None:
        gdf = gdf[gdf.z_ph < CONFIG["max_elevation"]]
    
    # Prepare for sampling from rasters
    x_sample_points = xr.DataArray(gdf.geometry.x, dims='points')
    y_sample_points = xr.DataArray(gdf.geometry.y, dims='points')
    
    # Sample GEBCO bathymetry at ICESat-2 points
    try:
        if os.path.exists(path_to_gebco):
            gebco = rioxarray.open_rasterio(path_to_gebco, chunks=True)
            gebco_samples = gebco.sel(x=x_sample_points, y=y_sample_points, method='nearest').values.flatten()
            gdf['gebco_depth'] = gebco_samples
            print("Added GEBCO depth samples")
        else:
            print(f"GEBCO file not found: {path_to_gebco}")
            gdf['gebco_depth'] = np.nan
    except Exception as e:
        print(f"Error sampling GEBCO data: {e}")
        gdf['gebco_depth'] = np.nan
    
    # Sample water mask at ICESat-2 points
    try:
        water_mask = rioxarray.open_rasterio(CONFIG["water_mask"], chunks=True).squeeze().rio.reproject(gdf.crs)
        water_mask_samples = water_mask.sel(x=x_sample_points, y=y_sample_points, method='nearest').values.flatten().astype(bool)
        gdf['water_mask'] = water_mask_samples
        
        # Any nans (outside bounds) should be assigned as land
        gdf.water_mask = gdf.water_mask.fillna(False)
        
        # Print water percentage
        water_pct = gdf.water_mask.sum() / len(gdf) * 100
        print(f"Water percentage: {water_pct:.2f}%")
    except Exception as e:
        print(f"Error sampling water mask: {e}")
        gdf['water_mask'] = True
    
    # Save the final dataset with auxiliary data
    gdf.to_parquet(output_parquet_path)
    # print(f"Saved final dataset to: {output_parquet_path}")

    # Save to gpkg for GIS compatibility next to parquet
    gpkg_path = os.path.splitext(output_parquet_path)[0] + ".gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    print(f"Final dataset saved to: {output_parquet_path}/.gpkg")
    
    
    return output_parquet_path


def main():
    """Main function."""
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate inputs
    if not os.path.exists(CONFIG["aoi"]):
        print(f"Error: AOI file not found: {CONFIG['aoi']}")
        sys.exit(1)
    
    if not os.path.exists(CONFIG["water_mask"]):
        print(f"Error: Water mask file not found: {CONFIG['water_mask']}")
        sys.exit(1)
    
    # Process the data
    output_path = process_icesat2_data(timestamp)
    
    print("="*80)
    print("Processing completed successfully!")
    print(f"Output saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()