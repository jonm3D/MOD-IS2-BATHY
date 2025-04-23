#!/usr/bin/env python
"""
DSWx Mosaic Generator

This script creates a merged mosaic from DSWx-HLS data, optionally clipping it to a specified boundary.
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import tqdm
import rioxarray
from rioxarray.merge import merge_arrays
import geopandas as gpd
import rasterio


def create_dswx_mosaic(input_dir, output_path, bbox_path=None):
    """
    Create a merged mosaic from DSWx-HLS data.
    
    Parameters:
    -----------
    input_dir : str
        Path to directory containing DSWx-HLS data
    output_path : str
        Path where the output mosaic will be saved
    bbox_path : str, optional
        Path to GeoJSON file containing the bounding box to clip to
    
    Returns:
    --------
    dict
        Dictionary containing paths to the created files
    """
    # Get list of files
    pattern = "*BWTR.tif"
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")
    files.sort()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Temporary file path
    tmp_path = os.path.join(os.path.dirname(output_path), '.tmp.tif')
    
    # Load colormap from the first file using rasterio
    with rasterio.open(files[0]) as src:
        cmap = src.colormap(1)
        crs = src.crs
    print(f"CRS: {crs}")

    # Load and process data
    print("Loading and processing data...")
    data = []
    for cog in tqdm.tqdm(files):
        # Using masked=True 
        cog_data = rioxarray.open_rasterio(cog)
        # Replace missing data with nodata
        cog_data = cog_data.where(cog_data < 252, cog_data.rio.nodata)
        data.append(cog_data)

    # Compute final scores based on most common value (water or non water)
    print("Merging data...")
    merged_sum = merge_arrays(data, method='sum')
    merged_count = merge_arrays(data, method='count')
    merged_mean = merged_sum / merged_count
    
    # Round to 0 or 1 and convert to int to avoid writing many GBs
    merged = merged_mean.round().astype(np.uint8)
    
    # Write to disk
    print(f"Writing merged output to {output_path}...")
    merged.rio.to_raster(tmp_path)
    
    with rasterio.open(tmp_path) as src:
        vals = src.read(1)
        meta = src.meta
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(vals, 1)
        dst.write_colormap(1, cmap)
    
    results = {'merged': output_path}
    
    # Clip to bounding box if provided
    if bbox_path:
        # Create a default clip output filename
        clip_output = output_path.replace('.tif', '_clipped.tif')
        
        print(f"Clipping to bounding box from {bbox_path}...")
        
        # Read bbox
        bbox = gpd.read_file(bbox_path).to_crs(merged.rio.crs).geometry
        
        # Set nodata to 255 (uint8) to avoid the default unmasked rasterio behavior
        merged.rio.write_nodata(255, inplace=True)
        merged = merged.rio.clip(bbox, drop=True)
        
        # Write to disk with colormap
        merged.rio.to_raster(tmp_path)
        
        with rasterio.open(tmp_path) as src:
            vals = src.read(1)
            meta = src.meta
        
        with rasterio.open(clip_output, 'w', **meta) as dst:
            dst.write(vals, 1)
            dst.write_colormap(1, cmap)
        
        results['clipped'] = clip_output
    
    # Clean up temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Create a merged mosaic from DSWx-HLS data.')
    parser.add_argument('input_dir', help='Directory containing DSWx-HLS data')
    parser.add_argument('output_path', help='Path where the output mosaic will be saved (file or directory)')
    parser.add_argument('--bbox', '-b', dest='bbox_path', help='Path to GeoJSON file for clipping')
    
    args = parser.parse_args()
    
    try:
        # Handle the case where output_path is a directory
        output_path = args.output_path
        if os.path.isdir(output_path):
            # Extract a default filename from the input directory
            input_name = os.path.basename(os.path.normpath(args.input_dir))
            output_path = os.path.join(output_path, f"{input_name}_merged.tif")
            print(f"Output path is a directory. Using filename: {output_path}")
        
        results = create_dswx_mosaic(
            args.input_dir,
            output_path,
            args.bbox_path
        )
        
        print("\nProcessing complete!")
        print(f"Merged output: {results['merged']}")
        if 'clipped' in results:
            print(f"Clipped output: {results['clipped']}")
    
    except Exception as e:
        parser.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()