"""ATL24 ICESat-2 Processing Utility"""

import sliderule
from sliderule import toregion
import geopandas as gpd
import contextily as ctx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
import os
import datetime
import random
import warnings
import xarray as xr
import shapely

warnings.filterwarnings("ignore", category=UserWarning)


def get_bathy_photons(
    path_to_aoi,
    output_path='.',
    threshold_conf=0.9,
    t0=None,
    t1=None,
    anc_fields=["ellipse_h", 'surface_h'],
    class_ph = ["bathymetry"],
    plot_quicklook=True,
):
    """Process ATL24 data for given AOI."""
    
    # Initialize
    sliderule.init()

    # Convert dates to datetime
    if isinstance(t0, str):
        t0 = datetime.datetime.strptime(t0, '%Y-%m-%d')
    if isinstance(t1, str):
        t1 = datetime.datetime.strptime(t1, '%Y-%m-%d')
    
    # Defaults
    if t0 is None:
        t0 = datetime.datetime(2018, 9, 1)
    if t1 is None:
        t1 = datetime.datetime.now()
    path_to_supplemental_rasters = [
        "/Volumes/AQUARIUS/IS2_Retreivability_03_RetreivabilityIndex/ETOPO/ETOPO_30s_1_m41.tif",
        "/Volumes/AQUARIUS/IS2_Retreivability_03_RetreivabilityIndex/RetreivabilityIndex_SecchiDepth1p25.tif"
    ]
    
    # File naming
    aoi_name = os.path.splitext(os.path.basename(path_to_aoi))[0]
    file_basename = f"ATL24_{aoi_name.upper()}_{t0.strftime('%Y%m%d')}_{t1.strftime('%Y%m%d')}_{int(threshold_conf*10)}"
    # print(file_basename)
    
    output_gpkg_path = f"{output_path}/{file_basename}.gpkg"
    output_plot_path = f"{output_path}/{file_basename}_quicklook.png"

    # load and return if already exists
    # if os.path.exists(output_gpkg_path) and os.path.exists(output_plot_path):
    #     return gpd.read_file(output_gpkg_path)
    
    # Load AOI
    gdf_aoi = gpd.read_file(path_to_aoi)

    # require bbox geoms (no fancy polygons
    # convert with shapely and total bounds
    aoi = toregion(path_to_aoi)

    aoi_km2 = int(gdf_aoi.to_crs(gdf_aoi.estimate_utm_crs()).area.sum() / 1e6)
    
    # Load supplemental rasters
    supp = {}
    if os.path.exists(path_to_supplemental_rasters[0]):
        supp['z_etopo'] = (rioxarray.open_rasterio(path_to_supplemental_rasters[0], mask_and_scale=True)
                          .squeeze()
                          .rio.clip_box(*gdf_aoi.total_bounds, crs=gdf_aoi.crs)
                          .rio.clip(gdf_aoi.geometry, crs=gdf_aoi.crs, drop=True)
                          .rio.reproject(4326))
    if os.path.exists(path_to_supplemental_rasters[1]):
        supp['z_SD'] = (rioxarray.open_rasterio(path_to_supplemental_rasters[1], mask_and_scale=True)
                       .squeeze()
                       .rio.clip_box(*gdf_aoi.total_bounds, crs=gdf_aoi.crs)
                       .rio.clip(gdf_aoi.geometry, crs=gdf_aoi.crs, drop=True)
                       .rio.reproject(4326))
    
    # SlideRule query
    print(f"Querying {aoi_km2} km2 for {file_basename}...")
    parms = {
        "atl24": {
            'anc_fields': anc_fields,
            'confidence_threshold': threshold_conf,
            'class_ph': class_ph
        },
        # times formatted like '2019-11-15T00:00:00Z'
        't0': t0.strftime('%Y-%m-%dT%H:%M:%SZ'),
        't1': t1.strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    
    gdf1 = sliderule.run("atl24x", parms, aoi=aoi['poly'])
    
    # Process dataframe
    if gdf1.index.name == 'time_ns':
        gdf1 = gdf1.reset_index()
        gdf1.rename({"time_ns": "time"}, axis=1, inplace=True)
    
    gdf1.sort_values(by=["time", "rgt", "cycle", "spot"], inplace=True)
    gdf1.reset_index(drop=True, inplace=True)
    
    # Add columns
    gdf1['lon'] = gdf1.geometry.x
    gdf1['lat'] = gdf1.geometry.y
    gdf1['BEAM_ID'] = gdf1.apply(
        lambda x: f"{x['time'].strftime('%Y%m%d')}_{x['rgt']:04}_{x['cycle']:02}_{x['spot']:02}", 
        axis=1
    )
    
    # Renormalize along-track distance
    for beam_id in gdf1['BEAM_ID'].unique():
        mask = gdf1['BEAM_ID'] == beam_id
        min_x_atc = gdf1.loc[mask, 'x_atc'].min()
        gdf1.loc[mask, 'x_atc'] -= min_x_atc
    
    # Create quicklook plot
    if plot_quicklook:
        print(f"Creating quicklook plot for {file_basename}...")
        
        gdf_plot = gdf1.sort_values('ortho_h', ascending=False)
        beam_colors = plt.cm.get_cmap('viridis', len(gdf_plot.BEAM_ID.unique()))
        beam_colors = random.sample(list(beam_colors.colors), len(beam_colors.colors))
        
        f, ax = plt.subplots(2, 2, figsize=(15, 10))
        
        # Map
        ax_map = ax[0, 0]
        gdf_aoi.plot(ax=ax_map, facecolor='none', edgecolor='white')
        # colorbar label of 'depth'
        sc = gdf_plot.plot.scatter(x='lon', y='lat', ax=ax_map, alpha=0.5, s=5, 
                            marker='.', c=gdf_plot.ortho_h, cmap='plasma')

        # # Add colorbar with label
        # cbar = plt.colorbar(sc, ax=ax_map)
        # cbar.set_label('Depth', rotation=270, labelpad=15)

        ctx.add_basemap(ax_map, crs=gdf_aoi.crs.to_string(), 
                        source=ctx.providers.Esri.WorldImagery, attribution=False)
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        
        # Beam pie chart
        ax_beam_makeup = ax[0, 1]
        beam_id_value_counts = gdf1['BEAM_ID'].value_counts()
        beam_id_value_counts.plot.pie(ax=ax_beam_makeup, labels=None, colors=beam_colors)
        ax_beam_makeup.set_ylabel('')
        ax_beam_makeup.set_title('Beam Composition')
        
        # Profile
        ax_profile = ax[1, 0]
        for i, beam_id in enumerate(gdf1['BEAM_ID'].unique()):
            beam_data = gdf1[gdf1['BEAM_ID'] == beam_id]
            beam_data.plot.scatter(x='x_atc', y='ortho_h', ax=ax_profile, alpha=0.5, s=5, 
                                  marker='.', color=beam_colors[i], label=None)
        ax_profile.set_xlabel('Along Track Distance (m)')
        ax_profile.set_ylabel('Orthometric Height (m)')
        
        # Histogram
        ax_hist = ax[1, 1]
        gdf1['confidence'].plot.hist(ax=ax_hist, bins=50, alpha=0.5, orientation='vertical')
        ax_hist.set_xlabel('Confidence')
        ax_hist.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=300)
    
    # Sample supplemental rasters
    if supp:
        is2_sample_x = xr.DataArray(gdf1.geometry.x, dims=["points"])
        is2_sample_y = xr.DataArray(gdf1.geometry.y, dims=["points"])
        
        for key, da in supp.items():
            gdf1.loc[:, key] = da.sel(x=is2_sample_x, y=is2_sample_y, method="nearest").values
    
    # Save outputs
    gdf1.to_file(output_gpkg_path, driver="GPKG")
    
    utm_crs = gdf_aoi.estimate_utm_crs()
    output_xyz_path = f"{output_path}/{file_basename}_{utm_crs.to_epsg()}_EGM08.xyz"
    
    gdf_xyz = gdf1.to_crs(utm_crs)
    gdf_xyz['x_utm'] = gdf_xyz.geometry.x
    gdf_xyz['y_utm'] = gdf_xyz.geometry.y
    gdf_xyz = gdf_xyz[['x_utm', 'y_utm', 'ortho_h']]
    gdf_xyz.to_csv(output_xyz_path, sep=' ', header=False, index=True)
    print(f"XYZ saved to {output_xyz_path}")

    print(f"GPKG saved to {output_gpkg_path}")
    return gdf1