import os
import tempfile
import pandas as pd
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def initialize_earth_engine():
    """
    Initialize Google Earth Engine.
    """
    try:
        ee.Initialize()
    except ee.EEException as e:
        print(f"Earth Engine initialization error: {e}")
        ee.Authenticate()
        ee.Initialize()
    except Exception as e:
        print(f"Unexpected error during Earth Engine initialization: {e}")


def calculate_ndwi(data, start_date="2021-01-01", end_date="2021-12-31"):
    """
    Calculate NDWI for the given data using Google Earth Engine.

    Parameters:
    data (pd.DataFrame): DataFrame containing the geospatial data.
    info (dict): Dictionary containing metadata information.
    start_date (str): Start date for NDWI calculation.
    end_date (str): End date for NDWI calculation.

    Returns:
    pd.DataFrame: DataFrame with added NDWI values.
    """
    output_file_name = "ndwi.csv"

    # Use a temporary directory for storing the NDWI file
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_file_path = os.path.join(tmpdirname, output_file_name)

        df_gee = pd.DataFrame(index=data.index)
        df_gee["x"] = data.geometry.x.round(4)
        df_gee["y"] = data.geometry.y.round(4)
        df_gee["segment_id"] = data.segment_id
        df_gee = df_gee.groupby(["segment_id"]).median().reset_index()
        df_gee.reset_index(inplace=True)

        segment_rate_dicts = df_gee.to_dict(orient="records")
        sampling_points_list = [
            ee.Feature(ee.Geometry.Point(d["x"], d["y"]), {"index": d["index"]})
            for d in segment_rate_dicts
        ]
        sampling_points_fc = ee.FeatureCollection(sampling_points_list)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        )
        s2_image = collection.median()
        s2_ndwi = s2_image.normalizedDifference(["B3", "B8"])

        geemap.zonal_stats(
            s2_ndwi,
            sampling_points_fc,
            tmp_file_path,
            statistics_type="MEDIAN",
            scale=10,  # Full resolution for Sentinel-2 is 10 meters
            return_fc=False,
        )

        df_ndwi = (
            pd.read_csv(tmp_file_path)
            .set_index("index")
            .drop(columns="system:index")
            .rename(columns={"median": "ndwi"})
        )
        df_gee = df_gee.merge(df_ndwi, left_index=True, right_index=True)
        data_with_ndwi = data.drop(columns=["ndwi"], errors="ignore").merge(
            df_gee[["segment_id", "ndwi"]], on="segment_id", how="left"
        )

    return data_with_ndwi
