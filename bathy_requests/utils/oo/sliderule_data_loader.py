import sliderule
from sliderule import icesat2, earthdata
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from .base_data_loader import BaseDataLoader
# from .ndwi import calculate_ndwi, initialize_earth_engine
from .preprocessor import Preprocessor
# from ..utils.logging_config import setup_logging
import logging
import threading

# Configure logging
# setup_logging(log_filename="data_loader.log", log_level=logging.INFO)

lock = threading.Lock()
earthdata.set_max_resources(2000)

class SlideRuleDataLoader(BaseDataLoader):
    def __init__(self, file_path=None, base_url="slideruleearth.io", verbose=True):
        super().__init__(file_path)
        # self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        # initialize_earth_engine()
        self.init_sliderule(base_url)

    def init_sliderule(self, base_url):
        loglevel = logging.DEBUG if self.verbose else logging.INFO
        sliderule.init(url=base_url, loglevel=loglevel)

    def query_sliderule(self, params, granules_list):
        # self.logger.debug(f"Query parameters: {params}")
        gdf = icesat2.atl03sp(params, resources=granules_list)
        # self.logger.debug(f"Received GeoDataFrame with {len(gdf)} records.")
        return gdf

    def load_data(
        self,
        region,
        start_date,
        end_date,
        min_conf=0,
        surface_type="SRT_DYNAMIC",
        calculate_ndwi_flag=False,
        parquet_path=None,
    ):
        with lock:
            region = sliderule.toregion(region)
        print(parquet_path)

        params = {
            "poly": region["poly"],
            "t0": f"{start_date}T00:00:00Z",
            "t1": f"{end_date}T23:59:59Z",
            "cnf": min_conf,
            "srt": getattr(sliderule.icesat2, surface_type),
            "atl03_geo_fields": ["ref_azimuth", "ref_elev", "geoid"],
            # "output": { "path": parquet_path, 
            #             "format": "geoparquet",
            #             # "as_geo": True,
            #             "open_on_complete": False,
            # },
        }

        granules_list = earthdata.cmr(
            short_name="ATL03",
            polygon=region["poly"],
            time_start=start_date,
            time_end=end_date,
            version="006",
        )

        print(f"Found {len(granules_list)} granules.")

        if not granules_list:
            # self.logger.warning("No granules found for the given AOI and time range.")
            return gpd.GeoDataFrame()

        # self.logger.debug(f"Granules found: {granules_list}")
        gdf = self.query_sliderule(params, granules_list)
        gdf = self.process_data(gdf)

        # if calculate_ndwi_flag:
        #     preprocessor = Preprocessor(gdf)
        #     separated_beams = preprocessor.separate_beams()

        #     results = []
        #     for key, profile_gdf in separated_beams.items():
        #         rgt, cycle, spot = key
        #         metadata = {"rgt": rgt, "cycle": cycle, "spot": spot}
        #         profile_gdf = calculate_ndwi(
        #             profile_gdf, metadata, start_date, end_date
        #         )
        #         results.append(profile_gdf)

        #     gdf = pd.concat(results, ignore_index=True)

        return gdf

    def process_data(self, gdf):
        if gdf.empty:
            return gdf

        gdf["time"] = gdf.index
        gdf.reset_index(drop=True, inplace=True)

        gdf["lat_ph"] = gdf["geometry"].apply(lambda p: p.y)
        gdf["lon_ph"] = gdf["geometry"].apply(lambda p: p.x)

        gdf["rgt"] = gdf["rgt"].astype(int)
        gdf["cycle"] = gdf["cycle"].astype(int)
        gdf["track"] = gdf["track"].astype(int)
        gdf["spot"] = gdf["spot"].astype(int)
        gdf["x_ph"] = gdf["x_atc"].astype(float) + gdf["segment_dist"].astype(float)
        gdf["geoid"] = gdf["geoid"].astype(float)
        gdf["z_ph"] = gdf["height"] - gdf["geoid"]
        gdf["ref_azimuth"] = gdf["ref_azimuth"].astype(float)
        gdf["ref_elev"] = gdf["ref_elev"].astype(float)
        gdf["solar_elevation"] = gdf["solar_elevation"].astype(float)
        gdf["ph_index"] = np.arange(len(gdf))  # Add unique photon index

        required_columns = {
            "time": np.nan,
            "segment_id": -1,
            "segment_dist_x": np.nan,
            # "classification": -1,
            # "confidence": -1,
            # "ndwi": 0,  # Add NDWI column with default value 0
        }

        for col, default in required_columns.items():
            if col not in gdf:
                gdf[col] = default

        gdf = gdf[
            [
                "segment_id",
                "rgt",
                "cycle",
                "track",
                "spot",
                "x_ph",
                "z_ph",
                "ref_azimuth",
                "ref_elev",
                "solar_elevation",
                "lon_ph",
                "lat_ph",
                # "classification",
                "atl03_cnf",
                # "confidence",
                "ph_index",  # Include the photon index in the final data
                # "ndwi",  # Include the NDWI column in the final data
                "time",
                "geometry",
            ]
        ]

        # time is in ns, we dont need that level of precision
        # convert to standard timestamp
        # gdf["time"] = pd.to_datetime(gdf["time"], unit="s")

        # convert time to str to avoid issues with parquet
        # YYYY-MM-DD HH:MM:SS, round to nearest second

        gdf["time"] = gdf["time"].dt.round("s").astype(str)


        # int_columns = ["segment_id", "rgt", "cycle", "track", "spot", "classification"]
        int_columns = ["segment_id", "rgt", "cycle", "track", "spot"]

        gdf = gdf.astype({col: int for col in int_columns})

        return gdf
