import h5py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from .base_data_loader import BaseDataLoader
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .ndwi import calculate_ndwi, initialize_earth_engine
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from bokeh.plotting import figure, save, show
# from bokeh.tile_providers import get_provider
from bokeh.models import Arrow, NormalHead, LinearAxis, Range1d
from bokeh.layouts import row, column
import pyproj
import matplotlib.pyplot as plt
import xyzservices.providers as xyz


class HDF5DataLoader(BaseDataLoader):
    def __init__(self, file_path):
        self.file_path = file_path
        initialize_earth_engine()

    def load_heights(self, file, beam):
        return {
            "h_ph": file[f"{beam}/heights/h_ph"][:],
            "lat_ph": file[f"{beam}/heights/lat_ph"][:],
            "lon_ph": file[f"{beam}/heights/lon_ph"][:],
            "dist_ph_along": file[f"{beam}/heights/dist_ph_along"][:],
            "delta_time": file[f"{beam}/heights/delta_time"][:],
        }

    def load_geolocation(self, file, beam):
        return {
            "segment_dist_x": file[f"{beam}/geolocation/segment_dist_x"][:],
            "segment_id": file[f"{beam}/geolocation/segment_id"][:],
            "segment_ph_cnt": file[f"{beam}/geolocation/segment_ph_cnt"][:],
            "ref_azimuth": file[f"{beam}/geolocation/ref_azimuth"][:],
            "ref_elev": file[f"{beam}/geolocation/ref_elev"][:],
            "solar_elevation": file[f"{beam}/geolocation/solar_elevation"][:],
        }

    def load_geophys_corr(self, file, beam):
        return {"geoid": file[f"{beam}/geophys_corr/geoid"][:],
                "dem_h": file[f"{beam}/geophys_corr/dem_h"][:]}

    def load_metadata(self, file, beam):
        atlas_epoch = file["/ancillary_data/atlas_sdp_gps_epoch"][:][0]
        rgt = file["/orbit_info/rgt"][0]
        cycle = file["/orbit_info/cycle_number"][0]
        release = file["/ancillary_data/release"][0].decode("UTF-8").strip()
        sc_orient = file["/orbit_info/sc_orient"][0]
        spot = self.load_beam_info(file, beam)["atlas_spot"]
        return {
            "atlas_epoch": atlas_epoch,
            "rgt": rgt,
            "cycle": cycle,
            "release": release,
            "sc_orient": sc_orient,
            "path": self.file_path,
            "spot": spot,
        }

    def load_beam_info(self, file, beam):
        beam_info = {
            0: {
                "gt1l": {"beam_strength": "strong", "atlas_spot": 1, "track_pair": 1},
                "gt1r": {"beam_strength": "weak", "atlas_spot": 2, "track_pair": 1},
                "gt2l": {"beam_strength": "strong", "atlas_spot": 3, "track_pair": 2},
                "gt2r": {"beam_strength": "weak", "atlas_spot": 4, "track_pair": 2},
                "gt3l": {"beam_strength": "strong", "atlas_spot": 5, "track_pair": 3},
                "gt3r": {"beam_strength": "weak", "atlas_spot": 6, "track_pair": 3},
            },
            1: {
                "gt1l": {"beam_strength": "weak", "atlas_spot": 6, "track_pair": 1},
                "gt1r": {"beam_strength": "strong", "atlas_spot": 5, "track_pair": 1},
                "gt2l": {"beam_strength": "weak", "atlas_spot": 4, "track_pair": 2},
                "gt2r": {"beam_strength": "strong", "atlas_spot": 3, "track_pair": 2},
                "gt3l": {"beam_strength": "weak", "atlas_spot": 2, "track_pair": 3},
                "gt3r": {"beam_strength": "strong", "atlas_spot": 1, "track_pair": 3},
            },
        }

        sc_orient = file["/orbit_info/sc_orient"][0]
        beam_info[sc_orient][beam]["track"] = beam_info[sc_orient][beam]["track_pair"]
        beam_info[sc_orient][beam]["spot"] = beam_info[sc_orient][beam]["atlas_spot"]
        return beam_info[sc_orient][beam]

    def convert_seg_to_ph_res(self, photon_df, segment_df):
        if photon_df.empty or segment_df.empty:
            return pd.DataFrame()

        segs_with_data = segment_df["segment_ph_cnt"] != 0
        segment_clip = segment_df.loc[segs_with_data].copy()
        segment_clip["seg_cumsum"] = segment_clip["segment_ph_cnt"].cumsum()
        seg_max = segment_clip["seg_cumsum"].iloc[-1]
        ph_res_data = pd.DataFrame(np.arange(1, seg_max + 1), columns=["idx"])
        ph_res_data = pd.merge(
            ph_res_data, segment_clip, left_on="idx", right_on="seg_cumsum", how="left"
        )
        ph_res_data.ffill(inplace=True)
        ph_res_data.bfill(inplace=True)
        ph_res_data.drop(["idx", "seg_cumsum"], axis=1, inplace=True)
        ph_res_data = pd.concat([photon_df, ph_res_data], axis=1)
        return ph_res_data

    def preprocess_data(self, data):
        data["x_ph"] = data["segment_dist_x"] + data["dist_ph_along"]
        data["z_ph"] = data["h_ph"] - data["geoid"]
        data["z_dem"] = data["dem_h"] - data["geoid"]
        data["ph_index"] = np.arange(len(data))
        data["ndwi"] = 0

        required_columns = {
            "time": -1,
            "rgt": -1,
            "cycle": -1,
            "track": -1,
            "spot": -1,
            "solar_elevation": np.nan,
            "ref_azimuth": np.nan,
            "ref_elev": np.nan,
            "geoid": np.nan,
            "classification": -1,
            "confidence": -1,
        }

        for col, default in required_columns.items():
            if col not in data:
                data[col] = default

        data = data[
            [
                "time",
                "segment_id",
                "rgt",
                "cycle",
                "track",
                "spot",
                "x_ph",
                "z_ph",
                "z_dem",
                "ref_azimuth",
                "ref_elev",
                "solar_elevation",
                "lon_ph",
                "lat_ph",
                "classification",
                "confidence",
                "ph_index",
                "ndwi",
                "geometry",
            ]
        ]

        int_columns = ["segment_id", "rgt", "cycle", "track", "spot", "classification"]
        data = data.astype({col: int for col in int_columns})

        data = data[data["ndwi"] == 0]

        return data

    def load_single_beam(self, beam, calculate_ndwi_flag=False, start_date="2021-01-01", end_date="2022-12-31"):
        with h5py.File(self.file_path, "r") as file:
            if f"{beam}/heights/h_ph" not in file:
                return pd.DataFrame()

            heights = self.load_heights(file, beam)
            geolocation = self.load_geolocation(file, beam)
            geophys_corr = self.load_geophys_corr(file, beam)
            metadata = self.load_metadata(file, beam)
            beam_info = self.load_beam_info(file, beam)

        ph_res_data = pd.DataFrame(heights)
        segment_res_data = pd.DataFrame(geolocation)
        segment_res_data["geoid"] = geophys_corr["geoid"]
        segment_res_data["dem_h"] = geophys_corr["dem_h"]
        ph_res_data = self.convert_seg_to_ph_res(ph_res_data, segment_res_data)
        ph_res_data.dropna(subset=["h_ph"], inplace=True)

        for key, value in metadata.items():
            ph_res_data[key] = value

        for key, value in beam_info.items():
            ph_res_data[key] = value

        ph_res_data["time"] = ph_res_data["atlas_epoch"] + ph_res_data["delta_time"]

        ph_res_data["geometry"] = ph_res_data.apply(
            lambda row: Point(row.lon_ph, row.lat_ph), axis=1
        )
        gdf = gpd.GeoDataFrame(ph_res_data, geometry="geometry")
        gdf.crs = "EPSG:4979"

        gdf = self.preprocess_data(gdf)

        if calculate_ndwi_flag:
            gdf = calculate_ndwi(gdf, metadata, start_date, end_date)

        return gdf

    def load_data(self, beams, calculate_ndwi_flag=False, start_date="2021-01-01", end_date="2021-12-31"):
        if isinstance(beams, str):
            beams = [beams]

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda beam: self.load_single_beam(
                        beam, calculate_ndwi_flag, start_date, end_date
                    ),
                    beams,
                )
            )

        final_df = pd.concat(results, ignore_index=True)

        ndwi_columns = [col for col in final_df.columns if "ndwi" in col]
        if len(ndwi_columns) > 1:
            final_df["ndwi"] = final_df[ndwi_columns].max(axis=1)
            final_df.drop(columns=ndwi_columns, inplace=True)
        elif len(ndwi_columns) == 1:
            final_df.rename(columns={ndwi_columns[0]: "ndwi"}, inplace=True)

        # store in the object
        self.data = final_df

        return final_df


    def explore(self, output_directory=None, ):
        """
        Explore function to generate a Bokeh plot.
        Left: Tile map in Web Mercator coordinates with latitude and longitude data.
        Right: Scatter plot of lon_ph (converted to Web Mercator) vs orthometric height (z_ph).
        """

        # If at least two colors are provided, enable colored arrows.
        # use_color_arrows = len(arrow_colors) >= 2

        # Preparing the data
        df = self.data.loc[:, ['segment_id', 'lat_ph', 'lon_ph', 'z_ph']]
        df_ = df.groupby('segment_id').median()

        # Convert lat_ph, lon_ph to Web Mercator for consistent plotting
        wgs84 = pyproj.CRS.from_epsg(4979)
        web_merc = pyproj.CRS.from_epsg(3857)
        transformer = pyproj.Transformer.from_crs(wgs84, web_merc, always_xy=True)

        # Transform lat/lon to Web Mercator coordinates
        lon_ph, lat_ph = df.lon_ph.values, df.lat_ph.values
        lon_wm_, lat_wm_ = transformer.transform(lon_ph, lat_ph)

        # Map plot (left side) in Web Mercator
        fig_map = figure(x_range=(min(lon_wm_), max(lon_wm_)),
                        y_range=(min(lat_wm_), max(lat_wm_)),
                        width=400, sizing_mode='stretch_height',
                        x_axis_label='Longitude (Web Mercator)', y_axis_label='Latitude (Web Mercator)',
                        x_axis_type="mercator", y_axis_type="mercator",
                        tools='pan,zoom_in,zoom_out,reset,save,wheel_zoom', active_scroll='wheel_zoom')

        # Add tile map provider from xyzservices
        # fig_map.add_tile(xyz.OpenStreetMap.Mapnik)
        fig_map.add_tile(xyz.Esri.WorldImagery)

        # Draw track line on the map
        fig_map.line(lon_wm_, lat_wm_, color='red', line_width=2)

        # Scatter plot (right side): Web Mercator lon_ph vs. z_ph (Orthometric height)
        z_ph = self.data.z_ph

        # Convert lon_ph to Web Mercator for the scatter plot
        lon_wm, lat_wm = transformer.transform(lon_ph, lat_ph)

        zlim_upper = np.percentile(z_ph, 90) + 15
        zlim_lower = max(np.percentile(z_ph, 0.1) - 20, -70)

        # Synchronize the left plot's y-axis (latitude) with the right plot's x-axis (longitude)
        fig_photon = figure(x_range=fig_map.y_range,  # Link the x_range of fig_photon with the y_range of fig_map
                            y_range=(zlim_lower, zlim_upper),
                            sizing_mode='stretch_both', x_axis_label='Latitude (Web Mercator)',
                            y_axis_label='Orthometric Height (m)',
                            tools='pan,zoom_in,zoom_out,reset,save,ywheel_zoom', active_scroll='ywheel_zoom')

        fig_photon.circle(lat_wm, z_ph, size=1, color='black')

        # Optional arrows on the map plot
        # arrow_colors=["white", "hotpink", "orange", "yellow", "lime", "cyan", "blue", "darkviolet", "black"]
        # arrow_indices = np.linspace(0, len(lon_wm_) - 2, len(arrow_colors), dtype=int)
        # for i, idx in enumerate(arrow_indices):
        #     fig_map.add_layout(Arrow(end=NormalHead(line_color="red", fill_color=arrow_colors[i], line_width=2),
        #                             x_start=lon_wm_[idx], y_start=lat_wm_[idx],
        #                             x_end=lon_wm_[idx + 1], y_end=lat_wm_[idx + 1]))

        # Layout both plots (map and scatter) side by side
        fig_grid = row([fig_map, fig_photon], sizing_mode='stretch_both')

        # Handle saving or displaying the plot
        if output_directory:
            output_filepath_html = os.path.join(output_directory, 'combined_plot.html')
            save(fig_grid, filename=output_filepath_html)
        else:
            show(fig_grid)