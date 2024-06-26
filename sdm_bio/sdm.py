import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyimpute import load_training_vector
from rasterio.mask import mask
from shapely.geometry import Point
from shapely.ops import cascaded_union


@dataclass
class RasterLayer:
    name: str
    path: str
    no_data: str | int = None


@dataclass
class Bounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


@dataclass
class SDM:
    data: pd.DataFrame
    crs: str = "EPSG:4326"
    geometry: gpd.GeoDataFrame = None
    bounds: Bounds = None
    random_points_generated: pd.DataFrame = None
    cache_path: str | None = None
    _backup: pd.DataFrame = None

    def __post_init__(self):
        self.geoframe = self._get_geo_dataframe()
        self.geometry = self.get_convex_hull_from_data()

    @property
    def presence_absence_df(self) -> pd.DataFrame:
        return pd.concat([self.data, self.random_points_generated]).reset_index(
            drop=True
        )

    @property
    def backup_presence_absence_df(self) -> pd.DataFrame:
        return pd.concat([self.data, self._backup]).reset_index(drop=True)

    def build_model(self):
        return

    def _get_geo_dataframe(self) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame with the occurrences from the provided data.
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with occurrences.
        """
        data = self.data.copy()
        data["geometry"] = list(zip(data.lon, data.lat))
        data = data[["target", "geometry"]].copy()
        data["geometry"] = data["geometry"].apply(Point)
        geo_dataframe = gpd.GeoDataFrame(data)

        # Create the GeoDataFrame
        output_dataframe = (
            gpd.GeoDataFrame(
                geo_dataframe,
                crs={"init": "epsg:4326"},
                geometry=geo_dataframe["geometry"],
            )
            .to_crs("epsg:4326")
            .reset_index(drop=True)
        )

        return output_dataframe

    def pandas_to_geoframe(self, dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Convert a Pandas DataFrame to a GeoDataFrame.
        Args:
            dataframe (pd.DataFrame): DataFrame to convert.
        Returns:
            gpd.GeoDataFrame: Converted GeoDataFrame.
        """
        data = dataframe.copy()
        data["geometry"] = list(zip(data.lon, data.lat))
        data = data[["target", "geometry"]].copy()
        data["geometry"] = data["geometry"].apply(Point)
        geo_dataframe = gpd.GeoDataFrame(data)

        # Create the GeoDataFrame
        output_dataframe = (
            gpd.GeoDataFrame(
                geo_dataframe,
                crs={"init": self.crs},
                geometry=geo_dataframe["geometry"],
            )
            .to_crs(self.crs)
            .reset_index(drop=True)
        )
        return output_dataframe

    def geoframe_to_pandas(self, geo_dataframe: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Convert a GeoDataFrame to a Pandas DataFrame with 'lat' and 'lon' columns.
        Args:
            geo_dataframe (gpd.GeoDataFrame): GeoDataFrame with 'geometry' column containing Point geometries.
        Returns:
            pd.DataFrame: DataFrame with 'lat' and 'lon' columns.
        """
        geometry = geo_dataframe["geometry"]
        lats = [point.y for point in geometry]
        lons = [point.x for point in geometry]

        pandas_dataframe = pd.DataFrame({"lat": lats, "lon": lons})
        pandas_dataframe["target"] = geo_dataframe["target"]

        return pandas_dataframe

    def remove_points_within_radius(
        self, geoframe: gpd.GeoDataFrame, R: float
    ) -> pd.DataFrame:
        """
        Remove points within a specified radius from points with flag 1.
        Args:
            geoframe (gpd.GeoDataFrame): GeoDataFrame containing points.
            R (float): Radius within which to remove points.
        Returns:
            pd.DataFrame: DataFrame with points removed within the specified radius.
        """
        filtered_geoframe = geoframe.copy()

        for idx, row in geoframe[geoframe["target"] == 1].iterrows():
            point = row["geometry"]
            buffer = point.buffer(R)
            points_within_buffer = filtered_geoframe[
                (filtered_geoframe["target"] == 0)
                & (filtered_geoframe["geometry"].within(buffer))
            ]
            filtered_geoframe = filtered_geoframe.drop(points_within_buffer.index)

        return self.geoframe_to_pandas(filtered_geoframe)

    def remove_no_data_values(
        self, raster_data: pd.DataFrame, raster_info: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Remove no data values from raster data.
        Args:
            raster_data (pd.DataFrame): DataFrame containing raster data.
            raster_info (pd.DataFrame): DataFrame containing raster metadata.
        Returns:
            pd.DataFrame: Cleaned raster data.
        """
        for _, row in raster_info.iterrows():
            info_dict = row.to_dict()
            indexes = raster_data[
                raster_data[info_dict["name"]] == info_dict["no_data"]
            ].index
            raster_data = raster_data.drop(indexes)
        return raster_data

    def plot_points(self, df: pd.DataFrame):
        """
        Plot points on a map within specified latitude and longitude bounds.
        Args:
            df (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'target' columns.
        Returns:
            fig, ax: Figure and axes objects.
        """
        min_lat = df["lat"].min() - 0.5
        max_lat = df["lat"].max() + 0.5
        min_lon = df["lon"].min() - 0.5
        max_lon = df["lon"].max() + 0.5

        fig, ax = plt.subplots(figsize=(10, 12))
        df_1 = df[df["target"] == 1]
        df_2 = df[df["target"] == 0]

        ax.scatter(df_1["lon"], df_1["lat"], c="r", alpha=0.3, s=1)
        ax.scatter(df_2["lon"], df_2["lat"], c="b", alpha=0.3, s=1)

        ctx.add_basemap(
            ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron, alpha=1
        )
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Points on Map")
        ax.grid(True, alpha=0.3)

        return fig, ax

    def export_geoframe_to_json(self, gpd_dataframe: gpd.GeoDataFrame, filepath: str):
        """
        Export a GeoDataFrame to a GeoJSON file.
        Args:
            gpd_dataframe (gpd.GeoDataFrame): GeoDataFrame to export.
            filepath (str): File path to save the GeoJSON file.
        """
        gpd_dataframe.to_file(filepath, driver="GeoJSON")

    def export_self_geoframe_to_json(self, filepath: str):
        """
        Export the internal GeoDataFrame to a GeoJSON file.
        Args:
            filepath (str): File path to save the GeoJSON file.
        """
        geoframe = self.pandas_to_geoframe(self.presence_absence_df)
        geoframe.to_file(filepath, driver="GeoJSON")

    def get_convex_hull_from_data(self) -> gpd.GeoDataFrame:
        """
        Generate a convex hull from the data points.
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the convex hull.
        """
        return gpd.GeoDataFrame(
            {"geometry": [self.geoframe.unary_union.convex_hull]}, geometry="geometry"
        )

    def create_union_of_buffers(self, radius_km: float) -> gpd.GeoDataFrame:
        """
        Create a geometry that is the union of all circles with radius R km centered at each point.
        Args:
            radius_km (float): Radius of each circle in kilometers.
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the union of all buffers.
        """

        def convert_km_to_degrees(lat, km):
            """
            Convert kilometers to degrees based on latitude.
            """
            lat_spacing = km / 111  # 1 degree latitude is approximately 111 km
            lon_spacing = km / (
                111 * np.cos(np.radians(lat))
            )  # Adjust for the actual latitude
            return lat_spacing, lon_spacing

        # Calculate buffer sizes in degrees
        centroid_lat = self.geoframe.geometry.centroid.y.mean()
        lat_spacing, lon_spacing = convert_km_to_degrees(centroid_lat, radius_km)

        # Create buffers
        buffers = self.geoframe.copy()
        buffers["geometry"] = buffers["geometry"].buffer(
            lat_spacing
        )  # Buffering in degrees of latitude

        # Union of all buffers
        union_geometry = buffers.unary_union

        # Return as GeoDataFrame
        return gpd.GeoDataFrame({"geometry": [union_geometry]}, geometry="geometry")

    def generate_random_points(
        self, n_points: int, target: int = 0, seed: int = 42, overwrite: bool = True
    ) -> pd.DataFrame:
        """
        Generate random points within the geometry.
        Args:
            n_points (int): Number of points to generate.
            target (int): Target value for the generated points.
            seed (int): Seed for random number generator.
            overwrite (bool): Whether to overwrite existing random points.
        Returns:
            pd.DataFrame: DataFrame containing the generated points.
        """
        random_points = self.geometry.sample_points(n_points, rng=seed).explode(
            index_parts=False
        )
        coordinates = random_points.apply(
            lambda point: pd.Series({"lat": point.y, "lon": point.x})
        )
        random_points_generated = pd.DataFrame(coordinates).reset_index(drop=True)
        random_points_generated["target"] = target

        if overwrite:
            self._backup = self.random_points_generated
            self.random_points_generated = random_points_generated

        return random_points_generated

    def export_dataframe_with_raster_features(
        self,
        geojson: str,
        explanatory_rasters: list[RasterLayer],
        output_file_path: str,
    ) -> pd.DataFrame:
        """
        Export a DataFrame with raster features.
        Args:
            geojson (str): _description_
            explanatory_rasters (list[str]): _description_
            columns (list[str]): _description_
            output_file_path (str): _description_
            response_field (str, optional): _description_. Defaults to "Presence".
        """
        columns = [raster_layer.name for raster_layer in explanatory_rasters]
        columns.append("target")
        rasters = [raster_layer.path for raster_layer in explanatory_rasters]

        train_xs, train_y = load_training_vector(
            geojson, rasters, response_field="target"
        )
        df = pd.DataFrame(train_xs)
        df.loc[:, "target"] = train_y
        df = df[~df[0].isnull()]
        df.columns = columns

        df = df.merge(
            self.presence_absence_df[["lon", "lat"]], left_index=True, right_index=True
        )
        df.drop_duplicates().dropna().to_parquet(output_file_path)
        return

    def generate_uniform_points_within_polygon(
        self, geometry, spacing_km: float = 0.5, target: int = 0, overwrite: bool = True
    ):
        # Step 1: Determine the bounding box of the polygon
        bbox = geometry.geometry.total_bounds
        min_x, min_y, max_x, max_y = bbox

        # Calculate the latitude for conversion (use centroid latitude)
        centroid_lat = geometry.geometry.centroid.y.mean()

        # Conversion factors
        lat_spacing = spacing_km / 111  # 1 degree latitude is approximately 111 km
        lon_spacing = spacing_km / (
            111 * np.cos(np.radians(centroid_lat))
        )  # Adjust for the actual latitude

        # Step 2: Generate uniformly distributed points within the bounding box
        x_coords = np.arange(min_x, max_x, lon_spacing)
        y_coords = np.arange(min_y, max_y, lat_spacing)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]

        # Step 3: Filter out points that fall outside the polygon
        points_within_polygon = [
            point for point in points if geometry.geometry.contains(point).any()
        ]

        # Step 4: Convert the filtered points into a Pandas DataFrame
        points_df = gpd.GeoDataFrame(geometry=points_within_polygon, crs=geometry.crs)
        points_df["lon"] = points_df.geometry.x
        points_df["lat"] = points_df.geometry.y
        points_df["target"] = target
        points_df.drop(columns="geometry", inplace=True)

        if overwrite:
            self._backup = self.random_points_generated
            self.random_points_generated = points_df

        return points_df

    def generate_features_dataframe(
        self,
        data_path: str,
        raster_file_path: list[str],
        raster_file_cols: list[str],
        n_random_points: int = 5_000,
        spacing: float = 0.1,
        random_points: bool = True,
        use_convex_polygon: bool = True,
        features_path: str = "OUTPUT/features.parquet"
    ) -> pd.DataFrame:
        execution_date = datetime.now().strftime("%d-%m-%Y")

        if not os.path.exists(os.path.join(data_path, execution_date + "/")):
            created_dir = os.path.join(data_path, execution_date + "/")
            os.mkdir(created_dir)
            data_path = created_dir
        else:
            data_path = os.path.join(data_path, execution_date + "/")
        if random_points:
            _df = self.generate_random_points(n_points=n_random_points)
        else:
            if use_convex_polygon:
                _df = self.generate_uniform_points_within_polygon(
                    self.geometry, spacing_km=spacing
                )
            else:
                geometry = self.create_union_of_buffers(50)
                _df = self.generate_uniform_points_within_polygon(
                    geometry, spacing_km=spacing
                )

        self.presence_absence_df.to_parquet(
            os.path.join(data_path, "INFO/random_data.parquet")
        )
        raster_layers = []
        for col, path in zip(raster_file_cols, raster_file_path):
            with rasterio.open(path) as src:
                raster_layers.append(
                    RasterLayer(name=col, path=path, no_data=src.nodata)
                )

        pd.DataFrame(raster_layers).to_parquet(
            os.path.join(data_path, "INFO/raster_info.parquet")
        )
        self.export_self_geoframe_to_json(
            os.path.join(data_path, "INFO/presence_absence.json")
        )

        self.export_dataframe_with_raster_features(
            os.path.join(data_path, "INFO/presence_absence.json"),
            explanatory_rasters=raster_layers,
            output_file_path=os.path.join(data_path, features_path),
        )

        return pd.read_parquet(os.path.join(data_path, features_path))
