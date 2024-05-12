from pydantic import BaseModel, field_validator
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pyimpute import load_training_vector
from dataclasses import dataclass
import numpy as np

@dataclass
class RasterLayer:
    name: str
    path: str
    
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
        return pd.concat([self.data, self.random_points_generated]).reset_index(drop=True)
    
    @property
    def backup_presence_absence_df(self)->pd.DataFrame:
        return pd.concat([self.data, self._backup]).reset_index(drop=True)
    
    def build_model(self):
        return
    
    def _get_geo_dataframe(self) -> gpd.GeoDataFrame:
        """
        Create a dataframe with the occurences
        Args:
            data (pd.DataFrame): _description_
            crs (_type_): _description_

        Returns:
            gpd.GeoDataFrame: _description_
        """
        data = self.data.copy()
        data['geometry'] = list(zip(data.lon, data.lat))
        data = data[['target','geometry']].copy()
        data['geometry'] = data["geometry"].apply(Point)
        geo_dataframe = gpd.GeoDataFrame(data)

        # Create the geodataframe
        output_dataframe = gpd.GeoDataFrame(
            gpd.GeoDataFrame(data),
            crs = {'init': self.crs},
            geometry = geo_dataframe['geometry']
        ).to_crs(self.crs).reset_index(drop=True)
        
        return output_dataframe
    
    def pandas_to_geoframe(self, dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
        data = dataframe.copy()
        data['geometry'] = list(zip(data.lon, data.lat))
        data = data[['target','geometry']].copy()
        data['geometry'] = data["geometry"].apply(Point)
        geo_dataframe = gpd.GeoDataFrame(data)

        # Create the geodataframe
        output_dataframe = gpd.GeoDataFrame(
            gpd.GeoDataFrame(data),
            crs = {'init': self.crs},
            geometry = geo_dataframe['geometry']
        ).to_crs(self.crs).reset_index(drop=True)
        return output_dataframe
    
    def export_geoframe_to_json(self, gpd_dataframe: gpd.GeoDataFrame, filepath:str):
        gpd_dataframe.to_file(filepath, driver="GeoJSON")
        return
    
    def get_convex_hull_from_data(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({'geometry': [self.geoframe.unary_union.convex_hull]}, geometry='geometry')
    
    def generate_random_points(self, n_points: int, target: int = 0, seed: int = 42, overwrite: bool =True) -> pd.DataFrame:
        random_points = self.geometry.sample_points(n_points, rng=seed).explode(index_parts=False)
        coordinates = random_points.apply(lambda point: pd.Series({'lat': point.y, 'lon': point.x}))
        random_points_generated = pd.DataFrame(coordinates).reset_index(drop=True)
        random_points_generated['target'] = target
        if overwrite:
            self._backup = self.random_points_generated
            self.random_points_generated = random_points_generated
        return random_points_generated
    
    def export_dataframe_with_raster_features(
        self, geojson: str, explanatory_rasters: list[RasterLayer], output_file_path: str
        ):
        """
        Args:
            geojson (str): _description_
            explanatory_rasters (list[str]): _description_
            columns (list[str]): _description_
            output_file_path (str): _description_
            response_field (str, optional): _description_. Defaults to "Presence".
        """
        columns = [raster_layer.name for raster_layer in explanatory_rasters]
        rasters = [raster_layer.path for raster_layer in explanatory_rasters]
        
        train_xs, train_y = load_training_vector(geojson, rasters, response_field="target")
        df = pd.DataFrame(train_xs)
        df.loc[:,"target"] = train_y
        df = df[~df[0].isnull()]
        df.columns = columns

        df = df.merge(self.presence_absence_df, left_index=True, right_index=True)
        df.to_parquet(output_file_path)
        return
    
    def generate_uniform_points_within_polygon(self, spacing: float = 0.5, target: int = 0, overwrite: bool = True):
        # Step 1: Determine the bounding box of the polygon
        bbox = self.geometry.geometry.total_bounds
        min_x, min_y, max_x, max_y = bbox
        
        # Step 2: Generate uniformly distributed points within the bounding box
        x_coords = np.arange(min_x, max_x, spacing)
        y_coords = np.arange(min_y, max_y, spacing)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = [Point(x, y) for x, y in zip(xx.ravel(), yy.ravel())]
        
        # Step 3: Filter out points that fall outside the polygon
        points_within_polygon = [point for point in points if self.geometry.geometry.contains(point).any()]
        
        # Step 4: Convert the filtered points into a Pandas DataFrame
        points_df = gpd.GeoDataFrame(geometry=points_within_polygon, crs=self.geometry.crs)
        points_df['lon'] = points_df.geometry.x
        points_df['lat'] = points_df.geometry.y
        points_df['target'] = target
        points_df.drop(columns='geometry', inplace=True)
        if overwrite:
            self._backup = self.random_points_generated
            self.random_points_generated = points_df
        return points_df
    
    def add_features(self, feats: RasterLayer | list[RasterLayer]):
        if isinstance(feats, list):
            return
        elif isinstance(feats, RasterLayer):
            return
        else:
            return
    