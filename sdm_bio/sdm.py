from pydantic import BaseModel, field_validator
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pyimpute import load_training_vector

class RasterLayer(BaseModel):
    name: str
    path: str
    
class Bounds(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
class SDM(BaseModel):
    """
    Example Usage

    Args:
        BaseModel (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        sdm_model = SDM(data = dataframe)
        
        sdm_model.generate_random_points(n_points = 1000)
        sdm_model.presence_absence_df
    """
    data: pd.DataFrame
    crs: str = "EPSG:4326"
    geometry: gpd.GeoDataFrame = None
    bounds: Bounds = None
    random_points_generated: pd.DataFrame = None
    cache_path: str | None = None

    def __post_model_init__(self, __context):
        self.geoframe = self._get_geo_dataframe()
        if not self.geometry:
            self.geometry = self.get_convex_hull_from_data()
            
    @field_validator('data')
    @classmethod
    def name_must_contain_space(cls, v: pd.DataFrame) -> pd.DataFrame:
        if ['lat','lon','target'] not in v.columns:
            raise ValueError('[lat, lon, target] must be in columns of data')
        return v
    
    @property
    def presence_absence_df(self) -> pd.DataFrame:
        return pd.concat([self.data, self.random_points_generated])
    
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
    
    def get_convex_hull_from_data(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({'geometry': [self.geo_dataframe.unary_union.convex_hull]}, geometry='geometry')
    
    def generate_random_points(self, n_points: int, target: int = 0, seed: int = 42) -> pd.DataFrame:
        if not self.geometry:
            raise ValueError("Geometry not defined")
        random_points = self.geometry.sample_points(n_points, rng=seed).explode(index_parts=False)
        coordinates = random_points.apply(lambda point: pd.Series({'lat': point.y, 'lon': point.x}))
        self.random_points_generated = pd.DataFrame(coordinates).reset_index(drop=True)
        self.random_points_generated['target'] = target
        return self.random_points_generated
    
    def export_dataframe_with_raster_features(
        geojson: str, explanatory_rasters: list[RasterLayer], output_file_path: str,occurence_absence_dataframe_path:str,  *args, **kwargs
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
        
        occurence_absence = pd.read_parquet(occurence_absence_dataframe_path)[['Latitude','Longitude']].reset_index(drop=True)
        df = df.merge(occurence_absence.reset_index(), left_index=True, right_index=True)
        df.to_parquet(output_file_path)
        return

    def add_features(self, feats: RasterLayer | list[RasterLayer]):
        if isinstance(feats, list):
            return
        elif isinstance(feats, RasterLayer):
            return
        else:
            return
    