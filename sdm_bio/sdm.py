from pydantic import BaseModel, field_validator
import geopandas as gpd
import pandas as pd

class RasterLayer(BaseModel):
    name: str
    path: str
    
class Bounds(BaseModel):
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    crs: str = "EPSG:4326"
    
class StudyArea(BaseModel):
    geometry: gpd.GeoDataFrame
    bounds: Bounds
    
    def generate_background(self):
        return
    
    def random_points(self, n_points: int, target: int = 0):
        return
    
    
class SDM(BaseModel):
    data: pd.DataFrame
    domain: StudyArea
    cache_path: str | None = None
    
    @field_validator('data')
    @classmethod
    def name_must_contain_space(cls, v: pd.DataFrame) -> pd.DataFrame:
        if ['lat','lon','target'] not in v.columns:
            raise ValueError('[lat, lon, target] must be in columns of data')
        return v
    
    def build_model(self):
        return

    def add_features(self, feats: RasterLayer | list[RasterLayer]):
        if isinstance(feats, list):
            return
        elif isinstance(feats, RasterLayer):
            return
        else:
            return
    