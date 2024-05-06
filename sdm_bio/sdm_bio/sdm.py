import geopandas as gpd
from typing import Literal

class RasterLayer:
    def __init__(self, raster_filepath: str, name: str):
        self.raster_filepath = raster_filepath
        self.name = name
        pass
    
class StudyArea2:
    """
    GeoSpace determine an specific area os study
    """
    def __init__(self, shape_filepath: str, *args, **kwargs):
        self.shape_filepath = shape_filepath
        self.gpd_dataframe = gpd.read_file(shape_filepath, *args, **kwargs)
        self.raster_layers: dict[str,RasterLayer] = {}
        self.points: set[tuple[float, float]] = []
        
    def export_geofile(self, filename: str, driver: str = 'GeoJSON'):
        self.gpd_dataframe.to_file(filename, driver=driver)
        return
    
    def filter_points_inside_domain(self, filename: str):
        pass
    
    def add_layer(self, *args: list[RasterLayer]) -> None:
        for layer in args:
            self.raster_layers[layer.name] = layer
        return
    
    def add_points(self, points: list[tuple] | set[tuple]):
        if type(points) == list:
            points = set(points)
        self.points.add(points)
        return
    
class StudyArea:
    def __init__(self, area_type: Literal["box","convex_hull","polygon"], geoframe: gpd.GeoDataFrame = None):
        self.area_type = area_type
        self.geometry: gpd.GeoDataFrame = None
        
    def _define_study_area_box(self, *args, **kwargs):
        if "" in kwargs:
            print("")
        return
        
    def _define_study_area_convex_hull(self, *args, **kwargs):
        return
    
    def _define_study_area_shapefile(self, *args, **kwargs):
        return
    
    def define_study_area(self, study_area_type: Literal["box","convex_hull","shapefile"], shape_file_path:str= None):
        match study_area_type:
            case "shapefile":
                self._define_study_area_shapefile(shape_file_path)
            case "convex_hull":
                self._define_study_area_convex_hull()
            case "box":
                raise NotImplementedError("Not implemented shapefile")
            case _:
                raise NotImplementedError("Value Error")
        pass
    
class SDM:
    def __init__(self, points: list[tuple]):
        self.points = points
        self.study_area:StudyArea = None 
        pass
    
if __name__ == '__main__':
    """
    StudyArea(
        study_area_type: str = "box",
        points: list[] = [],
        crs: str = 'ESPG4326',
    )
    
    sdm_model = SDM(
        domain: StudyArea
    )
    
    sdm_model.fit()
    
    """
    points = []
    test = SDM(points=points)