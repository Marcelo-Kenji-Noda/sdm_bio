import geopandas as gpd
from typing import Literal
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
from shapely.geometry import Point

class RasterLayer:
    def __init__(self, raster_filepath: str, name: str):
        self.raster_filepath = raster_filepath
        self.name = name
        pass
    
class StudyArea:
    def __init__(
        self,
        area_type: Literal["box","convex_hull","polygon"],
        bounds: tuple[float, float, float, float],
        params: dict = {},
        mapstyle: str='carto',
        crs: str = 'epsg:4326',
        ):
        self.area_type = area_type
        self.bounds = bounds
        self.crs = crs
        self.params = params
        self.mapstyle = mapstyle
        if self._validate_area_type():
            self.geoframe = self._get_geoframe()
            
    def _validate_area_type(self):
        target_keys: set= set(self.params.keys())
        params_validator = {
            'box':set([]),
            'convex_hull':set(["file","points_geoframe"]),
            'polygon_params':set(["file"])
        }
        try:
            params_validator[self.area_type].issubset(target_keys)
        except Exception:
            raise Exception(f"Expected {params_validator[self.area_type]} but got {target_keys}")
        return True

    def plot_map(self):
        return
    
    def plot_grid_heatmap(self, min_lat, max_lat, min_lon, max_lon, grid):
        """
        Plot a grid as a heatmap with a basemap.

        Args:
            min_lat (float): The minimum latitude of the grid.
            max_lat (float): The maximum latitude of the grid.
            min_lon (float): The minimum longitude of the grid.
            max_lon (float): The maximum longitude of the grid.
            grid (np.ndarray): The grid containing values.

        Returns:
            fig, ax: Figure and axes objects.
        """
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 12))

        # Plot the heatmap
        extent = [min_lon, max_lon, min_lat, max_lat]
        heatmap = ax.imshow(grid, extent=extent, aspect='auto', cmap='viridis', alpha=0.9)


        # Add basemap
        ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=0.4)
        
        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', fraction=0.04, pad=0.04)
        cbar.set_label('Heatmap Intensity')
        
        # Set x and y axis labels
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        # Set plot title
        ax.set_title('Grid Heatmap with Carto-Positron Basemap')

        # Return figure and axes objects
        return fig, ax
    
class SDM:
    def __init__(self,domain:StudyArea, points: list[tuple], generate_random_points: int = 0, random_points_mode:str = None):
        self.domain = domain
        self.points = points
        self.generate_random_points = generate_random_points
        self.random_points_mode = random_points_mode
        
        ## Points dataframe
        self.points_occurence_pandas_dataframe: pd.DataFrame = self._get_pandas_points_dataframe()
        self.points_occurence_geopandas_dataframe: gpd.GeoDataFrame = self._get_geopandas_dataframe()
        pass
    
    def _get_pandas_points_dataframe(self) -> pd.DataFrame:
        geoframe_base = pd.DataFrame()
        geoframe_base['Latitude'] = [pt[0] for pt in self.points]
        geoframe_base['Longitude'] = [pt[1] for pt in self.points]
        return geoframe_base
    
    def _get_geopandas_dataframe(self) -> gpd.GeoDataFrame:
        coordinates = self.points_occurence_pandas_dataframe.copy()
        coordinates['geometry'] = list(zip(coordinates["Longitude"], coordinates["Latitude"]))
        coordinates = coordinates[['geometry']].copy()
        coordinates['geometry'] = coordinates["geometry"].apply(Point)
        
        geo_dataframe = gpd.GeoDataFrame(coordinates)
        return gpd.GeoDataFrame(geo_dataframe,crs = {'init': self.domain.crs},geometry = geo_dataframe['geometry']).to_crs(self.domain.crs).reset_index(drop=True)
        
    def create_dataframe_with_random_points_inside_geom(
        self,
        n_points: int = 100,
        seed: int | None = 42,
    ) -> pd.DataFrame:
        """
        Args:
            region_dataframe (gpd.GeoDataFrame): Geopandas Dataframe with the domain
            n_points (int, optional): Number of random points generated. Defaults to 100.
            seed (int | None, optional): Random seed. Defaults to 42.

        Returns:
            pd.DataFrame: dataframe with columns "Latitude" and "Longitude"
        """
        if self.domain.area_type == 'convex_hull':
            data = {'geometry': [self.self.points_occurence_geopandas_dataframe.unary_union.convex_hull]}
        else:
            raise NotImplementedError("Other method not implemented")
        gdf_union = gpd.GeoDataFrame(data, geometry='geometry')
        random_points = gdf_union.sample_points(n_points, rng=seed).explode(index_parts=False)
        coordinates = random_points.apply(lambda point: pd.Series({'Latitude': point.y, 'Longitude': point.x}))
        return pd.DataFrame(coordinates).reset_index(drop=True)