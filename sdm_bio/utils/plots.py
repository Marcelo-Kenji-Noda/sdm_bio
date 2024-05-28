import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point


def plot_points(df: pd.DataFrame, hide_bg_pts: bool = False, *args, **kwargs):
    """
    Plot points on a map within specified latitude and longitude bounds.

    Args:
        df (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'target' columns.

    Returns:
        fig, ax: Figure and axes objects.
    """
    min_lat = df['lat'].min() - 0.5
    max_lat = df['lat'].max() + 0.5
    min_lon = df['lon'].min() - 0.5
    max_lon = df['lon'].max() + 0.5

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 12))

    df_1 = df[df['target'] == 1]
    df_2 = df[df['target'] == 0]
    
    # Plot points
    ax.scatter(df_1['lon'], df_1['lat'], c='r', alpha=0.3, s=1, *args, **kwargs)
    if not hide_bg_pts:
        ax.scatter(df_2['lon'], df_2['lat'], c='b', alpha=0.3, s=1, *args, **kwargs)
    
    # Add basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=1)
    
    # Set x and y axis limits
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Set x and y axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Set plot title
    ax.set_title('Points on Map')

    # Show grid
    ax.grid(True, alpha=0.3)

    # Return figure and axes objects
    return fig, ax

def geoframe_to_pandas(geo_dataframe: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Convert a GeoDataFrame to a Pandas DataFrame with 'lat' and 'lon' columns.

    Args:
        geo_dataframe (gpd.GeoDataFrame): GeoDataFrame with 'geometry' column containing Point geometries.

    Returns:
        pd.DataFrame: DataFrame with 'lat' and 'lon' columns.
    """
    # Extract latitudes and longitudes from the geometry column
    geometry = geo_dataframe['geometry']
    lats = [point.y for point in geometry]
    lons = [point.x for point in geometry]

    # Create a new DataFrame with latitudes and longitudes
    pandas_dataframe = pd.DataFrame({'lat': lats, 'lon': lons})

    # Add other columns from the GeoDataFrame if needed
    pandas_dataframe['target'] = geo_dataframe['target']  # Example: Adding 'target' column

    return pandas_dataframe

def remove_points_within_radius(geoframe: gpd.GeoDataFrame, R: float) -> pd.DataFrame:
    """
    Remove points within a specified radius from points with a specific target value.

    Args:
        geoframe (gpd.GeoDataFrame): GeoDataFrame containing 'geometry' and 'target' columns.
        R (float): Radius within which points should be removed.

    Returns:
        pd.DataFrame: DataFrame with points removed within the specified radius.
    """
    # Create a copy of the input GeoDataFrame
    filtered_geoframe = geoframe.copy()
    
    # Iterate through each point with target 1
    for idx, row in geoframe[geoframe['target'] == 1].iterrows():
        # Extract the coordinates of the point with target 1
        point = row['geometry']
        
        # Create a buffer around the point with target 1
        buffer = point.buffer(R)
        
        # Find points with target 0 within the buffer and remove them
        points_within_buffer = filtered_geoframe[(filtered_geoframe['target'] == 0) & (filtered_geoframe['geometry'].within(buffer))]
        filtered_geoframe = filtered_geoframe.drop(points_within_buffer.index)
        
    return geoframe_to_pandas(filtered_geoframe)

def pandas_to_geoframe(dataframe: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert a Pandas DataFrame with 'lat' and 'lon' columns to a GeoDataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing 'lat', 'lon', and 'target' columns.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with Point geometries.
    """
    data = dataframe.copy()
    data['geometry'] = list(zip(data.lon, data.lat))
    data = data[['target', 'geometry']].copy()
    data['geometry'] = data["geometry"].apply(Point)
    geo_dataframe = gpd.GeoDataFrame(data)

    # Create the GeoDataFrame
    output_dataframe = gpd.GeoDataFrame(
        gpd.GeoDataFrame(data),
        crs='EPSG:4326',
        geometry=geo_dataframe['geometry']
    ).to_crs('EPSG:4326').reset_index(drop=True)
    return output_dataframe

def remove_no_data_values(raster_data: pd.DataFrame, raster_info: pd.DataFrame):
    """
    Remove rows from raster_data DataFrame that contain no-data values based on raster_info.

    Args:
        raster_data (pd.DataFrame): DataFrame containing raster data.
        raster_info (pd.DataFrame): DataFrame containing 'name' and 'no_data' columns.

    Returns:
        pd.DataFrame: DataFrame with no-data values removed.
    """
    for _, row in raster_info.iterrows():
        info_dict = row.to_dict()
        indexes = raster_data[raster_data[info_dict['name']] == info_dict['no_data']].index
        raster_data = raster_data.drop(indexes)
    return raster_data