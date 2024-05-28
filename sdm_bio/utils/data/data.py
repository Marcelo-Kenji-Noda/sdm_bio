from pathlib import Path

import pandas as pd


def correlation(dataset: pd.DataFrame, threshold: float = 0.7, *args, **kwargs) -> set:
    """
    Identify columns in the dataset that are highly correlated with other columns.

    Parameters:
    dataset (pd.DataFrame): The input DataFrame containing the data.
    threshold (float): The correlation threshold above which columns are considered highly correlated. Default is 0.7.

    Returns:
    set: A set of column names that have a correlation value greater than the specified threshold with any other column.

    Example:
    >>> data = {
    >>>     'A': [1, 2, 3, 4, 5],
    >>>     'B': [2, 4, 6, 8, 10],
    >>>     'C': [5, 4, 3, 2, 1],
    >>>     'D': [1, 3, 5, 7, 9]
    >>> }
    >>> df = pd.DataFrame(data)
    >>> high_corr_columns = correlation(df, threshold=0.8)
    >>> print(high_corr_columns)
    {'B', 'D'}
    """
    original_columns = set(dataset.columns)
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    
    if "verbose" in kwargs:
        assert isinstance(kwargs["verbose"], bool)
        print(
        f"""
        Removed columns: {list(col_corr)}
        """)
    return col_corr

def generate_model_data(dataframe: str | pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Preprocess the dataset for machine learning by removing specific columns, handling missing values, and removing highly correlated features.

    Parameters:
    dataframe_path (str): The file path to the Parquet file containing the dataset.

    Returns:
    pd.DataFrame: A DataFrame with the target variable separated, specific columns ('lat', 'lon') dropped, and highly correlated features removed.

    Example:
    >>> data = {
    >>>     'lat': [34.05, 36.16, 40.71, 34.05, 36.16],
    >>>     'lon': [-118.24, -115.15, -74.01, -118.24, -115.15],
    >>>     'feature1': [1, 2, 3, 4, 5],
    >>>     'feature2': [2, 4, 6, 8, 10],
    >>>     'feature3': [5, 4, 3, 2, 1],
    >>>     'target': [0, 1, 0, 1, 0]
    >>> }
    >>> df = pd.DataFrame(data)
    >>> df.to_parquet('sample_data.parquet')
    >>> model_data = generate_model_data('sample_data.parquet')
    >>> print(model_data)
       feature1  feature3
    0         1         5
    1         2         4
    2         3         3
    3         4         2
    4         5         1
    """
    if isinstance(dataframe, pd.DataFrame):
        df = dataframe.copy()
    elif isinstance(dataframe, str):
        df = pd.read_parquet(Path(dataframe)).dropna()
    else:
        raise ValueError("The param 'dataframe' should be either a string or a pd.DataFrame")
    assert set(['lat', 'lon']).intersection(set(df.columns)) == set(['lat','lon'])

    y = df.pop("target")
    X = df.copy().drop(columns=['lat', 'lon'])
    return X.drop(columns=correlation(X, *args, **kwargs)), y
