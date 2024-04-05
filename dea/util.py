"""Utility helper methods for diffusion entropy analysis"""
import os
import time
import numpy as np
import polars as pl

def make_sample_data(length: int, seed: float = time.time) -> np.ndarray:
    """Generates an array of sample data."""
    np.random.seed(seed)  # for baseline consistency 1010
    random_steps = np.random.choice([-1, 1], length)
    random_steps[0] = 0  # always start from 0
    random_walk = np.cumsum(random_steps)
    return random_walk

def get_data(filepath: str) -> pl.DataFrame:
    """Convenience function for reading input data
    
    Parameters
    ----------
    filepath : str
        System path to a file containing data.
        Must include the full file name, including the extension.
        Example: "/example/path/to/file.csv"
    
    Returns
    -------
    data : DataFrame
        A polars DataFrame containing the data.
    
    Raises
    ------
    ValueError
        If filepath points to a file of type other than
        CSV. Support for more types of files is a work in
        progress.
    """
    filetype = os.path.splitext(filepath)[1]
    supported_types = [".csv"]
    assert filetype in supported_types, f"'filetype' must be one of: \
        {supported_types}."
    if filetype == ".csv":
        data = pl.scan_csv(filepath)
    return data
