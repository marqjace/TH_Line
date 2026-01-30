import os
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import griddata

def make_transect_grid(xmin=-126.625, xmax=-124.375, ymin=0, ymax=1000, xn=36, yn=200):
    """
    Create a meshgrid for transect interpolation.

    :param xmin: Minimum longitude
    :param xmax: Maximum longitude
    :param ymin: Minimum depth
    :param ymax: Maximum depth
    :param xn: Number of points in longitude
    :param yn: Number of points in depth

    :return: Meshgrid of longitude and depth
    """
    x = np.linspace(xmin, xmax, xn)
    y = np.linspace(ymin, ymax, yn)

    return np.meshgrid(x, y)

def interp_to_grid(lon, depth, values, Xgrid, Ygrid):
    return griddata(
        points=(lon.values.ravel(), depth.values.ravel()),
        values=values.values.ravel(),
        xi=(Xgrid, Ygrid),
        method="linear"
    )

def transect(filepath, Xgrid, Ygrid):
    """
    Process a single transect file.

    :param filepath: Path to the transect netCDF file
    :param Xgrid: Meshgrid of longitude
    :param Ygrid: Meshgrid of depth

    :return: Dictionary with interpolated temperature and salinity data
    """
    ds = xr.open_dataset(filepath, drop_variables=['compass_timeouts_times_truck'])

    time = ds.time_raw
    mean_time = pd.to_datetime(time.values).mean()
    mean_time_pd = pd.to_datetime(mean_time)
    lon = ds.longitude
    depth = ds.depth

    temp_interp = interp_to_grid(lon, depth, ds.temp_raw, Xgrid, Ygrid)
    salt_interp = interp_to_grid(lon, depth, ds.salt_raw, Xgrid, Ygrid)

    temp_profile = np.nanmean(temp_interp, axis=1)
    salt_profile = np.nanmean(salt_interp, axis=1)

    if "salt_corrected" in ds:
        salt_corrected_interp = interp_to_grid(lon, depth, ds.salt_corrected, Xgrid, Ygrid)
        salt_profile = np.nanmean(salt_corrected_interp, axis=1)

    out = {
        "lon": lon,
        "temp_profile": temp_profile,
        "salt_profile": salt_profile,
        "mean_time": mean_time_pd,
    }

    return out

def process_transects(filepaths):
    """
    Process transects and create dictionaries of temperature and salinity.
    
    :param filepaths: Filepaths to merged transect netCDF files
    
    :return: Tuple of (results, temps, salts)
        - results: Dictionary with keys as transect names and values as transect data
    """
    Xgrid, Ygrid = make_transect_grid()

    results = {}
    for i, fp in enumerate(filepaths, start=1):
        # Extract the base filename without extension
        base = os.path.basename(fp)          # '10_25_b_merged.nc'
        name = base.split('_merged')[0]      # '10_25_b'
        
        print(f"Processing {i}/{len(filepaths)} {name}...")
        results[name] = transect(fp, Xgrid, Ygrid)

    # 
    temps = {
        k: {
            "temp": v['temp_profile'],
            "depth": Ygrid[:,0],
            "mean_time": v["mean_time"],
        }
        for k, v in results.items()
    }
    salts = {
    k: {
        "salt": v["salt_profile"],
        "depth": Ygrid[:,0],
        "mean_time": v["mean_time"],
    }
    for k, v in results.items()
    }

    print(f"Processing complete.\n")
    return results, temps, salts

# def timeseries_grid(time_min, time_max, temp_anom, depth):
#     xgrid = np.arange(time_min, time_max, np.timedelta64(30, 'D')) # Every 30 days in time
#     ygrid = np.arange(0,1000,5) # Every 5m in depth
    
#     Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

#     return griddata(
#         points=(temp_anom.values, depth.values),
#         values=values.values,
#         xi=(Xgrid, Ygrid),
#         method="linear"
#     )



