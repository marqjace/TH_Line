import os
import argparse
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata
from utils import woa_temp, woa_salt, anomaly
from utils.transects_func import process_transects
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
DEPTH_MAX = 1000
DEPTH_BINS = 200
DEPTH_GRID_STEP = 5
TIME_FREQ = '30D'
SURFACE_LAYERS = [-10, -5]

filepaths = [

    # Nov 2014 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2014/transect1/12_14_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2014/transect2/12_14_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2014/transect3/1_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2014/transect4/2_15_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2014/transect5/2_15_b_merged.nc',

    # Mar 2015 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect1/3_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect2/4_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect3/5_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect4/6_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect5/7_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect6/8_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2015/transect7/9_15_merged.nc',

    # Sep 2015 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect1/10_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect2/11_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect3/12_15_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect4/1_16_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect5/3_16_merged.nc', # Something wrong with this transect
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect6/4_16_merged.nc',
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2015/transect7/5_16_merged.nc',

    # May 2016 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect1/6_16_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect2/7_16_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect3/8_16_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect4/9_16_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect5/9_16_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect6/10_16_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/may_2016/transect7/10_16_b_merged.nc',

    # Oct 2016 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect1/11_16_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect2/12_16_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect3/1_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect4/2_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect5/3_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect6/4_17_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect7/4_17_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2016/transect8/5_17_merged.nc',

    # Jun 2017 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect1/6_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect2/7_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect3/8_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect4/9_17_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect5/10_17_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect6/10_17_b_merged.nc',
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jun_2017/transect7/11_17_merged.nc', # Something wrong with this transect

    # Apr 2018 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2018/transect1/4_18_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2018/transect2/5_18_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2018/transect3/6_18_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2018/transect4/8_18_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2018/transect5/9_18_a_merged.nc',
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2018/transect6/9_18_b_merged.nc', # Something wrong with this transect

    # Nov 2018 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect1/11_18_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect2/12_18_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect3/1_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect4/2_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect5/3_19_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect6/3_19_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2018/transect7/4_19_a_merged.nc',

    # Apr 2019 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2019/transect1/4_19_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2019/transect2/6_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2019/transect3/7_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2019/transect4/8_19_merged.nc',

    # Sep 2019 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect1/9_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect2/10_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect3/11_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect4/12_19_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect5/1_20_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect6/2_20_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect7/3_20_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2019/transect8/3_20_b_merged.nc',

    # Sep 2020 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2020/transect1/9_20_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2020/transect2/10_20_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2020/transect3/11_20_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2020/transect4/12_20_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2020/transect5/1_21_merged.nc',
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2020/transect6/2_21_merged.nc', # Something wrong with this transect

    # Nov 2021 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect1/11_21_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect2/12_21_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect3/1_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect4/2_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect5/3_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect6/4_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect7/5_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2021/transect8/6_22_merged.nc',

    # Jul 2022 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jul_2022/transect1/8_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jul_2022/transect2/9_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jul_2022/transect3/10_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jul_2022/transect4/11_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jul_2022/transect5/12_22_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jul_2022/transect6/1_23_merged.nc',

    # Jan 2023 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect1/2_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect2/3_23_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect3/3_23_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect4/4_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect5/5_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect6/6_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect7/7_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/jan_2023/transect8/8_23_merged.nc',

    # Oct 2023 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect1/10_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect2/11_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect3/12_23_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect4/1_24_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect5/2_24_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect6/2_24_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2023/transect7/3_24_merged.nc',

    # Apr 2024 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect1/4_24_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect2/5_24_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect3/6_24_merged.nc',
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect4/7_24_merged.nc', # Bad salinity data
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect5/8_24_a_merged.nc', # Bad salinity data
    # r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect6/8_24_b_merged.nc', # Bad salinity data

    # Oct 2024 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2024/transect1/10_24_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/oct_2024/transect2/11_24_merged.nc',

    # Mar 2025 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect1/corrected/3_25_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect2/corrected/4_25_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect3/corrected/4_25_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect4/corrected/5_25_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect5/corrected/6_25_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect6/corrected/6_25_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect7/corrected/7_25_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect8/corrected/8_25_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect9/corrected/9_25_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect10/corrected/9_25_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect11/corrected/10_25_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2025/transect12/corrected/10_25_b_merged.nc',

    # Nov 2025 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect1/11_25_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect2_b/12_25_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect3/1_26_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect4/2_26_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect5/2_26_b_merged.nc',

    # Mar 2026 Deployment
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect1/3_26_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect2/4_26_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect3/5_26_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect4/5_26_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect5/6_26_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect6/7_26_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect7/7_26_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect8/7_26_c_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect9/8_26_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/mar_2026/transect10/8_26_b_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/sep_2026/transect1/9_26_a_merged.nc',
]


def compute_mean_depth_profile(anomaly_dict):
    """
    Compute mean depth profile for each transect from anomaly data.
    
    Parameters
    ----------
    anomaly_dict : dict
        Dictionary with transect names as keys and anomaly data as values
        
    Returns
    -------
    dict
        Dictionary with transect names as keys and profile/mean_time as values
    """
    print('Creating a mean depth profile for each transect...')
    profiles = {}
    for transect, data in anomaly_dict.items():
        # Get the appropriate key ('temp_anomaly' or 'salt_anomaly')
        anomaly_key = [k for k in data.keys() if k.endswith('_anomaly')][0]
        profiles[transect] = {
            "profile": np.nanmean(data[anomaly_key], axis=1),
            "mean_time": data['mean_time'],
        }
    return profiles


def create_interpolated_grid(profile_dict):
    """
    Create an interpolated grid from profile data.
    
    Parameters
    ----------
    profile_dict : dict
        Dictionary containing profile and mean_time for each transect
        
    Returns
    -------
    tuple
        (pd.DataFrame, pd.DatetimeIndex) - Interpolated grid and time_grid
    """
    # Define depth array for profiles
    depth = np.linspace(0, DEPTH_MAX, DEPTH_BINS)
    
    # Get time range
    min_time = min(v["mean_time"] for v in profile_dict.values())
    max_time = max(v["mean_time"] for v in profile_dict.values())

    # Create time vs depth grid for interpolation.
    # Preserve the true latest transect date even when it does not fall on the
    # fixed 30-day spacing used for the interpolated grid.
    time_grid = pd.date_range(start=min_time, end=max_time, freq=TIME_FREQ)
    max_time = pd.Timestamp(max_time)
    if len(time_grid) == 0 or time_grid[-1] < max_time:
        time_grid = time_grid.union(pd.DatetimeIndex([max_time]))

    depth_grid = np.arange(0, DEPTH_MAX, DEPTH_GRID_STEP)
    Tgrid, Zgrid = np.meshgrid(time_grid, depth_grid)

    # Pre-allocate arrays for better performance
    n_profiles = len(profile_dict)
    n_points = n_profiles * len(depth)
    
    times_array = np.empty(n_points, dtype='datetime64[ns]')
    depths_array = np.empty(n_points, dtype=float)
    values_array = np.empty(n_points, dtype=float)
    
    # Fill arrays efficiently
    idx = 0
    for v in profile_dict.values():
        t = v["mean_time"]
        profile = v["profile"]
        n = len(profile)
        times_array[idx:idx+n] = np.datetime64(t)
        depths_array[idx:idx+n] = depth
        values_array[idx:idx+n] = profile
        idx += n

    # Convert to numeric for griddata
    times_numeric = (times_array - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D')
    Tgrid_numeric = (Tgrid - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')

    # Linear interpolation onto grid
    grid_interpolated = griddata(
        points=(times_numeric, depths_array),
        values=values_array,
        xi=(Tgrid_numeric, Zgrid),
        method='linear'
    )

    # Add surface layers
    surface = grid_interpolated[0, :]
    grid_with_surface = np.vstack([surface.copy(), surface.copy(), grid_interpolated])
    
    # Replace surface with 5m values
    grid_with_surface[2, :] = grid_with_surface[3, :]
    
    # Extend depth grid with surface layers
    depth_grid_extended = np.concatenate((SURFACE_LAYERS, depth_grid))

    # Convert to DataFrame
    return pd.DataFrame(grid_with_surface, index=depth_grid_extended), time_grid


def process_anomaly_data(transects, woa_months, anomaly_func):
    """
    Process transect data to create interpolated anomaly grid.
    
    Parameters
    ----------
    transects : dict
        Dictionary of transect data
    woa_months : dict
        Dictionary of WOA monthly climatology data
    anomaly_func : callable
        Function to calculate anomaly (temperature_anomaly or salinity_anomaly)
        
    Returns
    -------
    tuple
        (pd.DataFrame, pd.DatetimeIndex) - Interpolated grid and time_grid
    """
    # Calculate anomaly
    anomaly_dict = anomaly_func(transects, woa_months)
    
    # Compute mean depth profiles
    profile_dict = compute_mean_depth_profile(anomaly_dict)
    
    # Create interpolated grid and return time_grid
    grid, time_grid = create_interpolated_grid(profile_dict)
    
    return grid, time_grid


def get_processed_filepaths(output_file):
    """
    Load list of previously processed filepaths from existing netCDF file.
    
    Parameters
    ----------
    output_file : str
        Path to the output netCDF file
        
    Returns
    -------
    set
        Set of processed filepaths, or empty set if file doesn't exist
    """
    if not os.path.exists(output_file):
        return set()
    
    try:
        ds = xr.open_dataset(output_file)
        if 'processed_filepaths' in ds.attrs:
            # Split by delimiter and return as set
            processed = set(ds.attrs['processed_filepaths'].split('||'))
            ds.close()
            return processed
        ds.close()
    except Exception as e:
        print(f"Warning: Could not read processed filepaths from {output_file}: {e}")
    
    return set()


def find_new_filepaths(all_filepaths, processed_filepaths):
    """
    Find filepaths that haven't been processed yet.
    
    Parameters
    ----------
    all_filepaths : list
        List of all filepaths to potentially process
    processed_filepaths : set
        Set of already processed filepaths
        
    Returns
    -------
    list
        List of new filepaths that need processing
    """
    return [fp for fp in all_filepaths if fp not in processed_filepaths]


def main(force_rebuild=False):
    # Define output file path
    data_path = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\timeseries\data'
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    output_file = os.path.join(data_path, 'timeseries_anomaly.nc')
    
    # Check for previously processed filepaths
    processed_filepaths = get_processed_filepaths(output_file)
    new_filepaths = find_new_filepaths(filepaths, processed_filepaths)
    
    if not new_filepaths and processed_filepaths and not force_rebuild:
        print(f"No new transects to process. Output file is up to date: {output_file}")
        print(f"Total transects already processed: {len(processed_filepaths)}")
        return
    
    if new_filepaths:
        print(f"Found {len(new_filepaths)} new transects to process:")
        for fp in new_filepaths[:5]:  # Show first 5
            print(f"  - {os.path.basename(fp)}")
        if len(new_filepaths) > 5:
            print(f"  ... and {len(new_filepaths) - 5} more")
    
    if force_rebuild and processed_filepaths:
        print("Force rebuild requested. Reprocessing all transects...")

    # Process all transects (including previously processed ones for consistent interpolation)
    print(f"\nProcessing all {len(filepaths)} transects to create consistent interpolated grid...")
    results, temp_transects, salt_transects = process_transects(filepaths)

    # Define WOA dictionaries
    woa_temp_months = {
        '1': woa_temp.woa_temp_jan, '2': woa_temp.woa_temp_feb,
        '3': woa_temp.woa_temp_mar, '4': woa_temp.woa_temp_apr,
        '5': woa_temp.woa_temp_may, '6': woa_temp.woa_temp_jun,
        '7': woa_temp.woa_temp_jul, '8': woa_temp.woa_temp_aug,
        '9': woa_temp.woa_temp_sep, '10': woa_temp.woa_temp_oct,
        '11': woa_temp.woa_temp_nov, '12': woa_temp.woa_temp_dec
    }

    woa_salt_months = {
        '1': woa_salt.woa_salt_jan, '2': woa_salt.woa_salt_feb,
        '3': woa_salt.woa_salt_mar, '4': woa_salt.woa_salt_apr,
        '5': woa_salt.woa_salt_may, '6': woa_salt.woa_salt_jun,
        '7': woa_salt.woa_salt_jul, '8': woa_salt.woa_salt_aug,
        '9': woa_salt.woa_salt_sep, '10': woa_salt.woa_salt_oct,
        '11': woa_salt.woa_salt_nov, '12': woa_salt.woa_salt_dec
    }

    # Process temperature anomaly
    print('Processing temperature anomaly...')
    tanom_grid, time_grid = process_anomaly_data(
        temp_transects, 
        woa_temp_months, 
        anomaly.temperature_anomaly
    )

    # Process salinity anomaly
    print('Processing salinity anomaly...')
    sanom_grid, _ = process_anomaly_data(
        salt_transects, 
        woa_salt_months, 
        anomaly.salinity_anomaly
    )

    # Create xarray Dataset
    anom_ds = xr.Dataset(
        data_vars={
            'temperature_anomaly': (
                ['depth', 'time'],
                tanom_grid.values,
                {
                    'units': '°C',
                    'description': 'Interpolated temperature anomaly'
                },
            ),
            'salinity_anomaly': (
                ['depth', 'time'],
                sanom_grid.values,
                {
                    'units': 'PSU',
                    'description': 'Interpolated salinity anomaly'
                },
            )
        },
        coords={
            'depth': (
                'depth',
                tanom_grid.index,
                {
                    'units': 'meters',
                    'description': 'Gridded depth below sea surface (5 m bins)',
                }
            ),
            'time': (
                'time',
                time_grid,
                {
                    'description': 'Gridded time (30 day intervals)'
                }
            )
        },
        attrs={
            'title': 'Gridded Trinidad Head Temperature Anomaly Time Series Dataset',
            'source': 'Oregon State University Glider Research Group',
            'created_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'contact': 'Jace Marquardt (jace.marquardt@oregonstate.edu)',
            'references': 'World Ocean Atlas 2018 Temperature Data',
            'processed_filepaths': '||'.join(filepaths),  # Store processed filepaths
            'num_transects': len(filepaths)
        }
    )

    # Save to file
    anom_ds.to_netcdf(output_file)

    print(f"\nSaved anomaly grids to {output_file}")
    print(f"Total transects processed: {len(filepaths)}")
    if new_filepaths:
        print(f"New transects added: {len(new_filepaths)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the output netCDF even when no new transects are detected.",
    )
    args = parser.parse_args()
    main(force_rebuild=args.force)
