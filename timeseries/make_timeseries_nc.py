import os
import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata
from utils import woa_temp, woa_salt, anomaly
from utils.transects_func import process_transects

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
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect4/7_24_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect5/8_24_a_merged.nc',
    r'C:/Users/marqjace/data/seaglider/TH_line/deployments/apr_2024/transect6/8_24_b_merged.nc',

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
]

def main():
    results, temp_transects, salt_transects = process_transects(filepaths)

    #################### Temperature Anomaly Calculation ##################
    woa_temp_months = {
        '1': woa_temp.woa_temp_jan,
        '2': woa_temp.woa_temp_feb,
        '3': woa_temp.woa_temp_mar,
        '4': woa_temp.woa_temp_apr,
        '5': woa_temp.woa_temp_may,
        '6': woa_temp.woa_temp_jun,
        '7': woa_temp.woa_temp_jul,
        '8': woa_temp.woa_temp_aug,
        '9': woa_temp.woa_temp_sep,
        '10': woa_temp.woa_temp_oct,
        '11': woa_temp.woa_temp_nov,
        '12': woa_temp.woa_temp_dec
    }

    temp_anom = anomaly.temperature_anomaly(temp_transects, woa_temp_months)
    
    print('Creating a mean depth profile for each transect...')
    for transect, data in temp_anom.items():
        temp_anom[transect] = {
            "profile": np.nanmean(data['temp_anomaly'], axis=1),   # Creates a profile of the mean temperature anomaly values across depth
            "mean_time": data['mean_time'],
        }

    depth = np.linspace(0,1000,200)
    min_time = min(v["mean_time"] for v in temp_anom.values())
    max_time = max(v["mean_time"] for v in temp_anom.values())

    # Create time vs depth grid for interpolation
    time_grid = np.arange(min_time, max_time, pd.Timedelta(days=30)) # Time grid: every 30 days
    depth_grid = np.arange(0, 1000, 5) # Depth grid: every 5 m
    Tgrid, Zgrid = np.meshgrid(time_grid, depth_grid) # Meshgrid

    times_temp = []
    depths_temp = []
    values_temp = []

    for v in temp_anom.values():
        t = v["mean_time"]
        profile = v["profile"]
        times_temp.extend([t] * len(profile))
        depths_temp.extend(depth)
        values_temp.extend(profile)

    # Convert to numpy.datetime64
    times_Temp = np.array([np.datetime64(t) for t in times_temp])
    depths_Temp = np.array(depths_temp)
    values_Temp = np.array(values_temp)

    # Numeric times for griddata
    times_numeric_Temp = (times_Temp - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D')
    Tgrid_numeric_Temp = (Tgrid - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    Zgrid_numeric_Temp = Zgrid.astype(float)

    # Linear interpolation onto grid
    tanom_grid = griddata(
        points=(times_numeric_Temp, depths_Temp),
        values=values_Temp,
        xi=(Tgrid_numeric_Temp, Zgrid),
        method='linear'
    )

    # Pull out surface values
    surface = tanom_grid[0, :]

    # Create artificial layers at -5m and -10m depth
    surface_5m = surface.copy()
    surface_10m = surface.copy()

    # Stack above surface
    tanom_np = np.vstack([
        surface_10m,
        surface_5m,
        tanom_grid
    ])

    # Extend depth grid
    depth_grid_extended = np.concatenate(([-10, -5], depth_grid))

    # Replace surface with 5 m values
    tanom_np[2, :] = tanom_np[3, :]

    # Convert to Pandas DataFrame for rolling filters
    tanom_grid = pd.DataFrame(tanom_np, index=depth_grid_extended)

    ###################### Salinity Anomaly Calculations ##################

    woa_salt_months = {
        '1': woa_salt.woa_salt_jan,
        '2': woa_salt.woa_salt_feb,
        '3': woa_salt.woa_salt_mar,
        '4': woa_salt.woa_salt_apr,
        '5': woa_salt.woa_salt_may,
        '6': woa_salt.woa_salt_jun,
        '7': woa_salt.woa_salt_jul,
        '8': woa_salt.woa_salt_aug,
        '9': woa_salt.woa_salt_sep,
        '10': woa_salt.woa_salt_oct,
        '11': woa_salt.woa_salt_nov,
        '12': woa_salt.woa_salt_dec
    }

    salt_anom = anomaly.salinity_anomaly(salt_transects, woa_salt_months)
    
    print('Creating a mean depth profile for each transect...')
    for transect, data in salt_anom.items():
        salt_anom[transect] = {
            "profile": np.nanmean(data['salt_anomaly'], axis=1),   # Creates a profile of the mean salinity anomaly values across depth
            "mean_time": data['mean_time'],
        }

    depth = np.linspace(0,1000,200)
    min_time = min(v["mean_time"] for v in salt_anom.values())
    max_time = max(v["mean_time"] for v in salt_anom.values())

    # Create time vs depth grid for interpolation
    time_grid = np.arange(min_time, max_time, pd.Timedelta(days=30)) # Time grid: every 30 days
    depth_grid = np.arange(0, 1000, 5) # Depth grid: every 5 m
    Tgrid, Zgrid = np.meshgrid(time_grid, depth_grid) # Meshgrid

    times_temp = []
    depths_temp = []
    values_temp = []

    for v in salt_anom.values():
        t = v["mean_time"]
        profile = v["profile"]
        times_temp.extend([t] * len(profile))
        depths_temp.extend(depth)
        values_temp.extend(profile)

    # Convert to numpy.datetime64
    times_Salt = np.array([np.datetime64(t) for t in times_temp])
    depths_Salt = np.array(depths_temp)
    values_Salt = np.array(values_temp)

    # Numeric times for griddata
    times_numeric_Salt = (times_Salt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 'D')
    Tgrid_numeric_Salt = (Tgrid - np.datetime64('1970-01-01')) / np.timedelta64(1, 'D')
    Zgrid_numeric_Salt = Zgrid.astype(float)

    # Linear interpolation onto grid
    sanom_grid = griddata(
        points=(times_numeric_Salt, depths_Salt),
        values=values_Salt,
        xi=(Tgrid_numeric_Salt, Zgrid_numeric_Salt),
        method='linear'
    )

    # Pull out surface values
    surface = sanom_grid[0, :]

    # Create artificial layers at -5m and -10m depth
    surface_5m = surface.copy()
    surface_10m = surface.copy()

    # Stack above surface
    sanom_np = np.vstack([
        surface_10m,
        surface_5m,
        sanom_grid
    ])

    # Extend depth grid
    depth_grid_extended = np.concatenate(([-10, -5], depth_grid))

    # Replace surface with 5 m values
    sanom_np[2, :] = sanom_np[3, :]

    # Convert to Pandas DataFrame for rolling filters
    sanom_grid = pd.DataFrame(sanom_np, index=depth_grid_extended)


    ############### Save as xarray Dataset #################

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
            'source': 'Seaglider transects processed by Oregon State University Glider Research Group',
            'created_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'contact': 'Jace Marquardt (jace.marquardt@oregonstate.edu)',
            'references': 'World Ocean Atlas 2018 Temperature Data'
        }
    )

    data_path = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\timeseries\data'
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    output_file = os.path.join(data_path, 'timeseries_anomaly.nc')
    anom_ds.to_netcdf(output_file)

    print(f"Saved temperature anomaly grid to {output_file}")


if __name__ == "__main__":
    main()