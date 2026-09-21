import os
import numpy as np
import pandas as pd
import xarray as xr
from rich import print
from datetime import datetime
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress


def main():

    ##################### Load Data #####################
    filepath = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\timeseries\data\timeseries_anomaly.nc'
    print(f'\nLoading data from "{filepath}"....')

    ds = xr.open_dataset(filepath)
    ds = ds.sortby('time')

    # # --- Optional time selection ---
    # time_mask = (ds['time'] >= np.datetime64('2025-11-01')) & (ds['time'] <= np.datetime64('2026-04-01'))
    # ds = ds.sel(time=time_mask)

    time = ds['time'].values
    latest_transect_time = pd.to_datetime(time.max())
    depth = ds['depth'].values
    tanom_smoothed = ds['temperature_anomaly'].rolling(time=3, depth=4, min_periods=1).mean()
    sanom_smoothed = ds['salinity_anomaly'].rolling(time=3, depth=4, min_periods=1).mean()
    print(f'\nApplying smoothing...')

    # Extract 50m depth for thi index
    tanom_50m = tanom_smoothed.sel(depth=50, method="nearest")
    thi_time = pd.to_datetime(tanom_50m.time.values)
    fifty_meters = tanom_50m.values

    # Set boundaries and levels for plotting
    boundaries_temp = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    levels_temp = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    divnorm_temp=colors.TwoSlopeNorm(vcenter=0., vmin=-4, vmax=4)

    boundaries_salt = [-.6, -.4, -.2, 0, .2, .4, .6]
    levels_salt = [-.6, -.4, -.2, .2, .4, .6]
    divnorm_salt=colors.TwoSlopeNorm(vcenter=0., vmin=-.75, vmax=.75)

    # Calculate current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Timestamp for file naming
    timestamp_print = datetime.now().strftime("%Y-%m-%d") # Timestamp for printing

    # Create figures directory if it doesn't exist
    figures_directory = f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Repositories/TH_Line/timeseries/figures/'
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)


    ##################### Plot Timeseries #####################
    print(f'\nCreating figures....')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13,8), dpi=300, layout='constrained')

    plot1 = ax1.pcolormesh(time, depth, tanom_smoothed, cmap='bwr', norm=divnorm_temp, shading='nearest')
    lines1 = ax1.contour(time, depth, tanom_smoothed, colors='black', norm=divnorm_temp, levels=levels_temp, alpha=0.75)

    dep_nov_14 = ax1.hlines(y=570, xmin=datetime(2014,12,4), xmax=datetime(2015,3,9), color='k')
    dep_mar_15 = ax1.hlines(y=570, xmin=datetime(2015,3,9), xmax=datetime(2015,9,17), color='k')
    dep_sep_15 = ax1.hlines(y=570, xmin=datetime(2015,9,17), xmax=datetime(2016,5,16), color='k')
    dep_may_16 = ax1.hlines(y=570, xmin=datetime(2016,5,23), xmax=datetime(2016,10,21), color='k')
    dep_oct_16 = ax1.hlines(y=570, xmin=datetime(2016,10,21), xmax=datetime(2017,6,5), color='k')
    dep_jun_17 = ax1.hlines(y=570, xmin=datetime(2017,6,5), xmax=datetime(2017,11,6), color='k')
    dep_apr_18 = ax1.hlines(y=570, xmin=datetime(2018,4,17), xmax=datetime(2018,10,2), color='k')
    dep_nov_18 = ax1.hlines(y=570, xmin=datetime(2018,11,7), xmax=datetime(2019,4,9), color='k')
    dep_apr_19 = ax1.hlines(y=570, xmin=datetime(2019,4,9), xmax=datetime(2019,8,19), color='k')
    dep_sep_19 = ax1.hlines(y=570, xmin=datetime(2019,9,16), xmax=datetime(2020,3,19), color='k')
    dep_sep_20 = ax1.hlines(y=570, xmin=datetime(2020,9,16), xmax=datetime(2021,2,6), color='k')
    dep_nov_21 = ax1.hlines(y=570, xmin=datetime(2021,11,12), xmax=datetime(2022,6,16), color='k')
    dep_jul_22 = ax1.hlines(y=570, xmin=datetime(2022,7,29), xmax=datetime(2023,1,26), color='k')
    dep_jan_23 = ax1.hlines(y=570, xmin=datetime(2023,1,26), xmax=datetime(2023,8,14), color='k')
    dep_oct_23 = ax1.hlines(y=570, xmin=datetime(2023,10,13), xmax=datetime(2024,4,12), color='k')
    dep_apr_24 = ax1.hlines(y=570, xmin=datetime(2024,4,12), xmax=datetime(2024,8,9), color='k')
    dep_oct_24 = ax1.hlines(y=570, xmin=datetime(2024,10,21), xmax=datetime(2024,12,4), color='k')
    dep_mar_25 = ax1.hlines(y=570, xmin=datetime(2025,3,21), xmax=datetime(2025,11,11), color='k')
    dep_nov_25 = ax1.hlines(y=570, xmin=datetime(2025,11,11), xmax=datetime(2026,2,25), color='k')
    dep_mar_26 = ax1.hlines(y=570, xmin=datetime(2026,3,14), xmax=datetime(2026,8,27), color='k')
    dep_sep_26 = ax1.hlines(y=570, xmin=datetime(2026,9,18), xmax=datetime.now(), color='k')

    ax1.clabel(lines1, lines1.levels, inline=True, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_yticks((0, 200, 400, 600))
    ax1.set_ylim(600, 0)
    ax1.set_xlim(time.min(), time.max() + pd.Timedelta(30, unit='D'))
    ax1.set_ylabel('Depth (m)', fontsize='large')
    ax1.spines[:].set_linewidth(2)
    ax1.tick_params(width=4, top=True, right=True, direction='in', labelsize=12)
    cbar1 = fig.colorbar(plot1, ax=ax1, pad=0.02, shrink=0.75)
    cbar1.outline.set_linewidth(2)
    cbar1.set_label(label=r'($\degree$C)', rotation=0, labelpad=10)

    plot2 = ax2.pcolormesh(time, depth, sanom_smoothed, cmap='BrBG_r', norm=divnorm_salt, shading='nearest')
    lines2 = ax2.contour(time, depth, sanom_smoothed, colors='black', norm=divnorm_salt, levels=levels_salt, alpha=0.75)

    dep2_nov_14 = ax2.hlines(y=570, xmin=datetime(2014,12,4), xmax=datetime(2015,3,9), color='k')
    dep2_mar_15 = ax2.hlines(y=570, xmin=datetime(2015,3,9), xmax=datetime(2015,9,17), color='k')
    dep2_sep_15 = ax2.hlines(y=570, xmin=datetime(2015,9,17), xmax=datetime(2016,5,16), color='k')
    dep2_may_16 = ax2.hlines(y=570, xmin=datetime(2016,5,23), xmax=datetime(2016,10,21), color='k')
    dep2_oct_16 = ax2.hlines(y=570, xmin=datetime(2016,10,21), xmax=datetime(2017,6,5), color='k')
    dep2_jun_17 = ax2.hlines(y=570, xmin=datetime(2017,6,5), xmax=datetime(2017,11,6), color='k')
    dep2_apr_18 = ax2.hlines(y=570, xmin=datetime(2018,4,17), xmax=datetime(2018,10,2), color='k')
    dep2_nov_18 = ax2.hlines(y=570, xmin=datetime(2018,11,7), xmax=datetime(2019,4,9), color='k')
    dep2_apr_19 = ax2.hlines(y=570, xmin=datetime(2019,4,9), xmax=datetime(2019,8,19), color='k')
    dep2_sep_19 = ax2.hlines(y=570, xmin=datetime(2019,9,16), xmax=datetime(2020,3,19), color='k')
    dep2_sep_20 = ax2.hlines(y=570, xmin=datetime(2020,9,16), xmax=datetime(2021,2,6), color='k')
    dep2_nov_21 = ax2.hlines(y=570, xmin=datetime(2021,11,12), xmax=datetime(2022,6,16), color='k')
    dep2_jul_22 = ax2.hlines(y=570, xmin=datetime(2022,7,29), xmax=datetime(2023,1,26), color='k')
    dep2_jan_23 = ax2.hlines(y=570, xmin=datetime(2023,1,26), xmax=datetime(2023,8,14), color='k')
    dep2_oct_23 = ax2.hlines(y=570, xmin=datetime(2023,10,13), xmax=datetime(2024,4,12), color='k')
    dep2_apr_24 = ax2.hlines(y=570, xmin=datetime(2024,4,12), xmax=datetime(2024,8,9), color='k')
    dep2_oct_24 = ax2.hlines(y=570, xmin=datetime(2024,10,21), xmax=datetime(2024,12,4), color='k')
    dep2_mar_25 = ax2.hlines(y=570, xmin=datetime(2025,3,21), xmax=datetime(2025,11,11), color='k')
    dep2_nov_25 = ax2.hlines(y=570, xmin=datetime(2025,11,11), xmax=datetime(2026,2,25), color='k')
    dep2_mar_26 = ax2.hlines(y=570, xmin=datetime(2026,3,14), xmax=datetime(2026,8,27), color='k')
    dep2_sep_26 = ax2.hlines(y=570, xmin=datetime(2026,9,18), xmax=datetime.now(), color='k')

    ax2.clabel(lines2, lines2.levels, inline=True, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_yticks((0, 200, 400, 600))
    ax2.set_ylim(600, 0)
    ax2.set_xlim(time.min(), time.max() + pd.Timedelta(30, unit='D'))
    ax2.set_xlabel('Time', fontsize='large')
    ax2.set_ylabel('Depth (m)', fontsize='large')
    ax2.spines[:].set_linewidth(2)
    ax2.tick_params(width=4, top=True, right=True, direction='in', labelsize=12)
    cbar2 = fig.colorbar(plot2, ax=ax2, pad=0.02, shrink=0.75)
    cbar2.outline.set_linewidth(2)
    cbar2.set_label(label=r'(PSU)', rotation=0, labelpad=10)

    fig.suptitle(
        f'Temperature (top) and salinity (bottom) anomalies off Trinidad Head, California\nAveraged over 200 km from the coast',
        fontsize='xx-large',
    )

    fig.text(0.8, 0.015, f'Latest transect date: {latest_transect_time.strftime("%Y-%m-%d")}', fontsize='small')
    fig.text(0.025, 0.015, f'Courtesy of the Oregon State University Glider Research Group, NANOOS, CeNCOOS', fontsize='small')

    fig.get_layout_engine().set(
        w_pad=0.15,    # Horizontal padding in inches
        h_pad=0.15,    # Vertical padding in inches
        wspace=0.05,  # 5% width spacing between columns
        hspace=0.01,   # Reduce spacing between stacked subplots
    )


    plt.savefig(os.path.join(figures_directory, f't_anom_timeseries_{timestamp}.png'))
    print(f'Figure saved to "{figures_directory}t_anom_timeseries_{timestamp}.png"\n')


    ##################### Load Indices Data #####################
    # SCTI / ONI Data
    # Data Access Here: https://spraydata.ucsd.edu/products/socal-index/

    with xr.open_dataset(
        r'C:/Users/marqjace/data/seaglider/TH_line/scti_oni/socal_index_monthly_v1_8571_f367_229e_U1788907474988.nc',
        decode_times=True
    ) as dat:
        scti = dat['scti']
        oni = dat['oni']
        scti_time = pd.to_datetime(dat['time'].values)

    # California MOCI
    # García-Reyes, M. and Sydeman, W.J. (2017). California Multivariate Ocean Climate Indicator (MOCI) [Data set, V2]. Farallon Institute website, http://www.faralloninstitute.org/moci. Accessed [28 May 2025].
    with open(r'C:/Users/marqjace/data/seaglider/TH_line/california_moci/CaliforniaMOCI.csv', 'r') as file:
        dat2 = pd.read_csv(file)
    dat2 = dat2.drop(['Year', 'Season', 'Central California (34.5-38N)', 'Southern California (32-34.5N)'], axis=1)
    dat2 = dat2.set_index(['time'])

    norcal_moci = dat2['North California (38-42N)']
    norcal_time = pd.to_datetime(norcal_moci.index)


    ##################### Plot with MOCI Indices #####################
    print(f'Plotting t_anom_indices_MOCI_{timestamp}.png....')

    fig, ax = plt.subplots(1, 1, figsize=(18, 7), dpi=300)
    ax2 = ax.twinx()

    # --- Plot indices ---
    oni_plot = ax.plot(
        scti_time,
        oni,
        label='Oceanic Niño Index (NOAA)',
        color='k',
        linewidth=2
    )

    scti_plot = ax.plot(
        scti_time,
        scti,
        label='So Cal T Index (Rudnick)',
        color='blue',
        linewidth=2
    )

    thi_plot = ax.plot(
        thi_time,
        fifty_meters,
        label='Trinidad Head Index (Barth)',
        color='magenta',
        linewidth=2
    )

    moci_plot = ax2.plot(
        norcal_time,
        norcal_moci,
        label='California Multivariate Ocean Climate Indicator',
        color='green',
        linewidth=2
    )

    # --- Axis formatting ---
    ax.set_xlabel('Year', fontsize='x-large', labelpad=15)
    ax.set_ylabel(r'Temperature Anomaly ($\degree$C)', fontsize='x-large', labelpad=10)
    ax2.set_ylabel('MOCI Index', fontsize='x-large', labelpad=10)

    ax.set_xlim(scti_time.min(), thi_time.max())
    ax.set_ylim(-2,4)
    ax2.set_ylim(-8, 12)
    ax2.set_yticks([-8, -4, 0, 4, 8, 12, 16])

    # --- Zero line + background shading ---
    ax.axhline(0, color='k', linewidth=1, alpha=1)

    # --- Styling ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.tick_params(width=2, length=10, top=False, right=False, left=False, direction='out')
    ax2.tick_params(width=2, length=10, top=False, right=False, left=False, direction='out')

    # --- Legend (combined) ---
    lns = oni_plot + scti_plot + thi_plot + moci_plot
    labs = [l.get_label() for l in lns]
    ax.legend(
        lns,
        labs,
        loc='upper left',
        frameon=False,
        fontsize='large',
        labelcolor='linecolor'
    )

    plt.grid(alpha=0.8, which='major', axis='y')
    plt.title('California Temperature Anomaly Indices', pad=15, fontsize='xx-large')
    fig.text(0.8, 0.025, f'Latest Data: {thi_time.max().strftime("%Y-%m-%d")}', fontsize='large')
    fig.subplots_adjust(bottom=0.15)

    plt.savefig(
        os.path.join(figures_directory, f't_anom_indices_MOCI_{timestamp}.png')
    )

    print(f'Figure saved to "{figures_directory}t_anom_indices_MOCI_{timestamp}.png"\n')


    ##################### Plot NO MOCI Indices #####################
    print(f'Plotting t_anom_indices_{timestamp}.png....')

    fig, ax = plt.subplots(1, 1, figsize=(18, 7), dpi=300)

    # --- Plot indices ---
    oni_plot = ax.plot(
        scti_time,
        oni,
        label='Oceanic Niño Index (NOAA)',
        color='k',
        linewidth=2
    )

    scti_plot = ax.plot(
        scti_time,
        scti,
        label='So Cal T Index (Rudnick)',
        color='blue',
        linewidth=2
    )

    thi_plot = ax.plot(
        thi_time,
        fifty_meters,
        label='Trinidad Head Index (Barth)',
        color='magenta',
        linewidth=2
    )

    # --- Axis formatting ---
    ax.set_xlabel('Year', fontsize='x-large', labelpad=15)
    ax.set_ylabel(r'Temperature Anomaly ($\degree$C)', fontsize='x-large', labelpad=10)

    ax.set_xlim(scti_time.min(), thi_time.max())
    ax.set_ylim(-2,4)

    # --- Zero line + background shading ---
    ax.axhline(0, color='k', linewidth=1, alpha=1)

    # --- Styling ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(width=2, length=10, top=False, right=False, left=False, direction='out')

    # --- Legend (combined) ---
    lns = oni_plot + scti_plot + thi_plot
    labs = [l.get_label() for l in lns]
    ax.legend(
        lns,
        labs,
        loc='upper left',
        frameon=False,
        fontsize='large',
        labelcolor='linecolor'
    )

    plt.grid(alpha=0.8, which='major', axis='y')
    plt.title('California Temperature Anomaly Indices', pad=15, fontsize='xx-large')
    fig.text(0.8, 0.025, f'Latest Data: {thi_time.max().strftime("%Y-%m-%d")}', fontsize='large')
    fig.subplots_adjust(bottom=0.15)

    plt.savefig(
        os.path.join(figures_directory, f't_anom_indices_{timestamp}.png')
    )

    print(f'Figure saved to "{figures_directory}t_anom_indices_{timestamp}.png"\n')

    print('\nDone!')

if __name__ == "__main__":
    main()