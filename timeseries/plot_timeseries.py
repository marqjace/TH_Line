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
    filepath = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\timeseries\data\tanom_timeseries_data.nc'
    print(f'\nLoading data from "{filepath}"...')

    ds = xr.open_dataset(filepath)
    ds = ds.sortby('time')

    time = ds['time'].values
    depth = ds['depth'].values
    tanom_smoothed = ds['temperature_anomaly'].rolling(time=3, depth=4, center=True).mean()
    sanom_smoothed = ds['salinity_anomaly'].rolling(time=3, depth=4, center=True).mean()
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

    # Create figures directory if it doesn't exist
    figures_directory = f'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Repositories/TH_Line/timeseries/figures/'
    if not os.path.isdir(figures_directory):
        os.makedirs(figures_directory, exist_ok=True)

    ##################### Plot Timeseries #####################
    print(f'\nCreating figures...')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,8), dpi=300)

    plot1 = ax1.contourf(time, depth, tanom_smoothed, cmap='RdYlBu_r', norm=divnorm_temp, levels=boundaries_temp)
    lines1 = ax1.contour(time, depth, tanom_smoothed, colors='black', norm=divnorm_temp, levels=levels_temp, alpha=0.75)

    ax1.clabel(lines1, lines1.levels, inline=True, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_yticks((0, 200, 400, 600))
    ax1.set_ylim(600, 0)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Depth (m)')
    ax1.spines[:].set_linewidth(2)
    ax1.tick_params(width=2, top=True, right=True, direction='in')
    ax1.set_title('Trinidad Head Averaged Over Inshore 200km (Filtered)', pad=10)
    cbar1 = plt.colorbar(plot1, shrink=0.5, location='right', pad=0.015)
    cbar1.outline.set_linewidth(2)
    cbar1.set_label(label=r'($\degree$C)', rotation=0, labelpad=10)

    plot2 = ax2.contourf(time, depth, sanom_smoothed, cmap='BrBG_r', norm=divnorm_salt, levels=boundaries_salt)
    lines2 = ax2.contour(time, depth, sanom_smoothed, colors='black', norm=divnorm_salt, levels=levels_salt, alpha=0.75)

    ax2.clabel(lines2, lines2.levels, inline=True, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_yticks((0, 200, 400, 600))
    ax2.set_ylim(600, 0)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Depth (m)')
    ax2.spines[:].set_linewidth(2)
    ax2.tick_params(width=2, top=True, right=True, direction='in')
    cbar2 = plt.colorbar(plot2, shrink=0.5, location='right', pad=0.015)
    cbar2.outline.set_linewidth(2)
    cbar2.set_label(label=r'(PSU)', rotation=0, labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_directory, f't_anom_timeseries_{timestamp}.png'))
    print(f'\nFigure saved to "{figures_directory}t_anom_timeseries_{timestamp}.png"')


    ##################### Load Indices Data #####################
    # SCTI / ONI Data
    # Data Access Here: https://spraydata.ucsd.edu/products/socal-index/

    with xr.open_dataset(
        r'C:/Users/marqjace/data/seaglider/TH_line/scti_oni/socal_index_monthly_v1_8571_f367_229e_U1769796645810.nc',
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
    print(f'Plotting t_anom_indices_MOCI_{timestamp}.png...')

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
        label='Trinidad Head Index',
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
    ax.set_xlabel('Year', fontsize='x-large')
    ax.set_ylabel(r'Temperature Anomaly ($\degree$C)', fontsize='x-large')
    ax2.set_ylabel('MOCI Index', fontsize='x-large')

    ax.set_xlim(pd.Timestamp('2006-06-01'), pd.Timestamp('2026-02-01'))
    ax2.set_ylim(-8, 15)
    ax2.set_yticks([-4, 0, 4, 8, 12])

    # --- Date ticks ---
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Zero line + background shading ---
    ax.axhline(0, color='k', linewidth=1, alpha=0.5)

    ax.axvspan(
        pd.Timestamp('2006-06-01'),
        pd.Timestamp('2026-02-01'),
        ymin=0,
        ymax=0.35,
        alpha=0.15,
        color='gray'
    )

    # --- Styling ---
    ax.spines[:].set_linewidth(2)
    ax2.spines[:].set_linewidth(2)

    ax.tick_params(width=2, top=True, right=False, direction='in')
    ax2.tick_params(width=2, top=True, right=True, direction='in')

    # --- Legend (combined) ---
    lns = oni_plot + scti_plot + thi_plot + moci_plot
    labs = [l.get_label() for l in lns]
    ax.legend(
        lns,
        labs,
        loc='upper left',
        frameon=False,
        fontsize='x-large',
        labelcolor='linecolor'
    )

    plt.title('Temperature Anomaly Indices', pad=15, fontsize='x-large')
    plt.tight_layout()

    plt.savefig(
        os.path.join(figures_directory, f't_anom_indices_MOCI_{timestamp}.png')
    )

    print(f'\nFigure saved to "{figures_directory}t_anom_indices_MOCI_{timestamp}.png"')


    ##################### Plot NO MOCI Indices #####################
    print(f'Plotting t_anom_indices_{timestamp}.png...')

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
        label='Trinidad Head Index',
        color='magenta',
        linewidth=2
    )

    # --- Axis formatting ---
    ax.set_xlabel('Year', fontsize='x-large')
    ax.set_ylabel(r'Temperature Anomaly ($\degree$C)', fontsize='x-large')

    ax.set_xlim(pd.Timestamp('2006-06-01'), pd.Timestamp('2026-02-01'))
    ax2.set_ylim(-8, 15)
    ax2.set_yticks([-4, 0, 4, 8, 12])

    # --- Date ticks ---
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Zero line + background shading ---
    ax.axhline(0, color='k', linewidth=1, alpha=0.5)

    ax.axvspan(
        pd.Timestamp('2006-06-01'),
        pd.Timestamp('2026-02-01'),
        ymin=0,
        ymax=0.35,
        alpha=0.15,
        color='gray'
    )

    # --- Styling ---
    ax.spines[:].set_linewidth(2)
    ax2.spines[:].set_linewidth(2)

    ax.tick_params(width=2, top=True, right=False, direction='in')
    ax2.tick_params(width=2, top=True, right=True, direction='in')

    # --- Legend (combined) ---
    lns = oni_plot + scti_plot + thi_plot
    labs = [l.get_label() for l in lns]
    ax.legend(
        lns,
        labs,
        loc='upper left',
        frameon=False,
        fontsize='x-large',
        labelcolor='linecolor'
    )

    plt.title('Temperature Anomaly Indices', pad=15, fontsize='x-large')
    plt.tight_layout()

    plt.savefig(
        os.path.join(figures_directory, f't_anom_indices_{timestamp}.png')
    )

    print(f'\nFigure saved to "{figures_directory}t_anom_indices_{timestamp}.png"')


    ##################### Linear Regression #####################

    # Convert to pandas Series with datetime index
    thi_series = pd.Series(fifty_meters, index=thi_time).sort_index()
    moci_series = pd.Series(norcal_moci.values, index=norcal_time).sort_index()

    # Ensure sorted time
    union_time = thi_series.index.union(moci_series.index)

    thi_interp = (
        thi_series
        .reindex(union_time)
        .interpolate(method='time')
    )

    th_on_moci = thi_interp.loc[moci_series.index]

    df = pd.concat([th_on_moci, moci_series], axis=1).dropna()
    df.columns = ['TH_50m_interp', 'MOCI']

    # Convert time to numeric for color mapping
    time_numeric = mdates.date2num(df.index)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=300)

    sc = ax.scatter(df['MOCI'], df['TH_50m_interp'], c=time_numeric, cmap='viridis', alpha=0.7)

    slope, intercept, r, p, stderr = linregress(df['MOCI'], df['TH_50m_interp'])
    r_2 = r**2
    x = np.linspace(df['MOCI'].min(), df['MOCI'].max(), 100)
    ax.plot(x, slope*x + intercept, 'k', lw=2,
            label=f'$R^2$ = {r_2:.2f}, p = {p:.3g}')

    ax.set_xlabel('California MOCI Index', fontsize='x-large')
    ax.set_ylabel('Trinidad Head Index (interpolated)', fontsize='x-large')
    ax.legend(frameon=False)

    ax.spines[:].set_linewidth(2)
    ax.tick_params(width=2, top=True, right=True, direction='in')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.title('Trinidad Head Index vs California MOCI', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_directory, f't_anom_vs_MOCI_regression_{timestamp}.png'))
    print(f'\nFigure saved to "{figures_directory}t_anom_vs_MOCI_regression_{timestamp}.png"')

    print('\nDone!')

if __name__ == "__main__":
    main()