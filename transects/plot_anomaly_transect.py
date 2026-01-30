# anomaly_transect.py
# Created by: Jace Marquardt
# Date Created: January 29, 2026

import os
import numpy as np
import xarray as xr
import cmocean
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm
from scipy.interpolate import griddata


def main():
    """
    This script generates temperature and salinity anomaly transect plots for the TH Line
    using WOA climatology data and Seaglider deployment data.
    """

    ################### WOA #####################

    # Open WOA Climatology Dataset and Select the TH Line transect
    ds1 = xr.open_dataset(r'C:\Users\marqjace\data\seaglider\TH_line\woa_climatology_data\temperature\woa18_decav_t01_04.nc', decode_times=False)
    ds2 = xr.open_dataset(r'C:\Users\marqjace\data\seaglider\TH_line\woa_climatology_data\salinity\woa18_decav_s01_04.nc', decode_times=False)
    # ds = xr.open_dataset(r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\woa18_decav_t01_01.nc', decode_times=False)
    # ^ Example using the January decadal average dataset 'woa18_decav_t01_01.nc'

    # Select the extent of the TH Line 200km inshore (-126.5 to -124.5W longitude, 0-1000m depth)
    ds1 = ds1.sel(lat=41.625, lon=slice(-129.375, -124.375), depth=slice(0,1000))
    ds2 = ds2.sel(lat=41.625, lon=slice(-129.375, -124.375), depth=slice(0,1000))

    # Name the variables
    t_an1 = ds1['t_an'][0,:,:]  # Temperature
    lon1 = ds1['lon']  # Longitude
    lat1 = ds1['lat']  # Latitude
    depth1 = ds1['depth']  # Depth

    s_an2 = ds2['s_an'][0,:,:]  # Salinity
    lon2 = ds2['lon']  # Longitude
    lat2 = ds2['lat']  # Latitude
    depth2 = ds2['depth']  # Depth

    # Define a new grid (112 points longitude, 200 points depth)
    z_new1 = np.arange(0,1000, 5)
    lon_new1 = np.arange(-129.375, -124.375, 0.0625)

    # Linearly interpolate the dataset "ds" across the new defined grid
    ds_z_new1 = ds1.interp(depth=z_new1, lon=lon_new1)
    ds_z_new2 = ds2.interp(depth=z_new1, lon=lon_new1)

    # Meshgrid the longitude and depth grid
    Xgrid1, Ygrid1 = np.meshgrid(ds_z_new1['lon'], ds_z_new1['depth'])
    Xgrid2, Ygrid2 = np.meshgrid(ds_z_new2['lon'], ds_z_new2['depth'])

    # Define new temperature variable dimensions. We only want longitude and depth
    ds_z_t_an1 = ds_z_new1['t_an'][0,:,:]
    ds_z_s_an2 = ds_z_new2['s_an'][0,:,:]

    # print(f'Original Grid Shape VS New Grid Shape:\nOriginal temperature value shape: {t_an1.shape}\nGridded temperature value shape: {ds_z_t_an1.shape}')

    # plt.contourf(Xgrid1, Ygrid1, ds_z_t_an1, cmap=cmocean.cm.thermal)
    # plt.contourf(Xgrid2, Ygrid2, ds_z_s_an2, cmap=cmocean.cm.haline)
    # plt.gca().invert_yaxis()
    # plt.show()


    ################### Glider Data #####################

    # Set up new grid (36 points is 2.25 deg longitude for every 5 km, 200 points depth to 1000m is every 5 meters)
    xn, yn = 80, 200
    xmin, xmax = -129.375, -124.375
    ymin, ymax = 0, 1000

    # Generate a regular grid to interpolate the data
    xgrid = np.linspace(xmin, xmax, xn)
    ygrid = np.linspace(ymin, ymax, yn)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)

    dat = xr.open_dataset(r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect3/1_26_merged.nc', decode_times=False)

    mask = ~np.isnan(dat.temp_raw) & ~np.isnan(dat.salt_raw)
    dat = dat.where(mask, drop=True)

    # variable assignment for conveniant access
    depth = dat.depth
    dives = dat.dives
    latitude = dat.latitude
    longitude = dat.longitude
    pres = dat.pressure
    temp = dat.temp_raw
    salt = dat.salt_raw
    # oxy = dat.oxygen

    # Interpolate using "linear" method

    temp_1_24 = griddata(points = (longitude, depth),
                values = temp,
                xi = (Xgrid, Ygrid),
                method = 'linear')

    salt_1_24 = griddata(points = (longitude, depth),
                values = salt,
                xi = (Xgrid, Ygrid),
                method = 'linear')

    # set the time coverage start and end
    time_start = dat.attrs['time_coverage_start']
    time_end = dat.attrs['time_coverage_end']

    # Create Temperature Anomaly Array
    t_anom = np.subtract(temp_1_24, ds_z_t_an1)
    s_anom = np.subtract(salt_1_24, ds_z_s_an2)

    # Set Colorbar and Contour Line Ranges
    boundaries_temp = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    levels_temp = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    boundaries_salt = [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]
    levels_salt = [-1, -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1]
    boundaries_oxy = [0, 50, 100, 150, 200, 250, 300]

    divnorm_temp=colors.TwoSlopeNorm(vcenter=0., vmin=-4, vmax=4)
    divnorm_salt=colors.TwoSlopeNorm(vcenter=0., vmin=-.75, vmax=.75)

    anomaly = 'anomaly'

    x = np.array((-124.7, -124.68, -124.58, -124.48, -124.43, -124.38, -124.36, -124.34, -124.31, -124.27, -124.25, -124.20, -124.15, -124.08, -124.07))
    y = np.array((1000, 970, 900, 800, 700, 600, 500, 400, 300, 200, 150, 100, 60, 30, 0))

    figures_dir = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\transects\figures'
    if os.path.exists(figures_dir) == False:
        os.makedirs(figures_dir)


    ################### T-Anom Plot #####################

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), dpi=300)

    contour1 = ax1.contourf(Xgrid, Ygrid, temp_1_24, cmap='jet')
    bottom_topo1 = ax1.plot(x, y, c='gray')
    # border = ax1.plot(borderx, bordery, c='gray')
    # ax1.fill_between(bottom_topo1, border, c='gray')
    yfill = norm.pdf(x, loc=y)
    ax1.fill_between(x, yfill, color='gray')
    ax1.invert_yaxis()
    ax1.set_title(f'TH Line - SG686 - {time_start} - {time_end}', fontsize='large', fontweight='semibold', pad=10)
    # ax1.set_title(f'TH Line - SG266 - 13 August 2024 - 25 August 2024', fontsize='large', fontweight='semibold', pad=10)
    ax1.set_ylabel('Depth (m)')
    ax1.set_xticks((-124, -124.5, -125, -125.5, -126, -126.5, -127, -127.5, -128, -128.5, -129))
    ax1.set_xticklabels(())
    ax1.set_xlim(-129, -124)
    ax1.spines[:].set_linewidth(2)
    ax1.tick_params(width=2, top=False, right=True, direction='in', which='both')
    plt.colorbar(contour1, label=r'T ($\degree$C)')

    twin1 = ax2.twiny()
    twin2 = ax2.twiny()

    contour2 = twin1.contourf(Xgrid, Ygrid, t_anom, cmap='RdYlBu_r', norm=divnorm_temp, levels=boundaries_temp)
    plt.colorbar(contour2, label=r'T$_{anomaly}$ ($\degree$C)')
    bottom_topo2 = twin1.plot(x, y, c='gray')

    ax2.invert_yaxis()
    twin1.spines.bottom.set_position(('data', 1000))
    ax2.set_ylabel('Depth (m)')
    twin1.set_xticks((-124, -124.5, -125, -125.5, -126, -126.5, -127, -127.5, -128, -128.5, -129))
    twin1.set_xticklabels((r'', '', '125$\degree$W', '', '126$\degree$W', '', '127$\degree$W', '', '128$\degree$W', '', ''))
    twin1.set_xlim(-129, -124)
    twin1.spines[:].set_linewidth(2)
    twin1.tick_params(width=2, top=False, right=True, direction='in', which='both')
    twin1.xaxis.set_ticks_position('bottom')
    ax2.set_xticks([])
    ax2.set_xticklabels([])

    twin2.spines.top.set_position(("outward", -190))
    twin2.set_xticks((-124.07, -125.4, -126.6, -127.8, -129))
    twin2.set_xticklabels(('0', '100', '200', '300', '400'))
    twin2.set_xlim(-129, -124)
    twin2.spines[:].set_linewidth(2)
    twin2.set_xlabel('Distance (km)', labelpad=-35)

    plt.savefig(os.path.join(figures_dir, f'TH_line_SG686_T_anom.png'), bbox_inches='tight')


    ################### S-Anom Plot #####################

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), dpi=300)

    contour1 = ax1.contourf(Xgrid, Ygrid, salt_1_24, cmap=cmocean.cm.haline)
    bottom_topo1 = ax1.plot(x, y, c='gray')
    # border = ax1.plot(borderx, bordery, c='gray')
    # ax1.fill_between(bottom_topo1, border, c='gray')
    yfill = norm.pdf(x, loc=y)
    ax1.fill_between(x, yfill, color='gray')
    ax1.invert_yaxis()
    ax1.set_title(f'TH Line - SG686 - {time_start} - {time_end}', fontsize='large', fontweight='semibold', pad=10)
    # ax1.set_title(f'TH Line - SG686 - 13 August 2024 - 25 August 2024', fontsize='large', fontweight='semibold', pad=10)
    ax1.set_ylabel('Depth (m)')
    ax1.set_xticks((-124, -124.5, -125, -125.5, -126, -126.5, -127, -127.5, -128, -128.5, -129))
    ax1.set_xticklabels(())
    ax1.set_xlim(-129, -124)
    ax1.spines[:].set_linewidth(2)
    ax1.tick_params(width=2, top=False, right=True, direction='in', which='both')
    plt.colorbar(contour1, label=r'S (PSU)')

    twin1 = ax2.twiny()
    twin2 = ax2.twiny()

    contour2 = twin1.contourf(Xgrid, Ygrid, s_anom, cmap='BrBG_r', norm=divnorm_salt, levels=boundaries_salt)
    plt.colorbar(contour2, label=r'S$_{anomaly}$ (PSU)')
    bottom_topo2 = twin1.plot(x, y, c='gray')

    ax2.invert_yaxis()
    twin1.spines.bottom.set_position(('data', 1000))
    ax2.set_ylabel('Depth (m)')
    twin1.set_xticks((-124, -124.5, -125, -125.5, -126, -126.5, -127, -127.5, -128, -128.5, -129))
    twin1.set_xticklabels((r'', '', '125$\degree$W', '', '126$\degree$W', '', '127$\degree$W', '', '128$\degree$W', '', ''))
    twin1.set_xlim(-129, -124)
    twin1.spines[:].set_linewidth(2)
    twin1.tick_params(width=2, top=False, right=True, direction='in', which='both')
    twin1.xaxis.set_ticks_position('bottom')
    ax2.set_xticks([])
    ax2.set_xticklabels([])

    twin2.spines.top.set_position(("outward", -190))
    twin2.set_xticks((-124.07, -125.4, -126.6, -127.8, -129))
    twin2.set_xticklabels(('0', '100', '200', '300', '400'))
    twin2.set_xlim(-129, -124)
    twin2.spines[:].set_linewidth(2)
    twin2.set_xlabel('Distance (km)', labelpad=-35)

    plt.savefig(os.path.join(figures_dir, f'TH_line_SG686_S_anom.png'), bbox_inches='tight')


if __name__ == "__main__":
    main()
