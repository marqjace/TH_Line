import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import cmocean
import cartopy.mpl.gridliner as gridliner

# Bathymetry data
access_pr_file = 'C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Python/northern_california_36_msl_2005.nc'

ds = xr.open_dataset(access_pr_file)

fig = plt.figure(figsize=[16,9])
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=180))

# Change longitude 
longitude_west = -125.5
longitude_east = -124
latitude_south = 40.5
latitude_north = 41.5

# Clip the dataset to the specified extent using .sel()
ds = ds.sel(lon=slice(longitude_west, longitude_east), lat=slice(latitude_south, latitude_north))

# set the exent of the plot - this was used before I used the ds.sel method to clip the dataset
ax.set_extent([longitude_west, longitude_east, latitude_south, latitude_north], crs=ccrs.PlateCarree())

# Create an array for levels (bathymetry depth bins)
array = np.arange(0, -6000, -100)
contours = [-5900, -5000, -4000, -3000, -2000, -1000, -800, -600, -400, -300, -200, -100]

# Add contourf plot of Band1 aka depth, colorbar, label
contourf = ds.Band1.plot.contourf(ax=ax,
                    levels = array,
                    transform=ccrs.PlateCarree(),
                    cmap=cmocean.cm.deep_r,
                    add_colorbar=False)

# Add a colorbar to the plot for the contourf plot only
cbar = plt.colorbar(contourf, ax=ax, orientation='vertical', fraction=0.025, pad=0.07)
cbar.set_label('Depth (m)', fontsize=12, loc = 'center')


# Add contour lines on top of the filled contour plot
contour_lines = ds.Band1.plot.contour(ax=ax,
                                levels=array,  # or specify the levels as needed
                                colors='black',  # color of the contour lines
                                linewidths=0.3,# linewidth of the contour lines
                                linestyles='-',
                                transform=ccrs.PlateCarree())

# Label the line contours (not the filled bands) so each line gets its true value
ax.clabel(contour_lines, contours, inline=True, fontsize=10, colors='k')

# Add contour lines on top of the filled contour plot
contour = ds.Band1.plot.contour(ax=ax,
                                levels=[-1000],   # Only plot the contour for Band1 = 1000
                                colors=['red'],  # Change the color to red
                                linewidths=1,  # Adjust the linewidth as needed
                                linestyles='-',  # Use a dashed line style
                                transform=ccrs.PlateCarree())

# PAM Loiter Section
ax.plot([longitude_west, -124.78], 
        [41.05, 41.05], 
        color='yellow', 
        linestyle='--',
        linewidth = 3,
        transform=ccrs.PlateCarree())

ax.text(-125.2, 41.07, 'Loiter Section', fontsize=12, color='yellow', transform=ccrs.PlateCarree())

# PAM Profile Section
ax.plot([-124.78, -124.3417], 
        [41.05, 41.05], 
        color='purple', 
        linestyle='--',
        linewidth = 3,
        transform=ccrs.PlateCarree())

ax.text(-124.65, 41.07, 'Profile Section', fontsize=12, color='purple', transform=ccrs.PlateCarree())

# TH35
ax.scatter(-124.9167, 41.05, color='red', s=70, transform=ccrs.PlateCarree(), label='TH35')
ax.text(-124.9167, 41.01, 'TH35', fontsize=15, color='red', transform=ccrs.PlateCarree())

# TH15
ax.scatter(-124.4758, 41.05, color='red', s=70, transform=ccrs.PlateCarree(), label='TH15')
ax.text(-124.4758, 41.01, 'TH15', fontsize=15, color='red', transform=ccrs.PlateCarree())

# ST3
ax.scatter(-124.3417, 41.05, color='red', s=70, transform=ccrs.PlateCarree(), label='ST3')
ax.text(-124.3417, 41.01, 'ST3', fontsize=15, color='red', transform=ccrs.PlateCarree())

# BOEM Lease Areas
OCS_P_0561 = [
    (41.018708, -124.555662), (41.007903, -124.555408), (40.997097, -124.555154),
    (40.997288, -124.540891), (40.986482, -124.54064), (40.975676, -124.540388),
    (40.96487, -124.540137), (40.954064, -124.539886), (40.943258, -124.539635),
    (40.943067, -124.553886), (40.932261, -124.553632), (40.932068, -124.567881),
    (40.921262, -124.567625), (40.910456, -124.56737), (40.910261, -124.581614),
    (40.899456, -124.581356), (40.899259, -124.595598), (40.888453, -124.595338),
    (40.877648, -124.595079), (40.87745, -124.609315), (40.866644, -124.609053),
    (40.855838, -124.608792), (40.845032, -124.60853), (40.844833, -124.62276),
    (40.844632, -124.636989), (40.855437, -124.637255), (40.866243, -124.637521),
    (40.877049, -124.637788), (40.887855, -124.638054), (40.887651, -124.652293),
    (40.898457, -124.652562), (40.909263, -124.652831), (40.920068, -124.6531),
    (40.930874, -124.653369), (40.941679, -124.653639), (40.952485, -124.653908),
    (40.952279, -124.668161), (40.963085, -124.668433), (40.97389, -124.668705),
    (40.984695, -124.668977), (40.995501, -124.66925), (41.006306, -124.669523),
    (41.017111, -124.669796), (41.016903, -124.684062), (41.027709, -124.684337),
    (41.038514, -124.684612), (41.049319, -124.684888), (41.060124, -124.685164),
    (41.070929, -124.68544), (41.081734, -124.685716), (41.092539, -124.685992),
    (41.092747, -124.67171), (41.092954, -124.657427), (41.103759, -124.657699),
    (41.114564, -124.657971), (41.114769, -124.643684), (41.114972, -124.629396),
    (41.115173, -124.615108), (41.115372, -124.600821), (41.11557, -124.586533),
    (41.126376, -124.586793), (41.137181, -124.587054), (41.137377, -124.572761),
    (41.137572, -124.558468), (41.126766, -124.558212), (41.11596, -124.557957),
    (41.105155, -124.557701), (41.094349, -124.557446), (41.083543, -124.557191),
    (41.072737, -124.556935), (41.061932, -124.556681), (41.051126, -124.556426),
    (41.04032, -124.556171), (41.029514, -124.555917), (41.018708, -124.555662),
]

OCS_P_0562 = [
    (41.027709, -124.684337), (41.016903, -124.684062), (41.017111, -124.669796),
    (41.006306, -124.669523), (40.995501, -124.66925), (40.984695, -124.668977),
    (40.97389, -124.668705), (40.963085, -124.668433), (40.952279, -124.668161),
    (40.952485, -124.653908), (40.941679, -124.653639), (40.930874, -124.653369),
    (40.920068, -124.6531), (40.909263, -124.652831), (40.898457, -124.652562),
    (40.887651, -124.652293), (40.887855, -124.638054), (40.877049, -124.637788),
    (40.866243, -124.637521), (40.855437, -124.637255), (40.844632, -124.636989),
    (40.844833, -124.62276), (40.834027, -124.622496), (40.823221, -124.622233),
    (40.82302, -124.636458), (40.812214, -124.636192), (40.801408, -124.635927),
    (40.801206, -124.650147), (40.7904, -124.649879), (40.779594, -124.649611),
    (40.77939, -124.663827), (40.768584, -124.663557), (40.757779, -124.663288),
    (40.757573, -124.677498), (40.746767, -124.677226), (40.735962, -124.676955),
    (40.735754, -124.691161), (40.735545, -124.705366), (40.746351, -124.705642),
    (40.757156, -124.705919), (40.767962, -124.706195), (40.778767, -124.706471),
    (40.789573, -124.706748), (40.800378, -124.707025), (40.811184, -124.707302),
    (40.821989, -124.707579), (40.832795, -124.707856), (40.8436, -124.708134),
    (40.854405, -124.708411), (40.854194, -124.722642), (40.864999, -124.722922),
    (40.875804, -124.723202), (40.87559, -124.737438), (40.886396, -124.73772),
    (40.897201, -124.738003), (40.908006, -124.738286), (40.90779, -124.752528),
    (40.918595, -124.752813), (40.9294, -124.753099), (40.929183, -124.767346),
    (40.939987, -124.767634), (40.950792, -124.767922), (40.950573, -124.782173),
    (40.961378, -124.782464), (40.961156, -124.796717), (40.971961, -124.79701),
    (40.971738, -124.811265), (40.982542, -124.811561), (40.982765, -124.797303),
    (40.99357, -124.797597), (41.004375, -124.79789), (41.015179, -124.798184),
    (41.025984, -124.798478), (41.036788, -124.798772), (41.03701, -124.784503),
    (41.047815, -124.784795), (41.048035, -124.770523), (41.058839, -124.770813),
    (41.059058, -124.756538), (41.059275, -124.742264), (41.05949, -124.727989),
    (41.070294, -124.728272), (41.070508, -124.713995), (41.081313, -124.714275),
    (41.081524, -124.699996), (41.081734, -124.685716), (41.070929, -124.68544),
    (41.060124, -124.685164), (41.049319, -124.684888), (41.038514, -124.684612),
    (41.027709, -124.684337),
]

# Plot OCS-P 0561
lons_0561 = [coord[1] for coord in OCS_P_0561]
lats_0561 = [coord[0] for coord in OCS_P_0561]
ax.fill(lons_0561, lats_0561, color='white', alpha=0.2, transform=ccrs.PlateCarree())
ax.plot(lons_0561, lats_0561, color='white', linewidth=1.5, transform=ccrs.PlateCarree(), label='OCS-P 0561')

# Plot OCS-P 0562
lons_0562 = [coord[1] for coord in OCS_P_0562]
lats_0562 = [coord[0] for coord in OCS_P_0562]
ax.fill(lons_0562, lats_0562, color='white', alpha=0.2, transform=ccrs.PlateCarree())
ax.plot(lons_0562, lats_0562, color='white', linewidth=1.5, transform=ccrs.PlateCarree(), label='OCS-P 0562')

# Lease labels
ax.text(-124.6, 40.98, 'OCS-P 0561', fontsize=10, fontweight='bold', color='gray', transform=ccrs.PlateCarree(), ha='center')
ax.text(-124.74, 40.87, 'OCS-P 0562', fontsize=10, fontweight='bold', color='gray', transform=ccrs.PlateCarree(), ha='center')


# Add grid lines and labels
gl = ax.gridlines(draw_labels=False, 
                  linestyle='--', 
                  ylocs=np.arange(40.5, 42, 0.25), #lat_south, lat_north, spacing
                  linewidth = 0.7
                  ) 


# Add lat lon labels
gl.bottom_labels = True
gl.right_labels = True
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}


# Add border features
state_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(state_provinces, edgecolor = 'k')
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, facecolor='darkgray')

ax.set_title('Proposed PAM Sampling TH Line', fontsize=16, fontweight='bold', pad=20)

plt.savefig('C:/Users/marqjace/OneDrive - Oregon State University/Desktop/Repositories/TH_Line/mapping/pam_map.png', dpi=300, bbox_inches='tight')