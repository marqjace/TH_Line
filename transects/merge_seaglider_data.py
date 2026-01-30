import glidertools as gt

def main():

    filenames = r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect3/p686*.nc'
    output_filename = r'C:/Users/marqjace/data/seaglider/TH_line/deployments/nov_2025/transect3/1_26_merged.nc'

    # Load Variables
    gt.load.seaglider_show_variables(filenames)

    # Define Variables
    names = [
        'ctd_depth',
        'ctd_time',
        'ctd_pressure',
        'salinity',
        'temperature',
        # 'salinity_corrected',
        # 'aanderaa4831_dissolved_oxygen',
        # 'aanderaa4330_dissolved_oxygen',
        # 'sbe43_dissolved_oxygen'
    ]

    # Load Data into Dictionary
    ds_dict = gt.load.seaglider_basestation_netCDFs(
        filenames, names,
        return_merged=False,
        keep_global_attrs=False,
    )

    # Print Keys
    print(ds_dict.keys())

    # Rename Variables
    ctd_data_point = ds_dict['sg_data_point']

    if 'salinity_corrected' in ctd_data_point:
        dat = ctd_data_point.rename({
            'salinity': 'salt_raw',
            'temperature': 'temp_raw',
            'ctd_pressure': 'pressure',
            'ctd_depth': 'depth',
            'ctd_time': 'time_raw',
            # 'salinity_corrected': 'salt_corrected',
            # 'aanderaa4831_dissolved_oxygen': 'oxygen',
            # 'aanderaa4330_dissolved_oxygen': 'oxygen',
            # 'sbe43_dissolved_oxygen': 'oxygen'
        })
    else:
        dat = ctd_data_point.rename({
            'salinity': 'salt_raw',
            'temperature': 'temp_raw',
            'ctd_pressure': 'pressure',
            'ctd_depth': 'depth',
            'ctd_time': 'time_raw',
            # 'aanderaa4831_dissolved_oxygen': 'oxygen',
            # 'aanderaa4330_dissolved_oxygen': 'oxygen',
            # 'sbe43_dissolved_oxygen': 'oxygen'
        })

    print(dat)

    # Save Merged File to NetCDF
    dat.to_netcdf(output_filename)
    dat.close()

if __name__ == "__main__":
    main()
