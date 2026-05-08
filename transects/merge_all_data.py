import glidertools as gt

def main():

    filenames = r'C:/Users/marqjace/data/seaglider/sg266/2026_03_13_deployment/p686*.nc'
    output_filename = r'C:/Users/marqjace/data/seaglider/sg266/2026_03_13_deployment/pre_merged.nc'

    # Load Variables
    gt.load.seaglider_show_variables(filenames)

    # Define Variables
    names = [
        'salinity',
        'temperature',
        'ctd_pressure',
        'ctd_depth',
        'ctd_time',
        'wlbb2fl_sig695nm_adjusted',
        'wlbb2fl_sig700nm_adjusted',
        'wlbb2fl_sig460nm_adjusted',
        'aanderaa4831_dissolved_oxygen',
    ]

    # Load Data into Dictionary
    ds_dict = gt.load.seaglider_basestation_netCDFs(
        filenames, names,
        return_merged=True,
        keep_global_attrs=True,
    )

    merged = ds_dict['merged']

    dat = merged.rename({
        'salinity': 'salt_raw',
        'temperature': 'temp_raw',
        'ctd_pressure': 'pressure',
        'ctd_depth': 'depth',
        'ctd_time_dt64': 'time',
        'ctd_time': 'time_raw',
        'wlbb2fl_sig695nm_adjusted': 'fluorescence_raw',
        'wlbb2fl_sig700nm_adjusted': 'opbs_raw',
        'wlbb2fl_sig460nm_adjusted': 'cdom_raw',
        'aanderaa4831_dissolved_oxygen': 'oxygen_raw',
    })

    if 'time' in dat:
        dat = dat.drop(["wlbb2fl_results_time", "aa4831_time", "aa4831_time_dt64"])

    print(dat)

    # Save Merged File to NetCDF
    dat.to_netcdf(output_filename)
    dat.close()

if __name__ == "__main__":
    main()
