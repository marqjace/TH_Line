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

def process_transects(
    filepaths,
    existing_times=None,
    return_skipped=False,
):
    """
    Process transects and create dictionaries of temperature and salinity.

    Parameters
    ----------
    filepaths : list of str
        Filepaths to merged transect netCDF files
    existing_times : set of np.datetime64 or pandas.Timestamp, optional
        Transect mean_times that have already been processed
    return_skipped : bool, optional
        If True, also return list of skipped transects

    Returns
    -------
    results : dict
        Transect metadata keyed by transect name
    temps : dict
        Temperature profiles keyed by transect name
    salts : dict
        Salinity profiles keyed by transect name
    skipped : list (optional)
        List of skipped transect names
    """

    if existing_times is None:
        existing_times = set()

    # Normalize times once for safe comparison
    existing_times = {np.datetime64(t) for t in existing_times}

    Xgrid, Ygrid = make_transect_grid()

    results = {}
    skipped = []

    for i, fp in enumerate(filepaths, start=1):
        base = os.path.basename(fp)
        name = base.split("_merged")[0]

        print(f"Processing {i}/{len(filepaths)} {name}...")

        out = transect(fp, Xgrid, Ygrid)

        transect_time = np.datetime64(out["mean_time"])

        # -------- Incremental skip ----------
        if transect_time in existing_times:
            skipped.append(name)
            continue
        # ------------------------------------

        results[name] = out

    temps = {
        k: {
            "temp": v["temp_profile"],
            "depth": Ygrid[:, 0],
            "mean_time": v["mean_time"],
        }
        for k, v in results.items()
    }

    salts = {
        k: {
            "salt": v["salt_profile"],
            "depth": Ygrid[:, 0],
            "mean_time": v["mean_time"],
        }
        for k, v in results.items()
    }

    print("Processing complete.\n")

    if return_skipped:
        return results, temps, salts, skipped

    return results, temps, salts

