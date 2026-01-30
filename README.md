# Timeseries

**make_timeseries_nc.py** -> Creates a single merged NetCDF file from all of the merged transect files listed in filepaths.
To run:
`uv run ./timeseries/make_timeseries_nc.py`

**plot_timeseries.py** -> Plots temperature and salinity anomaly timeseries figure and temperature anomaly index figures.
To run:
`uv run ./timeseries/plot_timeseries.py`

_Dependencies listed in 'timeseries/pyproject.toml'_

# Transects

**plot_anomaly_transect.py** -> Plots temperature and salinity anomaly for individual transect using merged NetCDF file.
To run:
`uv run ./transects/plot_anomaly_transect.py`

_Dependencies listed in 'transects/pyproject.toml'_
