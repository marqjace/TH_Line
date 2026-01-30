# Timeseries

**make_timeseries_nc.py** -> Creates a single merged NetCDF file from all of the merged transect files listed in filepaths.
To run:
`cd .\timeseries\`
`uv run .\make_timeseries_nc.py`

**plot_timeseries.py** -> Plots temperature and salinity anomaly timeseries figure and temperature anomaly index figures.
To run:
`cd .\timeseries\`
`uv run .\plot_timeseries.py`

_Dependencies listed in `.\timeseries\pyproject.toml`_

# Transects

**plot_anomaly_transect.py** -> Plots temperature and salinity anomaly for individual transect using merged NetCDF file.
To run:
`cd .\transects\`
`uv run .\plot_anomaly_transect.py`

_Dependencies listed in `.\transects\pyproject.toml`_
