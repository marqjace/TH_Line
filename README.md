# Timeseries

## make_timeseries_nc.py

Creates a single merged NetCDF file from all of the merged transect files listed in filepaths.</br>
To run:</br>
`cd .\timeseries\`</br>
`uv run .\make_timeseries_nc.py`

## plot_timeseries.py

Plots temperature and salinity anomaly timeseries figure and temperature anomaly index figures.</br>
To run:</br>
`cd .\timeseries\`</br>
`uv run .\plot_timeseries.py`

_Dependencies listed in .\timeseries\pyproject.toml_

# Transects

## plot_anomaly_transect.py

Plots temperature and salinity anomaly for individual transect using merged NetCDF file.</br>
To run:</br>
`cd .\transects\`</br>
`uv run .\plot_anomaly_transect.py`

## merge_seaglider_data.py

Merges per dive NetCDF files into a single NetCDF.</br>
To run:</br>
`cd .\transects\`</br>
`uv run .\merge_seaglider_data.py`

_Dependencies listed in .\transects\pyproject.toml_
