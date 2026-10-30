# Trinidad Head time-series workflow

This directory contains a two-stage workflow for turning repeated glider
transects off Trinidad Head, California, into gridded temperature and salinity
anomalies, then plotting those anomalies and comparing derived indices with
other ocean-climate indices.

## Workflow at a glance

1. [`make_timeseries_nc.py`](./make_timeseries_nc.py) reads the configured
   merged transect files and World Ocean Atlas (WOA) monthly climatologies.
   It averages each transect into depth profiles, calculates climatology
   anomalies, interpolates those profiles onto a common time-depth grid, and
   writes a NetCDF dataset.
2. [`plot_timeseries.py`](./plot_timeseries.py) reads an anomaly NetCDF file,
   smooths and plots its temperature and salinity anomalies, and creates
   temperature-index comparisons.

The plotting script is a separate step; it does not call the NetCDF builder.
The configured output and input filenames currently differ (see
**Important path and configuration notes**).

## Building the anomaly dataset

### Source data and configuration

The `filepaths` list in `make_timeseries_nc.py` is the explicit set of merged
transect files to process. It is not discovered automatically. The listed files
are expected to contain `time_raw`, `longitude`, `depth`, `temp_raw`, and
`salt_raw`; when `salt_corrected` is present, that variable is used in place of
`salt_raw` for the salinity profile. Each file is opened while
[`utils/transects_func.py`](./utils/transects_func.py) interpolates its data.

Temperature and salinity reference fields come from the monthly WOA NetCDF
files configured in [`utils/woa_temp.py`](./utils/woa_temp.py) and
[`utils/woa_salt.py`](./utils/woa_salt.py). Those modules load all twelve
monthly files when imported. They select a latitude of 41.625 degrees north
and the configured longitude and depth ranges, then interpolate the reference
fields onto 5 m depth and 0.0625-degree longitude intervals.

### Processing steps

1. **Interpolate each transect.** `transects_func.process_transects` creates a
   longitude-depth mesh and uses linear `scipy.interpolate.griddata`
   interpolation for each transect's raw temperature and salinity. It takes a
   NaN-aware mean across longitude at each depth to produce a pair of depth
   profiles. The transect's timestamp is the mean of its `time_raw` values.
2. **Calculate monthly anomalies.** The anomaly helpers in
   [`utils/anomaly.py`](./utils/anomaly.py) select a WOA month from the
   transect's generated name (the month is its first underscore-delimited
   component, such as `12` in `12_14_a`). At each depth and longitude, the
   transect profile is differenced from that month's WOA field.
3. **Average across longitude and grid in time.** `compute_mean_depth_profile`
   reduces each anomaly field to one mean anomaly per depth. The builder then
   linearly interpolates all transect profiles onto a regular time-depth grid:
   nominally every 30 days and every 5 m from 0 to 995 m. The last observed
   transect time is included even if it does not fall on a 30-day interval.
   Two nominal surface rows at -10 m and -5 m are added from the surface
   interpolation; the 0 m row is set from the next (5 m) interpolated row.
4. **Write NetCDF.** The dataset contains `temperature_anomaly` (degrees C)
   and `salinity_anomaly` (PSU), each with dimensions `depth` and `time`.
   Global metadata records the creation time, source, number of transects, and
   configured transect file paths.

The interpolation is linear, so points outside the data-supported area may
remain missing. The builder suppresses `RuntimeWarning`s during its processing.

### Reprocessing behavior

The output dataset's `processed_filepaths` metadata is compared with the
configured list. If the list is unchanged and an output with that metadata
exists, the script exits without rebuilding. If any path is new—or if
`--force` is supplied—it processes the complete configured transect list and
rewrites the dataset; it does not append only the newly listed transects.

## Plotting and index comparisons

`plot_timeseries.py` sorts the NetCDF time coordinate and applies rolling
averages over three time samples and four depth samples (`min_periods=1`) to
both anomaly variables. It then:

- saves a two-panel temperature/salinity anomaly time-depth plot, with
  contour lines and manually specified deployment intervals;
- derives Trinidad Head temperature-index series from the smoothed 50 m
  temperature and the mean over the upper 100 m;
- plots those series alongside configured comparison data, including
  hard-coded Newport/NH Line values and a separate OOI Newport index file;
- reads SoCal temperature and Oceanic Niño indices from a configured NetCDF
  file, and North California MOCI values from a configured CSV; and
- saves a 50 m versus upper-100 m scatter plot with a fitted linear regression
  and R-squared value.

The MOCI comparison figure uses a second vertical axis for MOCI. Other index
plots use temperature anomaly axes. Figure files are written to the configured
`figures` directory with a timestamp in each filename. In total, the script
currently saves five PNG figures per successful run.

## Function-by-function walkthrough

### `utils/transects_func.py`

- **`make_transect_grid(xmin, xmax, ymin, ymax, xn, yn)`** builds a
  two-dimensional longitude/depth mesh using evenly spaced NumPy coordinates.
  Its defaults make 36 longitude points from -126.55 to -124.375 degrees and
  200 depth points from 0 to 1000 m. It returns `(Xgrid, Ygrid)`, each shaped
  depth-by-longitude.
- **`interp_to_grid(lon, depth, values, Xgrid, Ygrid)`** flattens the observed
  longitude, depth, and measured values, then uses linear `griddata`
  interpolation to estimate `values` at every grid cell. Locations outside
  the convex hull of the observations are typically NaN.
- **`transect(filepath, Xgrid, Ygrid)`** opens one merged transect NetCDF file
  (dropping the named `compass_timeouts_times_truck` variable), computes the
  mean of its `time_raw` timestamps, and interpolates raw temperature and
  salinity to the supplied mesh. It averages each interpolated field over
  longitude at each depth using `nanmean`. If `salt_corrected` exists, it
  interpolates and averages that in place of raw salinity. It returns a
  dictionary containing longitude, temperature and salinity profiles, and
  mean timestamp. The opened dataset is not explicitly closed in this
  function.
- **`process_transects(filepaths)`** creates one default mesh, processes the
  paths in list order, and names each result from the file's basename before
  `_merged` (for example `10_25_b`). It returns three dictionaries:
  `results` contains each complete per-file result, while `temps` and `salts`
  reshape the profiles into the input structure expected by the anomaly
  functions, with depth and mean time alongside each profile.

### `utils/anomaly.py`

- **`temperature_anomaly(transect_dict, woa_dict)`** converts climatology keys
  to strings, obtains the month from the first underscore-separated part of
  each transect name, and subtracts that month's WOA temperature field from
  the transect temperature profile. `[:, np.newaxis]` makes the profile a
  depth-by-one array so it broadcasts across climatology longitudes. It
  returns per-transect anomaly arrays and their mean timestamps. A transect
  whose name does not identify a key in `woa_dict` is omitted.
- **`salinity_anomaly(transect_dict, woa_dict)`** performs the same operation
  using the `salt` profile and monthly WOA salinity field; its output key is
  `salt_anomaly`.

### `utils/woa_temp.py` and `utils/woa_salt.py`

- **`woa_temp(filepath, month_idx, lat, lon_range, depth_range)`** opens one
  WOA temperature NetCDF file with time decoding disabled, selects the
  requested latitude/longitude/depth area, and interpolates `t_an` onto the
  fixed 5 m and 0.0625-degree grids. It returns the selected source arrays,
  the interpolated temperature field, and longitude/depth meshgrids.
- **`woa_salt(filepath, month_idx, lat, lon_range, depth_range)`** does the
  corresponding work for `s_an` and returns it as `ds_z_s_an`.
- In each module, after the function definition, module-level code calls the
  function for all twelve configured monthly files and assigns the resulting
  interpolated fields to names such as `woa_temp_jan` and `woa_salt_jan`.
  Therefore importing either module reads every configured file immediately;
  `month_idx` is accepted by both functions but is not used in their current
  calculations. The `lon_new` coordinate array is used for interpolation.

### `make_timeseries_nc.py`

- **`compute_mean_depth_profile(anomaly_dict)`** loops over anomaly records,
  finds the dictionary entry whose name ends with `_anomaly`, and calculates
  its NaN-aware mean across longitude (axis 1). It returns one anomaly value
  per depth per transect, retaining each transect's mean time.
- **`create_interpolated_grid(profile_dict)`** creates a 200-point input depth
  axis from 0 to 1000 m and a 30-day output time axis spanning the transect
  timestamps. It adds the exact last transect time if it is not already on
  that axis, then makes a 5 m output depth axis from 0 to 995 m. It expands
  all profiles into point arrays and uses linear `griddata` in time-depth
  space. Finally it prepends rows labeled -10 m and -5 m using the
  interpolated surface row and sets the 0 m row equal to the 5 m row. It
  returns the gridded values as a depth-indexed DataFrame plus the time grid.
- **`process_anomaly_data(transects, woa_months, anomaly_func)`** is the
  reusable three-step wrapper: call the given anomaly function, average each
  anomaly transect into a depth profile, then interpolate those profiles onto
  the common time-depth grid. It returns `(grid, time_grid)`.
- **`get_processed_filepaths(output_file)`** returns an empty set if there is
  no output file. Otherwise, it opens the NetCDF file and reads the
  `processed_filepaths` global attribute, splitting it on `||`. If the
  attribute is absent it returns an empty set. It prints a warning and returns
  an empty set if opening or reading the file fails.
- **`find_new_filepaths(all_filepaths, processed_filepaths)`** filters the
  configured list, returning paths not present in the stored processed-path
  set. It compares path strings exactly; it does not inspect file contents or
  modification times.
- **`main(force_rebuild=False)`** coordinates dataset creation. It ensures
  the output directory exists, compares configured and stored paths, and
  returns early when the existing output appears current unless forced. For
  any rebuild it processes the full transect list, maps the imported monthly
  WOA fields by month, builds temperature and salinity anomaly grids, packages
  them and their coordinates/metadata into an xarray `Dataset`, and writes
  `timeseries_anomaly.nc`. The command-line block parses `--force` and passes
  it to this function.

### `plot_timeseries.py`

- **`decimal_year_to_timestamp(decimal_year)`**, defined inside `main`,
  separates the integer year from its fractional part and converts that
  fraction into elapsed days from January 1. It uses 366 days for leap years
  and 365 otherwise. This converts the hard-coded decimal-year Newport
  observations into pandas timestamps.
- **`main()`** performs the plotting workflow. It opens and time-sorts the
  configured anomaly dataset, computes rolling-smoothed temperature and
  salinity fields, then extracts the smoothed 50 m series and upper-100 m
  temperature mean. It converts the hard-coded Newport decimal-year data,
  reads a separate Newport OOI index, and prepares the comparison index
  datasets. It saves (1) the temperature/salinity depth-time panels, (2) a
  temperature-index plot with MOCI on a second y-axis, (3) an index plot
  without MOCI, (4) a comparison of the 50 m and upper-100 m Trinidad Head
  indices over time, and (5) a scatter/regression comparison of the two
  Trinidad Head series. The regression uses `np.polyfit` for its line and
  computes R-squared from residual and total sums of squares. The module's
  `if __name__ == "__main__"` guard calls `main()` only when the script is run
  directly.

## Running

From this directory, after configuring the input paths and installing the
Python packages used by the scripts (NumPy, pandas, xarray, SciPy, Matplotlib,
Rich, and cmocean), run:

```powershell
python make_timeseries_nc.py
python plot_timeseries.py
```

To force a complete rebuild even when no new transect paths are detected:

```powershell
python make_timeseries_nc.py --force
```

The scripts require access to their input NetCDF and CSV files and an xarray
NetCDF backend. They create their configured output directories when needed.

## Important path and configuration notes

- The builder writes `timeseries/data/timeseries_anomaly.nc`. The plotter
  currently opens the hard-coded file
  `timeseries/data/timeseries_anomaly_200km_Oct26.nc`. These names are not
  automatically connected: update the plotter's input path or otherwise
  provide the intended dataset at that location before plotting.
- Transect file paths, WOA climatology paths, index-data paths, and the figure
  directory are hard-coded Windows paths in the Python files. Update them for
  the machine and dataset being used.
- The plot title describes an average over 100 km, while the transect grid
  defaults and plotter input filename indicate a 200 km product. The plotted
  spatial extent is determined by the selected input data and configured
  longitude grids; review those settings and the title together when changing
  the product.
- The deployment bars are hand-entered date ranges in `plot_timeseries.py`;
  they do not come from the transect input list.
- The hard-coded index observations and external comparison files are
  maintained separately from the anomaly NetCDF, so updating transects does
  not update those comparison series.
