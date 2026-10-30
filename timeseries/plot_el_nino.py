import os
import numpy as np
import pandas as pd
import xarray as xr
from rich import print
from datetime import datetime
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


# ---------------------------------------------------------------------------
# Helper: select a two-year window from a dataset and re-index the time axis
# so that all panels share the same "day-of-event" x-axis (Month 1 … Month 24).
# ---------------------------------------------------------------------------

def _slice_event(ds, tanom_smoothed, sanom_smoothed, start_year, end_year):
    """Return (time_offset_days, depth, tanom, sanom) for start_year-01-01 → end_year-12-31."""
    t0 = np.datetime64(f'{start_year}-01-01')
    t1 = np.datetime64(f'{end_year}-12-31')
    mask = (ds['time'].values >= t0) & (ds['time'].values <= t1)
    ta = tanom_smoothed.isel(time=mask)
    sa = sanom_smoothed.isel(time=mask)
    # Days since Jan 1 of the first year so x-axes are directly comparable
    time_days = (ta.time.values - t0).astype('timedelta64[D]').astype(float)
    return time_days, ds['depth'].values, ta.values, sa.values


def _month_ticks(start_year):
    """Return (tick_positions_days, tick_labels) for Jan/Apr/Jul/Oct of both years."""
    months = [
        (1, 'Jan'), (4, 'Apr'), (7, 'Jul'), (10, 'Oct'),
    ]
    positions, labels = [], []
    for yr_offset in range(2):
        for month, name in months:
            t = pd.Timestamp(start_year + yr_offset, month, 1)
            t0 = pd.Timestamp(start_year, 1, 1)
            positions.append((t - t0).days)
            labels.append(f"{name}\n'{str(start_year + yr_offset)[2:]}")
    return positions, labels


def _deployment_segments_in_window(deployment_pairs, start_year, end_year):
    """Filter deployment hlines to those overlapping the two-year window."""
    t0 = pd.Timestamp(start_year, 1, 1)
    t1 = pd.Timestamp(end_year, 12, 31)
    segs = []
    for xmin, xmax in deployment_pairs:
        xmin_ts = pd.Timestamp(xmin)
        xmax_ts = pd.Timestamp(xmax)
        # clip to window
        seg_start = max(xmin_ts, t0)
        seg_end   = min(xmax_ts, t1)
        if seg_start < seg_end:
            segs.append(
                ((seg_start - t0).days, (seg_end - t0).days)
            )
    return segs


# ---------------------------------------------------------------------------
# All glider deployment intervals (from the original script)
# ---------------------------------------------------------------------------
DEPLOYMENT_PAIRS = [
    (datetime(2014,12,4),  datetime(2015,3,9)),
    (datetime(2015,3,9),   datetime(2015,9,17)),
    (datetime(2015,9,17),  datetime(2016,5,16)),
    (datetime(2016,5,23),  datetime(2016,10,21)),
    (datetime(2016,10,21), datetime(2017,6,5)),
    (datetime(2017,6,5),   datetime(2017,11,6)),
    (datetime(2018,4,17),  datetime(2018,10,2)),
    (datetime(2018,11,7),  datetime(2019,4,9)),
    (datetime(2019,4,9),   datetime(2019,8,19)),
    (datetime(2019,9,16),  datetime(2020,3,19)),
    (datetime(2020,9,16),  datetime(2021,2,6)),
    (datetime(2021,11,12), datetime(2022,6,16)),
    (datetime(2022,7,29),  datetime(2023,1,26)),
    (datetime(2023,1,26),  datetime(2023,8,14)),
    (datetime(2023,10,13), datetime(2024,4,12)),
    (datetime(2024,4,12),  datetime(2024,8,9)),
    (datetime(2024,10,21), datetime(2024,12,4)),
    (datetime(2025,3,21),  datetime(2025,11,11)),
    (datetime(2025,11,11), datetime(2026,2,25)),
    (datetime(2026,3,14),  datetime(2026,8,27)),
    (datetime(2026,9,18),  datetime.now()),
]


# ---------------------------------------------------------------------------
# Core stacked-panel plotting function
# ---------------------------------------------------------------------------

def plot_stacked_enso(
    ds,
    tanom_smoothed,
    sanom_smoothed,
    event_windows,          # list of (start_year, end_year, label_str)
    variable,               # 'temperature' or 'salinity'
    figures_directory,
    timestamp,
    latest_transect_time,
):
    """
    Create a figure with one row per ENSO event, all sharing the same
    2-year x-axis (days since Jan 1 of the event's start year).
    variable: 'temperature' → temperature anomaly panels
              'salinity'    → salinity anomaly panels
    """

    n_events = len(event_windows)

    if variable == 'temperature':
        cmap      = 'bwr'
        divnorm   = colors.TwoSlopeNorm(vcenter=0., vmin=-4, vmax=4)
        levels    = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
        cbar_label = r'($\degree$C)'
        title_var  = 'Temperature'
        fname_tag  = 't_anom'
    else:
        cmap      = 'BrBG_r'
        divnorm   = colors.TwoSlopeNorm(vcenter=0., vmin=-0.75, vmax=0.75)
        levels    = [-.6, -.4, -.2, .2, .4, .6]
        cbar_label = r'(PSU)'
        title_var  = 'Salinity'
        fname_tag  = 's_anom'

    fig, axes = plt.subplots(
        n_events, 1,
        figsize=(14, 4.5 * n_events),
        dpi=300,
    )
    fig.subplots_adjust(
        left=0.07,    # left border
        right=1.05,   # right border (leaves room for colorbars)
        top=0.90,     # top border (leaves room for suptitle)
        bottom=0.07,  # bottom border
        hspace=0.35,  # vertical gap between panels
    )
    if n_events == 1:
        axes = [axes]

    x_max_days = 365 * 2   # two full years

    for ax, (start_year, end_year, label) in zip(axes, event_windows):
        time_days, depth, ta, sa = _slice_event(
            ds, tanom_smoothed, sanom_smoothed, start_year, end_year
        )
        data = ta if variable == 'temperature' else sa

        if time_days.size == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_ylabel(label, fontsize=12, labelpad=8)
            continue

        pcm = ax.pcolormesh(
            time_days, depth, data,
            cmap=cmap, norm=divnorm, shading='nearest'
        )
        cnt = ax.contour(
            time_days, depth, data,
            colors='black', norm=divnorm, levels=levels, alpha=0.65, linewidths=0.8
        )
        ax.clabel(cnt, cnt.levels, inline=True, fontsize=8)

        # Glider deployment lines at 570 m
        segs = _deployment_segments_in_window(DEPLOYMENT_PAIRS, start_year, end_year)
        for seg_start, seg_end in segs:
            ax.hlines(y=570, xmin=seg_start, xmax=seg_end, color='k', linewidth=1.5)

        # Y axis
        ax.invert_yaxis()
        ax.set_yticks([0, 200, 400, 600])
        ax.set_ylim(600, 0)
        ax.set_ylabel('Depth (m)', fontsize=11)

        # X axis — shared month ticks
        tick_pos, tick_labs = _month_ticks(start_year)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, fontsize=9)
        ax.set_xlim(0, x_max_days)

        # Label: event name as y-axis title annotation
        ax.set_title(label, loc='left', fontsize=12, fontweight='bold', pad=4)

        ax.spines[:].set_linewidth(1.5)
        ax.tick_params(width=2, top=True, right=True, direction='in', labelsize=10, pad=8)

        # Colorbar per row
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.80)
        cbar.outline.set_linewidth(1.5)
        cbar.set_label(cbar_label, rotation=0, labelpad=10, fontsize=10)

    # Determine ENSO type label for the supertitle
    # (inferred from the first event window label)
    enso_type = event_windows[0][2].split()[0]  # 'El' or 'La'
    if 'El' in enso_type:
        enso_name = 'El Niño'
        file_tag  = 'el_nino'
    else:
        enso_name = 'La Niña'
        file_tag  = 'la_nina'

    fig.suptitle(
        f'{title_var} Anomaly during {enso_name} Events — Trinidad Head, CA\n'
        f'Averaged over 200 km from the coast',
        fontsize='x-large',
    )
    fig.text(
        0.78, 0.01,
        f'Latest transect date: {latest_transect_time.strftime("%Y-%m-%d")}',
        fontsize='small',
    )
    fig.text(
        0.02, 0.01,
        'Courtesy of the Oregon State University Glider Research Group, NANOOS, CeNCOOS',
        fontsize='small',
    )

    fname = os.path.join(
        figures_directory,
        f'{fname_tag}_stacked_{file_tag}_{timestamp}.png'
    )
    plt.savefig(fname)
    print(f'Figure saved to "{fname}"')
    return fname

# ---------------------------------------------------------------------------
# Helper: interpolate one event onto another's time grid for differencing
# ---------------------------------------------------------------------------

def _interp_to_grid(time_days_src, data_src, depth_src,
                    time_days_ref, depth_ref):
    """
    Bilinearly interpolate data_src (on depth × time)
    onto the time_days_ref × depth_ref grid using xarray.
    """
    da_src = xr.DataArray(
        data_src,
        dims=['depth', 'time'],          # depth is axis-0, time is axis-1
        coords={'time': time_days_src, 'depth': depth_src},
    )
    da_interp = da_src.interp(time=time_days_ref, depth=depth_ref,
                               method='linear')
    return da_interp.values


# ---------------------------------------------------------------------------
# Difference plot: 2026-present  minus  2015-2016
# ---------------------------------------------------------------------------

def plot_enso_difference(
    ds,
    tanom_smoothed,
    sanom_smoothed,
    event_a,          # (start_year, end_year, label)  — minuend   (2026)
    event_b,          # (start_year, end_year, label)  — subtrahend (2015)
    figures_directory,
    timestamp,
    latest_transect_time,
):
    """
    Two-panel figure (temperature, salinity) showing
    event_a anomaly  minus  event_b anomaly,
    both re-indexed to day-of-event so Jan 1 = day 0 for each.
    """

    start_a, end_a, label_a = event_a
    start_b, end_b, label_b = event_b

    # --- slice both events ---
    time_a, depth, ta_a, sa_a = _slice_event(ds, tanom_smoothed, sanom_smoothed, start_a, end_a)
    time_b, depth_b, ta_b, sa_b = _slice_event(ds, tanom_smoothed, sanom_smoothed, start_b, end_b)

    # --- interpolate event_b onto event_a's grid ---
    ta_b_interp = _interp_to_grid(time_b, ta_b, depth_b, time_a, depth)
    sa_b_interp = _interp_to_grid(time_b, sa_b, depth_b, time_a, depth)

    # --- differences ---
    ta_diff = ta_a - ta_b_interp
    sa_diff = sa_a - sa_b_interp

    # --- plot setup ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), dpi=300)
    fig.subplots_adjust(
        left=0.07, right=1.05, top=0.88, bottom=0.08, hspace=0.38
    )

    panel_cfg = [
        dict(
            ax=axes[0], data=ta_diff,
            cmap='bwr',
            divnorm=colors.TwoSlopeNorm(vcenter=0., vmin=-4, vmax=4),
            levels=[-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4],
            cbar_label=r'$\Delta\degree$C',
            title='Temperature Anomaly Difference',
        ),
        dict(
            ax=axes[1], data=sa_diff,
            cmap='BrBG_r',
            divnorm=colors.TwoSlopeNorm(vcenter=0., vmin=-0.75, vmax=0.75),
            levels=[-0.6, -0.4, -0.2, 0.2, 0.4, 0.6],
            cbar_label=r'$\Delta$PSU',
            title='Salinity Anomaly Difference',
        ),
    ]

    x_max_days = 365 * 2
    tick_pos, tick_labs = _month_ticks(start_a)   # labelled by event_a years

    for cfg in panel_cfg:
        ax   = cfg['ax']
        data = cfg['data']

        pcm = ax.pcolormesh(
            time_a, depth, data,
            cmap=cfg['cmap'], norm=cfg['divnorm'], shading='nearest',
        )
        cnt = ax.contour(
            time_a, depth, data,
            colors='black', norm=cfg['divnorm'],
            levels=cfg['levels'], alpha=0.65, linewidths=0.8,
        )
        ax.clabel(cnt, cnt.levels, inline=True, fontsize=8)

        # Glider deployment lines for event_a window only
        segs = _deployment_segments_in_window(DEPLOYMENT_PAIRS, start_a, end_a)
        for seg_start, seg_end in segs:
            ax.hlines(y=570, xmin=seg_start, xmax=seg_end, color='k', linewidth=1.5)

        ax.invert_yaxis()
        ax.set_yticks([0, 200, 400, 600])
        ax.set_ylim(600, 0)
        ax.set_ylabel('Depth (m)', fontsize=11)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, fontsize=9)
        ax.set_xlim(0, x_max_days)

        ax.set_title(cfg['title'], loc='left', fontsize=12, fontweight='bold', pad=4)
        ax.spines[:].set_linewidth(1.5)
        ax.tick_params(width=2, top=True, right=True, direction='in', labelsize=10, pad=8)

        cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.80)
        cbar.outline.set_linewidth(1.5)
        cbar.set_label(cfg['cbar_label'], rotation=0, labelpad=10, fontsize=10)

    fig.suptitle(
        f'Anomaly Difference: {label_a}  minus  {label_b}\n'
        f'Trinidad Head, CA — Averaged over 200 km from the coast',
        fontsize='x-large',
    )
    fig.text(
        0.78, 0.01,
        f'Latest transect date: {latest_transect_time.strftime("%Y-%m-%d")}',
        fontsize='small',
    )
    fig.text(
        0.02, 0.01,
        'Courtesy of the Oregon State University Glider Research Group, NANOOS, CeNCOOS',
        fontsize='small',
    )

    fname = os.path.join(
        figures_directory,
        f'anom_diff_2026_minus_2015_{timestamp}.png',
    )
    plt.savefig(fname)
    print(f'Figure saved to "{fname}"')
    return fname


# ---------------------------------------------------------------------------
# Drop-in replacement main() – keeps all original plots, adds the new ones
# ---------------------------------------------------------------------------

def main():

    ##################### Load Data #####################
    filepath = r'C:\Users\marqjace\OneDrive - Oregon State University\Desktop\Repositories\TH_Line\timeseries\data\timeseries_anomaly.nc'
    print(f'\nLoading data from "{filepath}"....')

    ds = xr.open_dataset(filepath)
    ds = ds.sortby('time')

    time = ds['time'].values
    latest_transect_time = pd.to_datetime(time.max())
    depth = ds['depth'].values
    tanom_smoothed = ds['temperature_anomaly'].rolling(time=3, depth=4, min_periods=1).mean()
    sanom_smoothed = ds['salinity_anomaly'].rolling(time=3, depth=4, min_periods=1).mean()
    print(f'\nApplying smoothing...')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    figures_directory = (
        r'C:/Users/marqjace/OneDrive - Oregon State University/'
        r'Desktop/Repositories/TH_Line/timeseries/figures/'
    )
    os.makedirs(figures_directory, exist_ok=True)

    # ------------------------------------------------------------------
    # El Niño events  (2-year windows: start_year, end_year, panel label)
    # ------------------------------------------------------------------
    el_nino_events = [
        (2015, 2016, 'El Niño 2015–2016'),
        (2023, 2024, 'El Niño 2023–2024'),
        (2026, 2027, 'El Niño 2026–present'),
    ]

    # ------------------------------------------------------------------
    # La Niña events
    # ------------------------------------------------------------------
    la_nina_events = [
        (2017, 2018, 'La Niña 2017–2018'),
        (2020, 2021, 'La Niña 2020–2021'),
        (2022, 2023, 'La Niña 2022–2023'),
    ]

    print('\nCreating stacked El Niño / La Niña figures....')

    for events, label in [(el_nino_events, 'El Niño'), (la_nina_events, 'La Niña')]:
        for var in ('temperature', 'salinity'):
            print(f'  Plotting {label} – {var}...')
            plot_stacked_enso(
                ds=ds,
                tanom_smoothed=tanom_smoothed,
                sanom_smoothed=sanom_smoothed,
                event_windows=events,
                variable=var,
                figures_directory=figures_directory,
                timestamp=timestamp,
                latest_transect_time=latest_transect_time,
            )
        
    # ------------------------------------------------------------------
    # Difference plot: 2026-present El Niño  minus  2015-2016 El Niño
    # ------------------------------------------------------------------
    print('\nCreating El Niño difference figure (2026 minus 2015)...')
    plot_enso_difference(
        ds=ds,
        tanom_smoothed=tanom_smoothed,
        sanom_smoothed=sanom_smoothed,
        event_a=(2026, 2027, 'El Niño 2026–present'),   # minuend
        event_b=(2015, 2016, 'El Niño 2015–2016'),       # subtrahend
        figures_directory=figures_directory,
        timestamp=timestamp,
        latest_transect_time=latest_transect_time,
    )

    print('\nDone!')

if __name__ == "__main__":
    main()