import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import io
import os
import tempfile
import datetime
from zoneinfo import ZoneInfo
import requests

# This makes the app use the full width of your screen
st.set_page_config(page_title="XCBenz Therm", layout="wide")

# --- CSS: MARGIN TUNING & TOUCH BAR ---
st.markdown("""
    <style>
    /* 1. Global Container: Increased top padding so Title is visible */
    .block-container {
        padding-top: 2.5rem !important; /* CHANGED from 0.5rem to 2.5rem */
        padding-bottom: 2rem !important;
    }
    
    /* 2. Title (H1): Reduce padding */
    h1 {
        padding-top: 0rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* 3. Subheader (H3): Pull it closer */
    h3 {
        padding-top: 0rem !important;
        margin-top: -10px !important;
        margin-bottom: 0px !important;
    }

    /* 4. Touch Bar Buttons (Big & easy to tap) */
    div.stButton > button {
        width: 100% !important;
        height: 50px !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        margin-top: 5px !important;
    }
    
    /* 5. Force columns to stay side-by-side on mobile */
    div[data-testid="column"] {
        min-width: 0px !important;
        flex: 1 1 auto !important;
    }
    </style>
""", unsafe_allow_html=True)

CACHE_DIR = "cache_data"

_GH_REPO = "sebosimo/Data_Fetch-ICON-CH1"
_GH_RAW  = f"https://raw.githubusercontent.com/{_GH_REPO}/data"

@st.cache_data(ttl=1800)
def _fetch_manifest():
    """Download and cache manifest.json from GitHub for 30 min."""
    r = requests.get(f"{_GH_RAW}/manifest.json", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=3600)
def _gh_nc_bytes(repo_path):
    """Download raw .nc bytes from GitHub, cached 1 hour per file."""
    r = requests.get(f"{_GH_RAW}/{repo_path}", timeout=30)
    r.raise_for_status()
    return r.content

def _open_nc(repo_path):
    """Open an xarray Dataset from a GitHub repo-relative path."""
    content = _gh_nc_bytes(repo_path)
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    ds = None
    try:
        ds = xr.open_dataset(tmp_path)
        ds.load()
    finally:
        if ds is not None:
            ds.close()  # Release file handle before deletion (required on Windows)
        os.unlink(tmp_path)
    return ds

def get_available_runs():
    """Returns a list of run folders from GitHub manifest, newest first."""
    try:
        return sorted(_fetch_manifest().get("runs", {}).keys(), reverse=True)
    except Exception:
        return []

def get_available_ch2_runs():
    """Returns a list of CH2 run folders from manifest, newest first."""
    try:
        return sorted(_fetch_manifest().get("runs_ch2", {}).keys(), reverse=True)
    except Exception:
        return []

def get_data_inventory(run_folder):
    """For a specific run, returns {location: [step_strings]} from GitHub manifest."""
    try:
        return _fetch_manifest().get("runs", {}).get(run_folder, {})
    except Exception:
        return {}

def get_ch2_data_inventory(run_folder):
    """For a specific CH2 run, returns {location: [step_strings]} from manifest."""
    try:
        return _fetch_manifest().get("runs_ch2", {}).get(run_folder, {})
    except Exception:
        return {}

def find_matching_ch2_run(ch1_run_tag):
    """Return the most recent CH2 run whose ref_time is <= the CH1 run's ref_time."""
    try:
        ch1_ref = datetime.datetime.strptime(ch1_run_tag, '%Y%m%d_%H%M')
        ch2_runs = sorted(
            _fetch_manifest().get("runs_ch2", {}).keys(), reverse=True
        )
        for ch2_tag in ch2_runs:
            if datetime.datetime.strptime(ch2_tag, '%Y%m%d_%H%M') <= ch1_ref:
                return ch2_tag
    except Exception:
        pass
    return None

def get_ch2_extension_steps(ch1_run_tag, ch2_run_tag, ch2_inv, location):
    """Return CH2 step labels whose valid_time is strictly after the last CH1 valid_time."""
    ch1_steps = get_data_inventory(ch1_run_tag).get(location, [])
    if not ch1_steps or not ch2_run_tag:
        return []
    try:
        ch1_ref = datetime.datetime.strptime(ch1_run_tag, '%Y%m%d_%H%M')
        ch2_ref = datetime.datetime.strptime(ch2_run_tag, '%Y%m%d_%H%M')
        last_h = int(ch1_steps[-1].replace("H", ""))
        cutoff = ch1_ref + datetime.timedelta(hours=last_h)
        return [
            s for s in ch2_inv.get(location, [])
            if ch2_ref + datetime.timedelta(hours=int(s.replace("H", ""))) > cutoff
        ]
    except Exception:
        return []

@st.cache_data(ttl=3600)
def render_time_height_plot(run_folder, location):
    """Generates a Time-Height cross-section of Lapse Rate."""
    steps = get_data_inventory(run_folder).get(location, [])
    if not steps:
        return None
    github_paths = [f"{CACHE_DIR}/{run_folder}/{location}/{step}.nc" for step in steps]

    times = []
    heights_list = []
    temps_list = []
    dewpoints_list = []
    u_list = []
    v_list = []

    for github_path in github_paths:
        try:
            ds = _open_nc(github_path)
            p = ds["P"].values * units.Pa
            t = (ds["T"].values * units.K).to(units.degC)
            t_kelvin = ds["T"].values * units.K
            t = t_kelvin.to(units.degC)
            qv = ds["QV"].values * units('kg/kg')
            u_val = (ds["U"].values * units('m/s')).to('km/h').m
            v_val = (ds["V"].values * units('m/s')).to('km/h').m
            td = mpcalc.dewpoint_from_specific_humidity(p, t_kelvin, qv).to(units.degC)
            if "HEIGHT" in ds:
                z = (ds["HEIGHT"].values * units.m).to(units.km).m
            else:
                z = mpcalc.pressure_to_height_std(p).to(units.km).m
            if "valid_time" in ds.attrs:
                vt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
            else:
                # Fallback: calc from ref_time and horizon
                ref = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
                vt = ref + datetime.timedelta(hours=int(ds.attrs["horizon"]))
            # Convert UTC → Europe/Zurich (handles CET/CEST automatically), strip tz for matplotlib
            vt = vt.astimezone(ZoneInfo("Europe/Zurich")).replace(tzinfo=None)
            times.append(vt)
            heights_list.append(z)
            temps_list.append(t.m)
            dewpoints_list.append(td.m)
            u_list.append(u_val)
            v_list.append(v_val)
        except:
            continue

    if not times: return None

    reg_z = np.arange(0, 7.05, 0.05) 
    reg_t = np.zeros((len(reg_z), len(times)))
    reg_td = np.zeros((len(reg_z), len(times)))
    reg_u = np.zeros((len(reg_z), len(times)))
    reg_v = np.zeros((len(reg_z), len(times)))
    
    for i in range(len(times)):
        z_col = heights_list[i]
        t_col = temps_list[i]
        td_col = dewpoints_list[i]
        u_col = u_list[i]
        v_col = v_list[i]
        sort_idx = np.argsort(z_col)
        reg_t[:, i] = np.interp(reg_z, z_col[sort_idx], t_col[sort_idx])
        reg_td[:, i] = np.interp(reg_z, z_col[sort_idx], td_col[sort_idx])
        reg_u[:, i] = np.interp(reg_z, z_col[sort_idx], u_col[sort_idx])
        reg_v[:, i] = np.interp(reg_z, z_col[sort_idx], v_col[sort_idx])

    dt_dz = -np.gradient(reg_t, axis=0) / 0.05 
    lapse_rate = dt_dz

    # --- MASKING & TERRAIN ---
    surface_heights = [np.min(h) for h in heights_list]
    mask = np.zeros_like(lapse_rate, dtype=bool)
    for i, s_h in enumerate(surface_heights):
        mask[:, i] = reg_z < s_h
    lapse_rate_masked = np.ma.masked_where(mask, lapse_rate)

    fig, ax = plt.subplots(figsize=(12, 13.5))
    time_nums = mdates.date2num(times)
    X, Y = np.meshgrid(time_nums, reg_z)
    
    levels = np.linspace(3, 9, 100)
    cmap = plt.get_cmap("RdYlGn") 
    
    c = ax.contourf(X, Y, lapse_rate_masked, levels=levels, cmap=cmap, extend='both')
    
    # Cloud Overlay (T - Td < 1.0°C)
    depression = reg_t - reg_td
    depression_masked = np.ma.masked_where(mask, depression)
    ax.contourf(X, Y, depression_masked, levels=[-100, 1.0], colors=['#778899'], alpha=0.6)
    
    # Wind Barbs Overlay
    reg_u_masked = np.ma.masked_where(mask, reg_u)
    reg_v_masked = np.ma.masked_where(mask, reg_v)
    
    skip_z = 10  # Every 500m (0.05 * 10)
    skip_t = 2   # Every 2nd time step
    ax.barbs(X[::skip_z, ::skip_t], Y[::skip_z, ::skip_t], 
             reg_u_masked[::skip_z, ::skip_t], reg_v_masked[::skip_z, ::skip_t],
             length=5, color='#444444', alpha=0.5, linewidth=0.8)
    
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    # Terrain Fill
    ax.fill_between(time_nums, 0, surface_heights, color='#e0e0e0', zorder=2)
    
    # X-Axis Formatting
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Day Labels
    unique_days = sorted(list(set(t.date() for t in times)))
    for day in unique_days:
        # Draw vertical line at midnight to separate days
        midnight = datetime.datetime.combine(day, datetime.time.min)
        if times[0].tzinfo:
            midnight = midnight.replace(tzinfo=times[0].tzinfo)
        if times[0] <= midnight <= times[-1]:
            ax.plot([mdates.date2num(midnight)]*2, [-0.08, -0.04], color='black', linewidth=1, transform=ax.get_xaxis_transform(), clip_on=False)

        day_times = [t for t in times if t.date() == day]
        if not day_times: continue

        if len(day_times) == 1 and day_times[0] == times[-1]:
            continue

        t_mid = (mdates.date2num(day_times[0]) + mdates.date2num(day_times[-1])) / 2
        ax.text(t_mid, -0.05, day.strftime('%a %d'), transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontweight='bold', fontsize=12)

    ax.set_ylim(0, 7)
    ax.set_ylabel("Altitude (km)", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.tick_params(axis='both', labelsize=12)
    
    cbar = plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.09, aspect=50, shrink=1.0)
    cbar.set_ticks([3, 6, 9])
    cbar.ax.set_xticklabels(['Stable (<0.3°C/100m)', '0.6°C/100m', 'Good (>0.9°C/100m)'], fontsize=11)
    cbar.ax.tick_params(labelsize=11)

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(bottom=0.14, top=0.99, left=0.07, right=0.99)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return buf.getvalue()

@st.cache_data(ttl=21600)
def render_ch2_time_height_plot(run_folder, location, steps_tuple):
    """5-day CH2 Time-Height lapse rate plot. steps_tuple is a tuple for cache-key stability."""
    steps = list(steps_tuple)
    if not steps:
        return None
    github_paths = [f"cache_data_ch2/{run_folder}/{location}/{step}.nc" for step in steps]

    times = []
    heights_list = []
    temps_list = []
    dewpoints_list = []
    u_list = []
    v_list = []

    for github_path in github_paths:
        try:
            ds = _open_nc(github_path)
            p = ds["P"].values * units.Pa
            t_kelvin = ds["T"].values * units.K
            t = t_kelvin.to(units.degC)
            qv = ds["QV"].values * units('kg/kg')
            u_val = (ds["U"].values * units('m/s')).to('km/h').m
            v_val = (ds["V"].values * units('m/s')).to('km/h').m
            td = mpcalc.dewpoint_from_specific_humidity(p, t_kelvin, qv).to(units.degC)
            if "HEIGHT" in ds:
                z = (ds["HEIGHT"].values * units.m).to(units.km).m
            else:
                z = mpcalc.pressure_to_height_std(p).to(units.km).m
            if "valid_time" in ds.attrs:
                vt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
            else:
                ref = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
                vt = ref + datetime.timedelta(hours=int(ds.attrs["horizon"]))
            vt = vt.astimezone(ZoneInfo("Europe/Zurich")).replace(tzinfo=None)
            times.append(vt)
            heights_list.append(z)
            temps_list.append(t.m)
            dewpoints_list.append(td.m)
            u_list.append(u_val)
            v_list.append(v_val)
        except Exception:
            continue

    if not times:
        return None

    reg_z = np.arange(0, 7.05, 0.05)
    reg_t = np.zeros((len(reg_z), len(times)))
    reg_td = np.zeros((len(reg_z), len(times)))
    reg_u = np.zeros((len(reg_z), len(times)))
    reg_v = np.zeros((len(reg_z), len(times)))

    for i in range(len(times)):
        sort_idx = np.argsort(heights_list[i])
        reg_t[:, i]  = np.interp(reg_z, heights_list[i][sort_idx], temps_list[i][sort_idx])
        reg_td[:, i] = np.interp(reg_z, heights_list[i][sort_idx], dewpoints_list[i][sort_idx])
        reg_u[:, i]  = np.interp(reg_z, heights_list[i][sort_idx], u_list[i][sort_idx])
        reg_v[:, i]  = np.interp(reg_z, heights_list[i][sort_idx], v_list[i][sort_idx])

    dt_dz = -np.gradient(reg_t, axis=0) / 0.05
    lapse_rate = dt_dz

    surface_heights = [np.min(h) for h in heights_list]
    mask = np.zeros_like(lapse_rate, dtype=bool)
    for i, s_h in enumerate(surface_heights):
        mask[:, i] = reg_z < s_h
    lapse_rate_masked = np.ma.masked_where(mask, lapse_rate)

    fig, ax = plt.subplots(figsize=(16, 13.5))
    time_nums = mdates.date2num(times)
    X, Y = np.meshgrid(time_nums, reg_z)

    levels = np.linspace(3, 9, 100)
    cmap = plt.get_cmap("RdYlGn")
    c = ax.contourf(X, Y, lapse_rate_masked, levels=levels, cmap=cmap, extend='both')

    # Cloud overlay
    depression = reg_t - reg_td
    depression_masked = np.ma.masked_where(mask, depression)
    ax.contourf(X, Y, depression_masked, levels=[-100, 1.0], colors=['#778899'], alpha=0.6)

    # Wind barbs
    reg_u_masked = np.ma.masked_where(mask, reg_u)
    reg_v_masked = np.ma.masked_where(mask, reg_v)
    skip_z = 10
    skip_t = 3   # sparser for 5-day range
    ax.barbs(X[::skip_z, ::skip_t], Y[::skip_z, ::skip_t],
             reg_u_masked[::skip_z, ::skip_t], reg_v_masked[::skip_z, ::skip_t],
             length=5, color='#444444', alpha=0.5, linewidth=0.8)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
    ax.fill_between(time_nums, 0, surface_heights, color='#e0e0e0', zorder=2)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    unique_days = sorted(list(set(t.date() for t in times)))
    for day in unique_days:
        midnight = datetime.datetime.combine(day, datetime.time.min)
        if times[0] <= midnight <= times[-1]:
            ax.plot([mdates.date2num(midnight)] * 2, [-0.08, -0.04],
                    color='black', linewidth=1,
                    transform=ax.get_xaxis_transform(), clip_on=False)
        day_times = [t for t in times if t.date() == day]
        if not day_times:
            continue
        if len(day_times) == 1 and day_times[0] == times[-1]:
            continue
        t_mid = (mdates.date2num(day_times[0]) + mdates.date2num(day_times[-1])) / 2
        ax.text(t_mid, -0.05, day.strftime('%a %d'),
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontweight='bold', fontsize=12)

    # Model label
    ax.text(0.01, 0.98, "ICON-CH2 · 5-day", transform=ax.transAxes,
            fontsize=11, color='white', va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='#333333', alpha=0.7))

    ax.set_ylim(0, 7)
    ax.set_ylabel("Altitude (km)", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    cbar = plt.colorbar(c, ax=ax, orientation='horizontal', pad=0.09, aspect=50, shrink=1.0)
    cbar.set_ticks([3, 6, 9])
    cbar.ax.set_xticklabels(['Stable (<0.3°C/100m)', '0.6°C/100m', 'Good (>0.9°C/100m)'],
                            fontsize=11)
    cbar.ax.tick_params(labelsize=11)

    ax.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(bottom=0.14, top=0.99, left=0.06, right=0.99)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(ttl=3600)
def render_custom_emagram(file_path):
    """The core plotting engine with custom skew and lapse-rate coloring."""
    ds = _open_nc(file_path)
    
    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K).to(units.degC)
    qv = ds["QV"].values * units('kg/kg')
    u_ms, v_ms = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    td = mpcalc.dewpoint_from_specific_humidity(p, t, qv)
    if "HEIGHT" in ds:
        z = (ds["HEIGHT"].values * units.m).to(units.km)
    else:
        z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

    inds = z.argsort()
    z_p, t_p, td_p = z[inds].m, t[inds].m, td[inds].m
    u_p, v_p, w_p = u_ms[inds].to('km/h').m, v_ms[inds].to('km/h').m, wind_speed_kmh[inds]
    
    mask = z_p <= 7.0
    z_p, t_p, td_p, u_p, v_p, w_p = z_p[mask], t_p[mask], td_p[mask], u_p[mask], v_p[mask], w_p[mask]

    # Skew Logic
    SKEW_FACTOR = 5 
    def skew_x(temp, height): return temp + (height * SKEW_FACTOR)

    skew_t = skew_x(t_p, z_p)
    skew_td = skew_x(td_p, z_p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True, 
                                   gridspec_kw={'width_ratios': [4, 1], 'wspace': 0})
    
    ax1.set_ylim(0, 7.0)
    min_x, max_x = min(np.min(skew_t), np.min(skew_td)), max(np.max(skew_t), np.max(skew_td))
    ax1.set_xlim(min_x - 5, max_x + 5)

    for temp_base in range(-60, 60, 5):
        ax1.plot([skew_x(temp_base, 0), skew_x(temp_base, 7)], [0, 7], color='blue', alpha=0.04, zorder=1)
    for theta_base in range(-60, 100, 5):
        ax1.plot([skew_x(theta_base, 0), skew_x(theta_base-70, 7)], [0, 7], color='brown', alpha=0.06, zorder=1)

    dt, dz = np.diff(t_p), np.diff(z_p)
    lapse_rate = -(dt / dz)
    points = np.array([skew_t, z_p]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=3, vmax=9) 
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=5, zorder=5)
    lc.set_array(lapse_rate)
    ax1.add_collection(lc)
    ax1.plot(skew_td, z_p, color='blue', linewidth=2, zorder=4, alpha=0.7)

    ax2.plot(w_p, z_p, color='blue', linewidth=1, alpha=0.6)
    ax2.set_xlim(0, 80)
    ax2.set_xticks([0, 20, 40, 60])
    ax2.set_xlabel("Wind (km/h)", fontsize=14)
    ax2.tick_params(axis='x', labelsize=13) 
    ax2.grid(True, alpha=0.15)
    ax2.tick_params(axis='y', left=False, labelleft=False)
    
    step = max(1, len(z_p) // 12)
    ax2.barbs(np.ones_like(z_p[::step]) * 70, z_p[::step], 
              u_p[::step], v_p[::step], length=5, color='black', alpha=0.7)

    ax1.set_ylabel("") 
    ax1.text(-0.01, -0.06, "km", transform=ax1.transAxes, fontsize=13, ha='right', va='top')
    ax1.set_xlabel("Temperature (°C)", fontsize=14)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.grid(True, axis='y', alpha=0.2)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

# --- APP UI ---
st.title("XCBenz Therm")

runs = get_available_runs()

if not runs:
    st.error("No data found. Please run fetch_data.py or pull data from GitHub.")
else:
    ch2_runs = get_available_ch2_runs()
    col_run, col_ch2 = st.columns(2)
    with col_run:
        selected_run = st.selectbox("CH1 Run (UTC)", runs, index=0)
    with col_ch2:
        if ch2_runs:
            default_ch2 = find_matching_ch2_run(selected_run)
            default_idx = ch2_runs.index(default_ch2) if default_ch2 in ch2_runs else 0
            ch2_run_tag = st.selectbox("CH2 Run (UTC)", ch2_runs, index=default_idx)
        else:
            st.selectbox("CH2 Run (UTC)", ["No data yet"], disabled=True)
            ch2_run_tag = None

    if st.session_state.get("_last_run") != selected_run:
        st.session_state.forecast_index = 0
        st.session_state["_last_run"] = selected_run

    inventory = get_data_inventory(selected_run)
    location_list = list(inventory.keys())

    selected_loc = st.selectbox("Select Flying Site", location_list,
                                index=location_list.index("Sion") if "Sion" in location_list else 0)

    available_horizons = inventory.get(selected_loc, [])

    ch2_inventory = get_ch2_data_inventory(ch2_run_tag) if ch2_run_tag else {}
    ch2_extension_steps = get_ch2_extension_steps(
        selected_run, ch2_run_tag, ch2_inventory, selected_loc
    )

    # Combined slider: CH1 steps first, then CH2 extension (tagged with "CH2:" prefix)
    slider_horizons = available_horizons + [f"CH2:{s}" for s in ch2_extension_steps]

    if slider_horizons:
        if 'forecast_index' not in st.session_state:
            st.session_state.forecast_index = 0
        
        st.session_state.forecast_index = min(st.session_state.forecast_index, len(slider_horizons) - 1)

        current_slider_value = slider_horizons[st.session_state.forecast_index]
        if 'slider_key' not in st.session_state or st.session_state.slider_key != current_slider_value:
            st.session_state.slider_key = current_slider_value

        def prev_step():
            st.session_state.forecast_index = max(0, st.session_state.forecast_index - 1)
            st.session_state.slider_key = slider_horizons[st.session_state.forecast_index]

        def next_step():
            st.session_state.forecast_index = min(len(slider_horizons) - 1, st.session_state.forecast_index + 1)
            st.session_state.slider_key = slider_horizons[st.session_state.forecast_index]

        def slider_callback():
            st.session_state.forecast_index = slider_horizons.index(st.session_state.slider_key)

        selected_hor = slider_horizons[st.session_state.forecast_index]

        # Route file path: CH2-prefixed steps read from cache_data_ch2/
        if selected_hor.startswith("CH2:"):
            step = selected_hor[4:]   # e.g. "H034"
            file_to_plot = f"cache_data_ch2/{ch2_run_tag}/{selected_loc}/{step}.nc"
        else:
            file_to_plot = f"{CACHE_DIR}/{selected_run}/{selected_loc}/{selected_hor}.nc"

        # Header Info
        ds = _open_nc(file_to_plot)
        if "valid_time" in ds.attrs:
            valid_dt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
        else:
            ref = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
            valid_dt = ref + datetime.timedelta(hours=int(ds.attrs["horizon"]))

        swiss_dt = valid_dt.astimezone(ZoneInfo("Europe/Zurich"))
        source_label = "ICON-CH2" if selected_hor.startswith("CH2:") else "ICON-CH1"

        st.subheader(f"{selected_loc} {swiss_dt.strftime('%A %H:%M')} (LT) · {source_label}")
        
        # --- TABS LOGIC ---
        tab1, tab2, tab3 = st.tabs(["📈 Sounding-Wind", "📅 CH1 (~33h)", "📅 CH2 (5-day)"])

        with tab1:
            with st.spinner("Generating Emagram..."):
                img = render_custom_emagram(file_to_plot)
                st.image(img, use_container_width=True)

            nav1, nav2 = st.columns([1, 1], gap="small")
            with nav1:
                st.button("◂ Previous", on_click=prev_step, width='stretch')
            with nav2:
                st.button("Next ▸", on_click=next_step, width='stretch')

            def format_slider(option):
                try:
                    if option.startswith("CH2:"):
                        step = option[4:]   # e.g. "H034"
                        base = datetime.datetime.strptime(ch2_run_tag, '%Y%m%d_%H%M').replace(
                            tzinfo=datetime.timezone.utc)
                        hrs = int(step.replace("H", ""))
                    else:
                        base = datetime.datetime.strptime(selected_run, '%Y%m%d_%H%M').replace(
                            tzinfo=datetime.timezone.utc)
                        hrs = int(option.replace("H", ""))
                    target = (base + datetime.timedelta(hours=hrs)).astimezone(ZoneInfo("Europe/Zurich"))
                    return target.strftime('%a %H:%M')
                except Exception:
                    return option

            st.select_slider(
                "Forecast Hour", 
                options=slider_horizons, 
                key="slider_key",
                label_visibility="collapsed",
                format_func=format_slider,
                on_change=slider_callback
            )

        with tab2:
            with st.spinner("Calculating full day evolution..."):
                img_time = render_time_height_plot(selected_run, selected_loc)
                if img_time:
                    st.image(img_time, use_container_width=True)
                else:
                    st.error("Could not generate overview. Ensure data is available.")

        with tab3:
            if ch2_run_tag:
                ch2_all_steps = tuple(ch2_inventory.get(selected_loc, []))
                if ch2_all_steps:
                    with st.spinner("Loading 5-day ICON-CH2 forecast..."):
                        img_ch2 = render_ch2_time_height_plot(
                            ch2_run_tag, selected_loc, ch2_all_steps
                        )
                        if img_ch2:
                            st.image(img_ch2, use_container_width=True)
                        else:
                            st.error("Could not generate CH2 plot.")
                else:
                    st.info("No CH2 data for this location yet.")
            else:
                st.info("No ICON-CH2 data available yet. It will appear after the next CI run that fetches a new CH2 run.")
            
    else:
        st.warning("No time steps found for this location.")