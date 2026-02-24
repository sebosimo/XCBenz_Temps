import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import os
import datetime

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

def get_available_runs():
    """Returns a list of run folders, newest first."""
    if not os.path.exists(CACHE_DIR):
        return []
    runs = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
    return sorted(runs, reverse=True)

def get_data_inventory(run_folder):
    """For a specific run, returns a list of locations and their available time steps."""
    inventory = {}
    run_path = os.path.join(CACHE_DIR, run_folder)
    
    if not os.path.exists(run_path):
        return {}

    locations = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    for loc in sorted(locations):
        loc_path = os.path.join(run_path, loc)
        steps = [f.replace(".nc", "") for f in os.listdir(loc_path) if f.endswith(".nc")]
        inventory[loc] = sorted(steps)
    return inventory

@st.cache_data(ttl=1800)
def render_time_height_plot(run_folder, location):
    """Generates a Time-Height cross-section of Lapse Rate."""
    loc_path = os.path.join(CACHE_DIR, run_folder, location)
    if not os.path.exists(loc_path): return None
    
    files = sorted([os.path.join(loc_path, f) for f in os.listdir(loc_path) if f.endswith(".nc")])
    if not files: return None

    times = []
    heights_list = []
    temps_list = []
    dewpoints_list = []
    u_list = []
    v_list = []
    
    for f in files:
        try:
            ds = xr.open_dataset(f)
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

    fig, ax = plt.subplots(figsize=(12, 8))
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
        ax.text(t_mid, -0.06, day.strftime('%a %d'), transform=ax.get_xaxis_transform(), 
                ha='center', va='top', fontweight='bold', fontsize=12)

    ax.set_ylim(0, 7)
    ax.set_ylabel("Altitude (km)", fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.tick_params(axis='both', labelsize=12)
    
    cbar = plt.colorbar(c, ax=ax)
    #cbar.set_label("Lapse Rate (°C/km)", fontsize=12)
    cbar.set_ticks([3, 6, 9])
    cbar.ax.set_yticklabels(['Stable (<0.3°C)', '0.6°C', 'Good (>0.9°C)'], rotation=270, va='center')
    cbar.ax.tick_params(labelsize=13)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    plt.subplots_adjust(bottom=0.15, top=0.98)
    return fig

@st.cache_data(ttl=3600)
def render_custom_emagram(file_path):
    """The core plotting engine with custom skew and lapse-rate coloring."""
    ds = xr.open_dataset(file_path)
    
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
    return fig

# --- APP UI ---
st.title("XCBenz Therm")

runs = get_available_runs()

if not runs:
    st.error("No data found. Please run fetch_data.py or pull data from GitHub.")
else:
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_run = st.selectbox("Model Run (UTC)", runs, index=0)

    if st.session_state.get("_last_run") != selected_run:
        st.session_state.forecast_index = 0
        st.session_state["_last_run"] = selected_run

    inventory = get_data_inventory(selected_run)
    location_list = list(inventory.keys())

    with col1:
        selected_loc = st.selectbox("Select Flying Site", location_list, 
                                    index=location_list.index("Sion") if "Sion" in location_list else 0)

    available_horizons = inventory.get(selected_loc, [])
    
    # Filter 2-hour steps for UI slider (Emagram only)
    slider_horizons = [h for h in available_horizons if int(h.split('_')[-1].replace("H", "")) % 2 == 0]
        
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
        file_to_plot = os.path.join(CACHE_DIR, selected_run, selected_loc, f"{selected_hor}.nc")
        
        # Header Info
        ds = xr.open_dataset(file_to_plot)
        if "valid_time" in ds.attrs:
            valid_dt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
        else:
            ref = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
            valid_dt = ref + datetime.timedelta(hours=int(ds.attrs["horizon"]))
        
        swiss_dt = valid_dt + datetime.timedelta(hours=1) 

        st.subheader(f"{selected_loc} {swiss_dt.strftime('%A %H:%M')} (LT)")
        
        # --- TABS LOGIC ---
        tab1, tab2 = st.tabs(["📈 Sounding-Wind", "📅 Lapsrate-Time"])
        
        with tab1:
            with st.spinner("Generating Emagram..."):
                fig = render_custom_emagram(file_to_plot)
                st.pyplot(fig, use_container_width=True)
            
            nav1, nav2 = st.columns([1, 1], gap="small")
            with nav1:
                st.button("◂ Previous", on_click=prev_step, use_container_width=True)
            with nav2:
                st.button("Next ▸", on_click=next_step, use_container_width=True)

            def format_slider(option):
                try:
                    base = datetime.datetime.strptime(selected_run, '%Y%m%d_%H%M')
                    hrs = int(option.split('_')[-1].replace("H", ""))
                    target = base + datetime.timedelta(hours=hrs + 1)
                    return target.strftime('%a %H:%M')
                except:
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
                fig_time = render_time_height_plot(selected_run, selected_loc)
                if fig_time:
                    st.pyplot(fig_time, use_container_width=True)
                else:
                    st.error("Could not generate overview. Ensure data is available.")
            
    else:
        st.warning("No time steps found for this location.")