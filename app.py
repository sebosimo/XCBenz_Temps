import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
import os
import datetime

# This makes the app use the full width of your screen
st.set_page_config(page_title="XCBenz Therm", layout="wide")

CACHE_DIR = "cache_data"
def get_available_runs():
    """Returns a list of run folders, newest first."""
    if not os.path.exists(CACHE_DIR):
        return []
    # Get all folder names in cache_data
    runs = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
    # Sort them alphabetically (which works for YYYYMMDD_HHMM format) and reverse for newest first
    return sorted(runs, reverse=True)

def get_data_inventory(run_folder):
    """For a specific run, returns a list of locations and their available time steps."""
    inventory = {}
    run_path = os.path.join(CACHE_DIR, run_folder)
    
    if not os.path.exists(run_path):
        return {}

    # 1. Find all location subfolders
    locations = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]
    
    for loc in sorted(locations):
        loc_path = os.path.join(run_path, loc)
        # 2. Find all .nc files for that location
        steps = [f.replace(".nc", "") for f in os.listdir(loc_path) if f.endswith(".nc")]
        inventory[loc] = sorted(steps)
        
    return inventory
@st.cache_data
def render_custom_emagram(file_path):
    """The core plotting engine with custom skew and lapse-rate coloring."""
    ds = xr.open_dataset(file_path)
    
    # 1. Physical Extraction
    # We use 'level' as the dimension because of our fetch_data fix
    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K).to(units.degC)
    qv = ds["QV"].values * units('kg/kg')
    u_ms, v_ms = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    # Calculate Dewpoint and Altitude
    td = mpcalc.dewpoint_from_specific_humidity(p, t, qv)
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

    # Sort by height and mask to 7km (Aviation relevant area)
    inds = z.argsort()
    z_p, t_p, td_p = z[inds].m, t[inds].m, td[inds].m
    u_p, v_p, w_p = u_ms[inds].to('km/h').m, v_ms[inds].to('km/h').m, wind_speed_kmh[inds]
    
    mask = z_p <= 7.0
    z_p, t_p, td_p, u_p, v_p, w_p = z_p[mask], t_p[mask], td_p[mask], u_p[mask], v_p[mask], w_p[mask]

    # 2. Skew Configuration (SKEW_FACTOR 5 means 0.5°C/100m is a vertical line)
    SKEW_FACTOR = 5 
    def skew_x(temp, height): return temp + (height * SKEW_FACTOR)

    skew_t = skew_x(t_p, z_p)
    skew_td = skew_x(td_p, z_p)

    # 3. Figure Setup (Wide layout for mobile)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True, 
                                   gridspec_kw={'width_ratios': [4, 1], 'wspace': 0})
    
    ax1.set_ylim(0, 7.0)
    # Dynamic Zoom: Center the X-axis on the data
    min_x, max_x = min(np.min(skew_t), np.min(skew_td)), max(np.max(skew_t), np.max(skew_td))
    ax1.set_xlim(min_x - 5, max_x + 5)

    # 4. Draw Background Helper Lines
    # Isotherms (Blue, tilted ~45°)
    for temp_base in range(-60, 60, 5):
        ax1.plot([skew_x(temp_base, 0), skew_x(temp_base, 7)], [0, 7], color='blue', alpha=0.04, zorder=1)

    # Dry Adiabats (Brown, 1.0°C/100m)
    for theta_base in range(-60, 100, 5):
        # Math: T decreases 10° per km. Skew correction adds 5° per km.
        # Result: Dry adiabat leans left at 5° per km.
        ax1.plot([skew_x(theta_base, 0), skew_x(theta_base-70, 7)], [0, 7], color='brown', alpha=0.06, zorder=1)

    # 5. Plot Temperature with Lapse-Rate Coloring (Heatmap)
    dt, dz = np.diff(t_p), np.diff(z_p)
    lapse_rate = -(dt / dz) # Positive = cooling with height
    
    points = np.array([skew_t, z_p]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # RdYlGn: Red (Lapse Rate < 4: Stable/Inversion), Green (Lapse Rate > 8: Unstable)
    norm = Normalize(vmin=3, vmax=9) 
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=5, zorder=5)
    lc.set_array(lapse_rate)
    ax1.add_collection(lc)

    # Plot Dewpoint (Blue)
    ax1.plot(skew_td, z_p, color='blue', linewidth=2, zorder=4, alpha=0.7)

    # 6. Wind Panel (Right Side)
    ax2.plot(w_p, z_p, color='blue', linewidth=1.5, alpha=0.3)
    ax2.set_xlim(0, 80)
    ax2.set_xticks([0, 20, 40, 60])
    ax2.set_xlabel("Wind (km/h)", fontsize=10)
    ax2.grid(True, alpha=0.15)
    
    # Draw Wind Barbs
    step = max(1, len(z_p) // 12)
    ax2.barbs(np.ones_like(z_p[::step]) * 70, z_p[::step], 
              u_p[::step], v_p[::step], length=5, color='black', alpha=0.7)

    # Axis Labels
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.2)
    
    return fig
st.title("XCBenz Therm")

# 1. Scan for runs
runs = get_available_runs()

if not runs:
    st.error("No data found. Please run fetch_data.py or pull data from GitHub.")
else:
    # 2. Site and Run Selection (Top Row)
    col1, col2 = st.columns([3, 1])
    
    with col2:
        selected_run = st.selectbox("Model Run (UTC)", runs, index=0)
    
    # Get inventory for the chosen run
    inventory = get_data_inventory(selected_run)
    location_list = list(inventory.keys())

    with col1:
        selected_loc = st.selectbox("Select Flying Site", location_list, 
                                    index=location_list.index("Sion") if "Sion" in location_list else 0)

    # 3. Time Slider
    # Get the specific hours available for the chosen site
    available_horizons = inventory.get(selected_loc, [])
        
    if available_horizons:
        # Site and Horizon selection
        selected_hor = st.select_slider("Select Forecast Hour", 
                                        options=available_horizons, 
                                        value=available_horizons[0])
        
        file_to_plot = os.path.join(CACHE_DIR, selected_run, selected_loc, f"{selected_hor}.nc")
        
        # Accurate Valid Time Header
        ds = xr.open_dataset(file_to_plot)
        valid_dt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
        swiss_dt = valid_dt + datetime.timedelta(hours=1) # Winter time correction

        st.subheader(f"{selected_loc} at {swiss_dt.strftime('%A %H:%M')} (Local Time)")
        
        # --- THE PLOT ---
        with st.spinner("Generating Emagram..."):
            fig = render_custom_emagram(file_to_plot)
            st.pyplot(fig, use_container_width=True)
            
        st.caption(f"Model Run: {selected_run} UTC | Forecast Step: {selected_hor} | Altitude: km")
    else:
        st.warning("No time steps found for this location.")