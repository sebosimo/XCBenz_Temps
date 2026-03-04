import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob
import numpy as np

# --- Configuration ---
CACHE_DIR = "cache_data"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_plot(file_path):
    """Processes a single NetCDF file and saves a Skew-T plot."""
    try:
        ds = xr.open_dataset(file_path)
        
        # Metadata from attributes
        loc_name = ds.attrs.get("location", "Unknown")
        ref_time_str = ds.attrs.get("ref_time", "")
        valid_time_str = ds.attrs.get("valid_time", "")
        horizon_h = ds.attrs.get("horizon_h", 0)
        
        # 1. Extract values and FORCE SQUEEZE to avoid IndexError
        # This turns (1, 80) into (80,)
        p = ds["P"].values.squeeze() * units.Pa
        t = (ds["T"].values.squeeze() * units.K).to(units.degC)
        u_ms = ds["U"].values.squeeze() * units('m/s')
        v_ms = ds["V"].values.squeeze() * units('m/s')
        hum = ds["HUM"].values.squeeze()
        
        if ds.attrs.get("HUM_TYPE") == "RELHUM":
            td = mpcalc.dewpoint_from_relative_humidity(t, hum / 100.0)
        else:
            td = mpcalc.dewpoint_from_specific_humidity(p, t, hum * units('kg/kg'))

        if "HEIGHT" in ds:
            # Use real height from model (in meters)
            z = (ds["HEIGHT"].values.squeeze() * units.m).to(units.km)
            print(f"Using real HEIGHT for {loc_name}")
        elif "H" in ds: # Alternative naming
            z = (ds["H"].values.squeeze() * units.m).to(units.km)
        else:
            # Fallback to approximation
            z = mpcalc.pressure_to_height_std(p).to(units.km)
            print(f"Using ISA approximation for {loc_name}")
        
        # Wind conversion
        u_kmh = u_ms.to('km/h').m
        v_kmh = v_ms.to('km/h').m
        wind_speed_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

        # Sorting and Masking
        inds = z.argsort()
        z_plot = z[inds].m
        t_plot, td_plot = t[inds].m, td[inds].m
        u_plot, v_plot, wind_plot = u_kmh[inds], v_kmh[inds], wind_speed_kmh[inds]
        
        z_max = 7.0
        mask = z_plot <= z_max
        z_plot, t_plot, td_plot, wind_plot = z_plot[mask], t_plot[mask], td_plot[mask], wind_plot[mask]
        u_plot, v_plot = u_plot[mask], v_plot[mask]

        # Skew Logic
        SKEW_FACTOR = 5 
        def skew_x(temp, height): return temp + (height * SKEW_FACTOR)

        # 2. Figure Setup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 9), sharey=True, 
                                       gridspec_kw={'width_ratios': [3, 1], 'wspace': 0})
        
        ax1.set_ylim(0, z_max)
        skew_t = skew_x(t_plot, z_plot)
        skew_td = skew_x(td_plot, z_plot)
        min_x, max_x = min(np.min(skew_t), np.min(skew_td)), max(np.max(skew_t), np.max(skew_td))
        padding = 8
        ax1.set_xlim(min_x - padding, max_x + padding)

        # Background Grid (Standard lines)
        z_ref = np.linspace(0, z_max, 100) * units.km
        p_ref = mpcalc.height_to_pressure_std(z_ref)
        ax1.grid(True, axis='y', color='gray', alpha=0.3)

        # Draw Isotherms
        for temp_base in range(-150, 151, 5):
            xb, xt = skew_x(temp_base, 0), skew_x(temp_base, z_max)
            if max(xb, xt) >= (min_x-padding) and min(xb, xt) <= (max_x+padding):
                ax1.plot([xb, xt], [0, z_max], color='blue', alpha=0.06, zorder=1)

        # Draw Thermo Data (Lapse Rate Coloring)
        dt, dz = np.diff(t_plot), np.diff(z_plot)
        lapse_rate = - (dt / dz) 
        points = np.array([skew_t, z_plot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(vmin=-3, vmax=10), linewidth=4, zorder=5)
        lc.set_array(lapse_rate)
        ax1.add_collection(lc)
        ax1.plot(skew_td, z_plot, color='blue', linewidth=1.5, zorder=5, alpha=0.8)

        # Panel 2: Wind
        ax2.plot(wind_plot, z_plot, color='blue', linewidth=1, alpha=0.4)
        ax2.set_xlim(0, 100) 
        ax2.set_xticks(np.arange(0, 101, 20))
        ax2.grid(True, alpha=0.2)
        step = max(1, len(z_plot) // 15) 
        ax2.barbs(np.ones_like(z_plot[::step]) * 85, z_plot[::step], 
                  u_plot[::step], v_plot[::step], length=6)

        # Title and Labels
        title_str = (f"{loc_name} | Run: {ref_time_str}\n"
                     f"Forecast: {valid_time_str} (+{horizon_h}h)")
        fig.suptitle(title_str, fontsize=14, y=0.96)
        ax1.set_ylabel("Altitude (km MSL)")
        ax1.set_xlabel("Temperature (°C)")

        # Save and Close
        out_name = f"{loc_name}_H{horizon_h:02d}.png".replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=120, bbox_inches='tight')
        plt.close(fig) # IMPORTANT: Prevents memory crash when looping hundreds of files
        return True
    except Exception as e:
        print(f"Error plotting {file_path}: {e}")
        return False

def main():
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("No data to plot.")
        return

    print(f"Starting plot generation for {len(files)} files...")
    count = 0
    for f in sorted(files):
        if generate_plot(f):
            count += 1
            if count % 10 == 0:
                print(f"Generated {count} plots...")

    print(f"Finished. {count} plots saved to '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
