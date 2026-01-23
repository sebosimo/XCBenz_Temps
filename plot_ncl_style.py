import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import geocat.viz as gv
# from geocat.viz import cmaps as gvcmaps # Failed
import matplotlib.colors as mcolors

# Define NCL-like colormap manually (approximate: White-Violet-Blue-Green-Yellow-Orange-Red)
# Bins based on reference: 0-4, 4-6, ...
bounds = [0, 4, 6, 10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

# Colors approx matching the bins (17 colors for 17 intervals)
hex_colors = [
    "#FFFFFF", # 0-4 (White)
    "#E0F0FF", # 4-6 (Very Light Blue)
    "#B0D0FF", # 6-10 (Light Blue)
    "#80B0FF", # 10-14 (Blue)
    "#4090FF", # 14-18 (Strong Blue)
    "#0060FF", # 18-22 (Deep Blue)
    "#008000", # 22-26 (Green)
    "#60B000", # 26-30 (Lighter Green)
    "#A0D000", # 30-35 (Yellow-Green)
    "#FFFF00", # 35-40 (Yellow)
    "#FFC000", # 40-45 (Orange-Yellow)
    "#FFA000", # 45-50 (Orange)
    "#FF6000", # 50-60 (Red-Orange)
    "#FF0000", # 60-70 (Red)
    "#C00000", # 70-80 (Dark Red)
    "#800040", # 80-90 (Purple-Red)
    "#600060", # 90-100 (Purple)
    "#600060", # > 100 (Purple - Extension)
]

try:
    NCL_CMAP = mcolors.ListedColormap(hex_colors)
    NCL_CMAP.set_over("#600060") # Explicitly set over (redundant but safe)
except:
    NCL_CMAP = 'turbo'

from scipy.interpolate import griddata
import os
import glob
from datetime import datetime, timedelta
import re

# Configuration
CACHE_DIR = "cache_wind"
OUTPUT_DIR = "map_artifacts/ncl_style"
GRID_RES = 0.02 # Resolution for regridding in degrees

def get_wind_data(nc_path):
    """Loads wind data and returns unstructured arrays."""
    try:
        ds = xr.open_dataset(nc_path)
        data_levels = []
        
        for v in ds.data_vars:
            if v.startswith("u_"):
                level_suffix = v[2:] # e.g. "800m_AGL"
                u_name = v
                v_name = f"v_{level_suffix}"
                
                if v_name in ds:
                    data_levels.append({
                        'level_name': level_suffix,
                        'u': ds[u_name],
                        'v': ds[v_name]
                    })
        return data_levels, ds
    except Exception as e:
        print(f"Error loading {nc_path}: {e}")
        return [], None

def regrid_data(u, v, lat_in, lon_in):
    """Regrids unstructured data to a regular grid."""
    
    # Define target grid based on domain of interest (Switzerland+)
    # Bounds: 5.5E - 11.0E, 45.5N - 48.2N
    grid_lon_1d = np.arange(5.5, 11.0, GRID_RES)
    grid_lat_1d = np.arange(45.5, 48.2, GRID_RES)
    grid_lon, grid_lat = np.meshgrid(grid_lon_1d, grid_lat_1d)
    
    # Flatten input coords
    # OPTIMIZATION: Stride the input data to speed up regridding
    stride = 5
    
    lon_in_s = lon_in[::stride]
    lat_in_s = lat_in[::stride]
    points = np.column_stack((lon_in_s, lat_in_s))
    
    print(f"  Regridding {len(points)} points (stride={stride}) to {grid_lon.shape} grid...", flush=True)
    
    # Interpolate U and V
    u_vals = u.values if hasattr(u, 'values') else u
    v_vals = v.values if hasattr(v, 'values') else v
    
    u_vals_s = u_vals[::stride]
    v_vals_s = v_vals[::stride]
    
    u_grid = griddata(points, u_vals_s, (grid_lon, grid_lat), method='linear')
    v_grid = griddata(points, v_vals_s, (grid_lon, grid_lat), method='linear')
    
    return grid_lon, grid_lat, u_grid, v_grid

def plot_single_level(u_grid, v_grid, lon_grid, lat_grid, level_name, time_str, out_path):
    """Plots a single level using NCL style."""
    
    # Calculate speed in km/h
    speed_ms = np.sqrt(u_grid**2 + v_grid**2)
    speed_kmh = speed_ms * 3.6
    
    fig = plt.figure(figsize=(15, 10)) # Wider aspect ratio
    
    # Use Mercator for correct shape
    ax = plt.axes(projection=ccrs.Mercator())
    
    # Set Extent (Switzerland)
    ax.set_extent([5.5, 11.0, 45.5, 48.0], crs=ccrs.PlateCarree())
    
    # Add Map Features
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor='none', linewidth=0.5)

    # Colormap
    cmap = NCL_CMAP
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='max')

    # Contourf (Speed)
    cf = ax.contourf(lon_grid, lat_grid, speed_kmh, 
                     levels=bounds, 
                     norm=norm,
                     cmap=cmap, 
                     transform=ccrs.PlateCarree(), 
                     extend='max')
    
    # --- Custom Trajectory Integration (Lagrangian) ---
    # User Request: "Length proportional to speed", "Arrow at end", "1.5x density"
    
    # 0. Project Grid to Mercator (for meters)
    proj = ccrs.Mercator()
    pts_proj = proj.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)
    
    # Extract 1D axes (rectilinear grid)
    x_grid_1d = pts_proj[0, :, 0] # Lon varies along axis 1 (columns) -> Row 0, All Cols
    y_grid_1d = pts_proj[:, 0, 1] # Lat varies along axis 0 (rows) -> All Rows, Col 0
    
    # Project vectors
    u_proj, v_proj = proj.transform_vectors(ccrs.PlateCarree(), lon_grid, lat_grid, u_grid, v_grid)
    
    # 1. Prepare Interpolator (RegularGridInterpolator expects (y, x) order)
    # x_grid_1d, y_grid_1d come from Mercator projection of 1D lat/lon.
    # Note: RegularGridInterpolator requires grid points to be strictly increasing.
    # We checked this in debug loops previously.
    
    from scipy.interpolate import RegularGridInterpolator
    from matplotlib.collections import LineCollection
    import matplotlib.markers as mmarkers
    
    # Interpolators for u_proj, v_proj
    # grid is (lat(y), lon(x)) usually, but here projected y, x.
    # Check shape: u_grid is (lat, lon) -> (y, x).
    interp_u = RegularGridInterpolator((y_grid_1d, x_grid_1d), u_proj, bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((y_grid_1d, x_grid_1d), v_proj, bounds_error=False, fill_value=0)
    
    # 2. Generate Seeds
    # Increase density: Stride of 3 or 4 pixels?
    # Grid is 136x275. 
    # Stride 4 -> ~34x68 = 2300 seeds.
    # Stride 3 -> ~45x90 = 4000 seeds.
    # Let's try Stride 4 first, then adjust. User wants 1.5x "lines".
    # Previous streamplot density=30 is opaque.
    stride = 3 
    
    # Use meshgrid for seeds
    # Jitter seeds slightly to avoid regular grid artifacts
    seed_x_1d = x_grid_1d[::stride]
    seed_y_1d = y_grid_1d[::stride]
    seed_x, seed_y = np.meshgrid(seed_x_1d, seed_y_1d)
    seed_pts = np.column_stack((seed_x.ravel(), seed_y.ravel()))
    
    # Add random jitter (half stride)
    dx = (x_grid_1d[1] - x_grid_1d[0])
    dy = (y_grid_1d[1] - y_grid_1d[0])
    jitter = np.random.uniform(-0.5, 0.5, seed_pts.shape) * [dx*stride, dy*stride]
    seed_pts += jitter
    
    # 3. Integrate Trajectories
    # Fixed Integration Time T.
    # Enforcing global consistency: Same speed = Same length across all maps.
    dt = 3600.0 # 1 hour integration
    
    n_steps = 10 # Smoother curves
    h = dt / n_steps # Time step per segment
    
    # Vectorized Integration (Euler or RK2)
    # Current positions
    xy = seed_pts.copy() # (N, 2)
    
    # Store segments: (N_seeds, n_steps+1, 2)
    trajs = np.zeros((len(xy), n_steps + 1, 2))
    trajs[:, 0, :] = xy
    
    for i in range(n_steps):
        # Current coords (y, x) for interpolator
        # Interpolator wants (y, x)
        pts_for_interp = xy[:, ::-1] # Swap to (y, x)
        
        u_local = interp_u(pts_for_interp)
        v_local = interp_v(pts_for_interp)
        
        # Simple Euler: new = old + v * h
        xy[:, 0] += u_local * h
        xy[:, 1] += v_local * h
        
        trajs[:, i+1, :] = xy
        
    # 4. Plotting
    # Filter out static points (speed ~ 0) to reduce clutter?
    # Compute total displacement
    disp = np.linalg.norm(trajs[:, -1, :] - trajs[:, 0, :], axis=1)
    min_disp = 1000.0 # 1km minimum length
    valid_mask = disp > min_disp
    
    trajs = trajs[valid_mask]
    
    # Create LineCollection
    # variable linewidth based on speed at NEW seed location (approx)
    # Re-evaluate speed at start
    pts_start = trajs[:, 0, ::-1] # y,x
    u_start = interp_u(pts_start)
    v_start = interp_v(pts_start)
    speed_start = np.sqrt(u_start**2 + v_start**2) * 3.6 # km/h
    
    lw = 0.4 + 1.2 * (speed_start / 100.0)
    lw = np.clip(lw, 0.4, 1.5)
    
    # Create segments for LineCollection: List of (N_points, 2)
    segments = [trajs[i] for i in range(len(trajs))]
    
    lc = LineCollection(segments, colors='black', linewidths=lw, transform=proj, capstyle='round')
    ax.add_collection(lc)
    
    # 5. Add Arrows at END of trajectories
    end_pts = trajs[:, -1, :]
    prev_pts = trajs[:, -2, :] # Use second to last to determine direction
    
    arrow_dirs = end_pts - prev_pts
    
    # Arrow size scaling? Fixed size usually better for aesthetics.
    # Quiver at end points.
    # U, V for quiver is delta_x, delta_y
    # User feedback: Arrows too big on 10m lines.
    # Reduced width from 0.0015 to 0.0008 to match thinner lines better.
    
    ax.quiver(end_pts[:, 0], end_pts[:, 1], arrow_dirs[:, 0], arrow_dirs[:, 1],
              transform=proj,
              color='black',
              scale=None, # Auto scale?
              angles='xy', scale_units='xy', # Use vectors as displacement
              width=0.0008, 
              headwidth=4,
              headlength=5,
              headaxislength=4.5,
              pivot='tip') # Draw at TIP (End of line)

    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.9, aspect=25, drawedges=True)
    cbar.ax.tick_params(labelsize=9, direction='in') # Ticks inside
    cbar.set_label("km/h", rotation=0, labelpad=15, y=1.02)
    
    # Titles
    try:
        dt_init = datetime.strptime(time_str, "%Y%m%d_%H%M") # Logic: Folder name is init time?
        # Actually in 'process_timestep', tag is folder name which IS init time (20260120_1800).
        
        # Calculate valid time
        match = re.search(r"_H(\d+)", out_path) # Changed regex to match filename structure more vaguely
        if match:
            lead_hours = int(match.group(1))
        else:
            lead_hours = 0
            
        valid_dt = dt_init + timedelta(hours=lead_hours)
        local_dt = valid_dt + timedelta(hours=1) # CET
        
        # Labels
        # Top Left: Model Run
        title_left = f"Model Run: {dt_init.strftime('%d.%m.%Y %HUTC')}"
        
        # Top Right: Forecast/Valid
        # Line 1: Forecast (Time + Lead)
        # Line 2: Local Time
        title_right = f"Valid: {valid_dt.strftime('%a %d %b %HUTC')} (+{lead_hours}h)\nLocal: {local_dt.strftime('%a %H:%M CET')}"

    except Exception as e:
        print(f"Title logic error: {e}")
        title_left = f"Run: {time_str}"
        title_right = ""

    gv.set_titles_and_labels(ax,
                            maintitle="",
                            lefttitle=f"{title_left}\nWind on {level_name.replace('_', ' ')}",
                            lefttitlefontsize=12,
                            righttitle=title_right,
                            righttitlefontsize=12, 
                            xlabel="",
                            ylabel="")

    # Remove Bottom Copyright (User requested removal)
    # ax.text(...) # Removed
    
    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

def process_timestep(nc_path):
    print(f"Processing {nc_path}...")
    tag_dir = os.path.dirname(nc_path)
    tag = os.path.basename(tag_dir) 
    filename = os.path.basename(nc_path)
    
    data_levels, ds = get_wind_data(nc_path)
    if not data_levels:
        return

    try:
        if 'latitude' in ds.coords:
            lat = ds.latitude.values
            lon = ds.longitude.values
        elif 'lat' in ds.coords:
            lat = ds.lat.values
            lon = ds.lon.values
        else:
             if 'clat' in ds.coords:
                 lat = np.degrees(ds.clat.values)
                 lon = np.degrees(ds.clon.values)
             else:
                 print("Could not find lat/lon coordinates.")
                 return
    except Exception as e:
        print(f"Coord error: {e}")
        return

    for item in data_levels:
        level_name = item['level_name']
        
        # User request: All altitude levels
             
        print(f"  Level: {level_name}")
        
        u = item['u']
        v = item['v']
        
        # Regrid
        grid_lon, grid_lat, u_grid, v_grid = regrid_data(u, v, lat, lon)
        
        # Plot
        h_str = filename.split("_")[-1].replace(".nc", "")
        out_name = f"wind_ncl_{level_name}_{tag}_{h_str}.png"
        out_path = os.path.join(OUTPUT_DIR, tag, out_name)
        
        plot_single_level(u_grid, v_grid, grid_lon, grid_lat, level_name, tag, out_path)

def main():
    # Find latest run directory
    # CACHE_DIR structure: cache_wind/YYYYMMDD_HHMM/*.nc
    
    # Get all subdirectories
    subdirs = [d for d in glob.glob(os.path.join(CACHE_DIR, "*")) if os.path.isdir(d)]
    
    if not subdirs:
        print("No run directories found in cache_wind.")
        return
        
    # Sort by name (timestamp)
    latest_run = max(subdirs, key=os.path.getmtime) # Or just string sort if names are consistent format
    # Using sorted() on names is safer for YYYYMMDD_HHMM format
    latest_run = sorted(subdirs)[-1]
    
    print(f"Processing latest run: {latest_run}")
    
    search_path = os.path.join(latest_run, "*.nc")
    files = glob.glob(search_path)
    
    if not files:
        print(f"No .nc files found in {latest_run}.")
        return

    files.sort()
    
    for f in files:
        process_timestep(f)


if __name__ == "__main__":
    main()
