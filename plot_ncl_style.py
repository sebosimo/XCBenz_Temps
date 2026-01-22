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
    
    # Vectors: Streamlines with Variable Density
    # Strategy: Seed points based on wind speed.
    # Manually project grid and vectors to Mercator for Streamplot
    # This avoids issues where start_points (in PlateCarree) are not correctly transformed by Cartopy
    proj = ccrs.Mercator()
    
    # Project grid points
    pts_proj = proj.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)
    
    # Extract 1D axes for streamplot (rectilinear grid)
    # Mercator projection of a Lat/Lon regular grid results in a rectilinear grid (separable x and y)
    x_grid_1d = pts_proj[0, :, 0] # Shape (N_lon,)
    y_grid_1d = pts_proj[:, 0, 1] # Shape (N_lat,)
    
    # Full 2D grid for seed selection
    x_grid_2d = pts_proj[:,:,0]
    y_grid_2d = pts_proj[:,:,1]
    
    # Project vectors
    u_proj, v_proj = proj.transform_vectors(ccrs.PlateCarree(), lon_grid, lat_grid, u_grid, v_grid)
    
    # Calculate speed from projected vectors for linewidth (should be similar magnitude)
    speed_proj = np.sqrt(u_proj**2 + v_proj**2) * 3.6 # km/h
    
    # Seed generation based on speed
    s_norm = speed_proj / 80.0
    s_norm = np.clip(s_norm, 0, 1)
    base_prob = 0.08
    seed_mask = np.random.rand(*speed_proj.shape) < (s_norm * base_prob + 0.005)

    # Filter seeds to be strictly inside the projected domain
    # NaN Check
    if np.isnan(x_grid_1d).any() or np.isnan(y_grid_1d).any():
        print("WARNING: NaNs detected in projected grid!")
    
    # Use nanmin/nanmax
    eps = 2000.0 
    x_min, x_max = np.nanmin(x_grid_1d) + eps, np.nanmax(x_grid_1d) - eps
    y_min, y_max = np.nanmin(y_grid_1d) + eps, np.nanmax(y_grid_1d) - eps
    
    # TEST: Use only ONE seed at the center to verify connectivity
    # xc = x_grid_1d[len(x_grid_1d)//2]
    # yc = y_grid_1d[len(y_grid_1d)//2]
    # start_points = np.array([[xc, yc]])
    
    # Full logic with robust filter
    pts_x = x_grid_2d[seed_mask]
    pts_y = y_grid_2d[seed_mask]
    
    valid_seeds = (pts_x > x_min) & (pts_x < x_max) & \
                  (pts_y > y_min) & (pts_y < y_max)
                  
    start_points = np.column_stack((pts_x[valid_seeds], pts_y[valid_seeds]))
    
    # Debug info
    print(f"DEBUG: Grid X: {np.nanmin(x_grid_1d):.1f} - {np.nanmax(x_grid_1d):.1f} (Sorted? {np.all(np.diff(x_grid_1d) > 0)})")
    print(f"DEBUG: Grid Y: {np.nanmin(y_grid_1d):.1f} - {np.nanmax(y_grid_1d):.1f} (Sorted? {np.all(np.diff(y_grid_1d) > 0)})")

    # Convert to list of tuples to avoid potential numpy array issues
    start_points_list = [tuple(p) for p in start_points]
    
    # Linewidth
    lw = 0.4 + 1.2 * (speed_proj / 100.0)
    lw = np.clip(lw, 0.4, 1.5)
    
    try:
        # Plot streamplot in Mercator coordinates
        # remove transform=proj to treat inputs as native data coordinates
        ax.streamplot(x_grid_1d, y_grid_1d, u_proj, v_proj, 
                  transform=None, 
                  color='black', 
                  linewidth=lw, 
                  arrowsize=0.7,
                  arrowstyle='->',
                  density=30, 
                  start_points=start_points_list,
                  maxlength=20.0)
    except Exception as e:
        print(f"Streamplot Failed: {e}")
        # Fallback with higher uniform density
        print("Using fallback streamplot with uniform density=4")
        ax.streamplot(x_grid_1d, y_grid_1d, u_proj, v_proj, 
                  transform=None,
                  color='black', 
                  linewidth=lw,
                  density=4)
 # Limit integration length if possible (in data coords?) check compat
              # maxlength not always supported in older MPL. If error, remove.
              # But user asked for "length", streamplot integration stops.
    
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
                            lefttitle=f"{title_left}\nWind on {level_name}",
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
        
        # User request: Only 800m
        if "800m" not in level_name:
             continue
             
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
    search_path = os.path.join(CACHE_DIR, "**", "*.nc")
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("No .nc files found in cache_wind.")
        return

    files.sort()
    
    for f in files:
        process_timestep(f)


if __name__ == "__main__":
    main()
