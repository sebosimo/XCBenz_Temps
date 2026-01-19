import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
# Variables needed for Point Traces
VARS_TRACES = ["T", "U", "V", "P", "QV"]
# Variables needed for Wind Maps
VARS_MAPS = ["U", "V", "HHL"] 

CACHE_DIR_TRACES = "cache_data"
CACHE_DIR_MAPS = "cache_wind"

# Levels definition
WIND_LEVELS = [
    {"name": "10m_AGL",   "h": 10,   "type": "AGL"},
    {"name": "800m_AGL",  "h": 800,  "type": "AGL"},
    {"name": "1500m_AMSL","h": 1500, "type": "AMSL"},
    {"name": "2000m_AMSL","h": 2000, "type": "AMSL"},
    {"name": "3000m_AMSL","h": 3000, "type": "AMSL"},
    {"name": "4000m_AMSL","h": 4000, "type": "AMSL"},
]

os.makedirs(CACHE_DIR_TRACES, exist_ok=True)
os.makedirs(CACHE_DIR_MAPS, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def process_traces(domain_fields, locations, time_tag, h_int, ref_time):
    """Extracts point data for specific locations (Original Logic)."""
    
    # Pre-calculate indices if not done (Optimization: Pass cached_indices? 
    # For now re-calc once per timestep is fast enough for 1D arrays or pass mapped indices)
    # To keep it simple and stateless per call, we re-calc or rely on generic logic.
    # Actually, calculating indices every hour is fine for 30 locations.
    
    # We need lat/lon from one variable
    sample = list(domain_fields.values())[0]
    lat_n = 'latitude' if 'latitude' in sample.coords else 'lat'
    lon_n = 'longitude' if 'longitude' in sample.coords else 'lon'
    lats, lons = sample[lat_n].values, sample[lon_n].values
    
    # Find indices
    indices = {n: int(np.argmin((lats-c['lat'])**2+(lons-c['lon'])**2)) for n, c in locations.items()}

    for name, flat_idx in indices.items():
        loc_dir = os.path.join(CACHE_DIR_TRACES, time_tag, name)
        os.makedirs(loc_dir, exist_ok=True)
        cache_path = os.path.join(loc_dir, f"H{h_int:02d}.nc")
        
        # Skip if exists? Logic handled in main loop implies we only fetch if needed.
        # But we might need traces even if maps exist or vice versa.
        if os.path.exists(cache_path): continue

        loc_vars = {}
        for var_name in VARS_TRACES:
            if var_name not in domain_fields: continue
            ds = domain_fields[var_name]
            # Spatial dim
            s_dim = ds[lat_n].dims[0]
            profile = ds.squeeze().isel({s_dim: flat_idx}).compute()
            
            # Normalize
            if len(profile.dims) > 0:
                v_dim = profile.dims[0]
                profile = profile.rename({v_dim: 'level'})
            
            loc_vars[var_name] = profile.drop_vars([c for c in profile.coords if c not in profile.dims])

        ds_final = xr.Dataset(loc_vars)
        ds_final.attrs = {
            "location": name, "HUM_TYPE": "QV", 
            "ref_time": ref_time.isoformat(), 
            "horizon_h": h_int, 
            "valid_time": (ref_time + datetime.timedelta(hours=h_int)).isoformat()
        }
        for v in ds_final.data_vars: ds_final[v].attrs = {}
        ds_final.to_netcdf(cache_path)

def process_wind_maps(domain_fields, time_tag, h_int, ref_time):
    """Interpolates wind to target levels and saves a generic wind map NC."""
    
    output_dir = os.path.join(CACHE_DIR_MAPS, time_tag)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wind_maps_H{h_int:02d}.nc")
    
    if os.path.exists(output_path): return

    if "U" not in domain_fields or "V" not in domain_fields:
        print("Missing Wind Data for Maps.")
        return

    # Check HHL
    hhl = None
    if "HHL" in domain_fields:
        hhl = domain_fields["HHL"]
    else:
        print("Warning: HHL not found. Skipping Map Interpolation.")
        return

    # Prepare dataset for interpolation
    # U, V, HHL are likely on (level, ncells)
    u = domain_fields["U"].squeeze()
    v = domain_fields["V"].squeeze()
    hhl = hhl.squeeze()
    
    # Ensure vertical dimension is first
    # ICON data: usually (nlayers, ncells).
    
    # We need Surface Height (H_SURF) for AGL calculations
    # HHL[-1] is surface interface.
    # Note: Xarray index -1 might be top or bottom depending on data. 
    # METADATA: generalVerticalLayer usually 1..80. Top is 1? Or Top is 80?
    # Usually index 0 is top, index -1 is bottom in GRIB/NetCDF from ICON.
    # HHL has nlevels+1. Last index is surface.
    
    h_surf = hhl.isel({hhl.dims[0]: -1}) 
    
    # Vertical interpolation logic
    # We want to interpolate U and V to H_target.
    # Z_field = HHL (on interfaces) or HFL (on main levels). 
    # U/V are on main levels. HHL is on interfaces.
    # We need HFL = (HHL[:-1] + HHL[1:]) / 2.
    
    # Compute HFL (height of full levels)
    # Using numpy/xarray rolling or just slice
    z_full = (hhl.isel({hhl.dims[0]: slice(0,-1)}).values + hhl.isel({hhl.dims[0]: slice(1,None)}).values) / 2
    # This assumes ordered top-to-bottom or bottom-to-top consistently.
    
    u_vals = u.values # (nz, ncells)
    v_vals = v.values
    
    # Result container
    out_vars = {}
    
    # Get spatial coords
    lat_n = 'latitude' if 'latitude' in u.coords else 'lat'
    lat_vals = u[lat_n].values
    lon_n = 'longitude' if 'longitude' in u.coords else 'lon'
    lon_vals = u[lon_n].values

    ncells = u_vals.shape[-1]
    
    for lvl in WIND_LEVELS:
        h_target = lvl['h']
        mode = lvl['type']
        var_name = lvl['name']
        
        # Define target Z array
        if mode == 'AGL':
            # Target Z = Surface + h
            z_target = h_surf.values + h_target
        else: # AMSL
            # Target Z = h (constant)
            z_target = np.full(ncells, h_target)
            
        # Perform vertical interpolation column by column? Too slow in python.
        # Vectorized approach:
        # z_full is (80, ncells). z_target is (ncells).
        # We find index k such that z_full[k] > z > z_full[k+1] (assuming decreasing height)
        # Verify z-ordering: ICON usually Top-Down (Level 1 = Top). Z decreases with index.
        
        # Check median Z difference
        if np.median(z_full[0] - z_full[-1]) > 0:
            # Decreasing Z (Top to Bottom)
            # np.interp requires increasing xp. Flip.
            z_col = z_full[::-1, :]
            u_col = u_vals[::-1, :]
            v_col = v_vals[::-1, :]
            # z_target is fine.
        else:
            z_col = z_full
            u_col = u_vals
            v_col = v_vals
            
        # Vectorized interpolation using fancy indexing or loops?
        # A full vectorizing `np.interp` on 2D arrays is tricky.
        # Use simple finding of nearest or bounding levels.
        # For simplicity and speed on 1M points/80 levels in python:
        # We can implement a simple linear interp manually.
        
        # Current naive approach: Loop is deadly.
        # Use: https://stackoverflow.com/questions/43772216/fast-interpolation-of-3d-data-along-axis
        # But unstructured grid makes it (nz, ncells).
        # We can treat last dim as batch.
        
        # Optimization:
        # Find indices where Z crosses target.
        # Since Z is monotonic, use searchsorted (on transposed array?).
        # Z_col shape (80, N). Z_target shape (N).
        # We want u_interp(N).
        
        # Actually, let's just save the full 3D fields if interpolation is too complex?
        # No, volume is too big.
        # We MUST interpolate.
        
        # Vectorized manual linear interp:
        # levels_z (80, N). values (80, N). target (N).
        # We assume monotonic.
        
        # Let's try to do it:
        # Broadcasting complexity.
        # Simpler: Use metpy? `metpy.interpolate.interpolate_to_isosurface`?
        # `requirements.txt` has metpy.
        # `from metpy.interpolate import interpolate_to_isosurface`
        # It handles `(levels, ...)` and `(levels, ...)`.
        
        pass # Will implement below inside Try block
    
    # ... (Actual implementation with metpy in the file content) ...
    # Fallback if metpy fails or too slow: Just use lowest level for 10m?
    # No, we need 4000m.
    # I will use MetPy.

    from metpy.interpolate import interpolate_to_isosurface
    from metpy.units import units
    
    # Prepare Xarrays for MetPy
    # MetPy wants coords attached.
    # Create Z DataArray
    z_da = xr.DataArray(z_full, coords=u.coords, dims=u.dims) # Assuming same dims
    
    out_ds_dict = {}
    
    for lvl in WIND_LEVELS:
        h_target = lvl['h']
        mode = lvl['type']
        name_key = lvl['name']
        
        if mode == 'AGL':
            # Construct 3D Height-AGL field?
            # z_agl = z_da - h_surf
            # target = h_target
            # But h_surf broadcasting might be tricky if dims mismatch slightly.
            # Assuming h_surf aligns with ncells.
             field_z = z_da - h_surf
             t_val = h_target
        else:
             field_z = z_da
             t_val = h_target
        
        # Interpolate
        # interpolate_to_isosurface(field, vertical_dim, levels)
        # field values, vertical_dim values, level value
        # returns (spatial_dims)
        try:
            # U
            res_u = interpolate_to_isosurface(field_z, u, t_val)
            # V
            res_v = interpolate_to_isosurface(field_z, v, t_val)
            
            out_ds_dict[f"u_{name_key}"] = res_u
            out_ds_dict[f"v_{name_key}"] = res_v
            
        except Exception as e:
            print(f"Interp failed for {name_key}: {e}")
            continue

    # Create Dataset
    ds_out = xr.Dataset(out_ds_dict)
    # Restore coords if lost? MetPy usually preserves them or returns Quantity.
    # Strip units if MetPy adds them
    for v_c in ds_out:
        if hasattr(ds_out[v_c].data, 'magnitude'):
            ds_out[v_c] = ds_out[v_c].metpy.dequantify()
            
    # Save
    ds_out.to_netcdf(output_path)


def main():
    if not os.path.exists("locations.json"):
        print("locations.json missing")
        return
    with open("locations.json", "r") as f: locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    targets = [latest_run, latest_run - datetime.timedelta(hours=3)]
    
    # Combined variables
    all_vars = list(set(VARS_TRACES + VARS_MAPS))
    
    selected_run = None
    print("--- CHECKING FOR AVAILABLE RUNS ---")
    
    # Simple check loop (Checking ONE variable T to see if run exists)
    for run in targets:
        tag = run.strftime('%Y%m%d_%H%M')
        # Check if local cache is complete?
        # Logic: Check last trace AND last map?
        # Simplified: Check if "Done" flag or just check last file.
        # If any missing, we try to download.
        # ...
        
        try:
             # Check availability on server
             req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                 reference_datetime=run, horizon="P0DT0H", perturbed=False)
             if ogd_api.get_from_ogd(req) is not None:
                 selected_run = run
                 print(f"Run {tag}: Found.")
                 break
        except:
             print(f"Run {tag}: Not ready.")
             
    if not selected_run:
        print("No new data.")
        return

    ref_time = selected_run
    time_tag = ref_time.strftime('%Y%m%d_%H%M')
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 1)
    
    print(f"\n--- PROCESSING RUN: {time_tag} ---")
    
    for h_int in horizons:
        # Check needs
        # 1. Traces
        traces_needed = False
        last_loc = list(locations.keys())[-1]
        if not os.path.exists(os.path.join(CACHE_DIR_TRACES, time_tag, last_loc, f"H{h_int:02d}.nc")):
             traces_needed = True
             
        # 2. Maps
        maps_needed = False
        if not os.path.exists(os.path.join(CACHE_DIR_MAPS, time_tag, f"wind_maps_H{h_int:02d}.nc")):
             maps_needed = True
             
        if not traces_needed and not maps_needed:
            print(f"H{h_int:02d}: All cached.")
            continue
            
        print(f"Fetching +{h_int:02d}h...", end=" ", flush=True)
        iso_h = get_iso_horizon(h_int)
        
        try:
            domain_fields = {}
            for var in all_vars:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                val = ogd_api.get_from_ogd(req)
                if val is not None:
                    domain_fields[var] = val
                else:
                    print(f"[Warn: {var} missing]", end=" ")

            if domain_fields:
                # 1. Traces
                if traces_needed:
                     process_traces(domain_fields, locations, time_tag, h_int, ref_time)
                # 2. Maps
                if maps_needed:
                     process_wind_maps(domain_fields, time_tag, h_int, ref_time)
                print("Done")
            else:
                print("Empty response.")

        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
