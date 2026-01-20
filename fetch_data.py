import os, sys, datetime, json, xarray as xr
import numpy as np
import warnings
from meteodatalab import ogd_api

# Suppress warnings
warnings.filterwarnings("ignore")

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

def is_run_complete_locally(time_tag, locations, max_h):
    """Checks if the very last file of a run exists."""
    last_loc = list(locations.keys())[-1]
    check_trace = os.path.join(CACHE_DIR_TRACES, time_tag, last_loc, f"H{max_h:02d}.nc")
    check_map = os.path.join(CACHE_DIR_MAPS, time_tag, f"wind_maps_H{max_h:02d}.nc")
    return os.path.exists(check_trace) and os.path.exists(check_map)

def process_traces(domain_fields, locations, time_tag, h_int, ref_time):
    """Extracts point data for specific locations."""
    sample = list(domain_fields.values())[0]
    lat_n = 'latitude' if 'latitude' in sample.coords else 'lat'
    lon_n = 'longitude' if 'longitude' in sample.coords else 'lon'
    lats, lons = sample[lat_n].values, sample[lon_n].values
    
    indices = {n: int(np.argmin((lats-c['lat'])**2+(lons-c['lon'])**2)) for n, c in locations.items()}

    for name, flat_idx in indices.items():
        loc_dir = os.path.join(CACHE_DIR_TRACES, time_tag, name)
        os.makedirs(loc_dir, exist_ok=True)
        cache_path = os.path.join(loc_dir, f"H{h_int:02d}.nc")
        
        if os.path.exists(cache_path): continue

        loc_vars = {}
        for var_name in VARS_TRACES:
            if var_name not in domain_fields: continue
            ds = domain_fields[var_name]
            s_dim = ds[lat_n].dims[0]
            profile = ds.squeeze().isel({s_dim: flat_idx}).compute()
            
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
    """Interpolates wind and saves map NC."""
    output_dir = os.path.join(CACHE_DIR_MAPS, time_tag)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wind_maps_H{h_int:02d}.nc")
    if os.path.exists(output_path): return

    if "U" not in domain_fields or "V" not in domain_fields:
        return # Cannot process
    if "HHL" not in domain_fields:
        print(f" Warning: HHL missing for H{h_int:02d}. Skipping maps.")
        return

    # Imports locally to avoid top-level require
    from metpy.interpolate import interpolate_to_isosurface
    from metpy.units import units

    u = domain_fields["U"].squeeze()
    v = domain_fields["V"].squeeze()
    hhl = domain_fields["HHL"].squeeze()
    
    # Calculate HFL (Height Full Levels)
    # HHL likely (81, ncells), U/V (80, ncells)
    try:
        # Check dim name consistency
        z_dim = hhl.dims[0]
        # Slice logic: Depends on order. 
        # Safest: Use 0:-1 and 1:None manually
        z_f = (hhl.isel({z_dim: slice(0,-1)}).values + hhl.isel({z_dim: slice(1,None)}).values) / 2
        
        h_surf = hhl.isel({z_dim: -1}) # Assuming last is surface
    except Exception as e:
        print(f" HHL processing error: {e}")
        return

    out_ds_dict = {}
    
    # Wrap in DataArray for MetPy (Needs matching dims to U/V)
    u_dims = u.dims
    z_da = xr.DataArray(z_f, coords=u.coords, dims=u_dims)

    for lvl in WIND_LEVELS:
        h_target = lvl['h']
        mode = lvl['type']
        name_key = lvl['name']
        
        try:
            if mode == 'AGL':
                 field_z = z_da - h_surf
            else: # AMSL
                 field_z = z_da
            
            res_u = interpolate_to_isosurface(field_z, u, h_target)
            res_v = interpolate_to_isosurface(field_z, v, h_target)
            
            out_ds_dict[f"u_{name_key}"] = res_u
            out_ds_dict[f"v_{name_key}"] = res_v
        except Exception as e:
            pass # Skip level

    if out_ds_dict:
        ds_out = xr.Dataset(out_ds_dict)
        for v_c in ds_out:
            if hasattr(ds_out[v_c].data, 'magnitude'):
                ds_out[v_c] = ds_out[v_c].metpy.dequantify()
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
    
    selected_run = None
    print("--- CHECKING FOR AVAILABLE RUNS ---")
    
    for run in targets:
        tag = run.strftime('%Y%m%d_%H%M')
        max_h = 45 if run.hour == 3 else 33
        
        # 1. Check Local Complete
        if is_run_complete_locally(tag, locations, max_h):
             print(f"Run {tag}: ✅ Already fully cached.")
             if run == targets[0]: return 
             continue
        
        # 2. Check Server Availability
        try:
             req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                 reference_datetime=run, horizon="P0DT0H", perturbed=False)
             if ogd_api.get_from_ogd(req) is not None:
                 print(f"Run {tag}: ✨ NEW DATA READY!")
                 selected_run = run
                 break
        except Exception:
             print(f"Run {tag}: ❌ Server says 'not ready'.")
             pass

    if not selected_run:
        print("RESULT: No new runs to download.")
        # Attempt cleanup even if no new run, to keep things tidy
        cleanup_old_runs()
        return

    ref_time = selected_run
    time_tag = ref_time.strftime('%Y%m%d_%H%M')
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 1)
    
    # HHL State - Lazy Load
    hhl_field = None

    print(f"\n--- PROCESSING RUN: {time_tag} ---")
    
    vars_all = list(set(VARS_TRACES + VARS_MAPS))
    
    # We remove HHL from the loop, as we have it (or not)
    vars_to_fetch = [v for v in vars_all if v != "HHL"]
    
    for h_int in horizons:
        # Check if needed
        traces_needed = False
        last_loc = list(locations.keys())[-1]
        if not os.path.exists(os.path.join(CACHE_DIR_TRACES, time_tag, last_loc, f"H{h_int:02d}.nc")):
             traces_needed = True
        
        maps_needed = False
        if not os.path.exists(os.path.join(CACHE_DIR_MAPS, time_tag, f"wind_maps_H{h_int:02d}.nc")):
             maps_needed = True

        if not traces_needed and not maps_needed: 
             continue

        print(f"Fetching +{h_int:02d}h...", end=" ", flush=True)
        iso_h = get_iso_horizon(h_int)
        
        domain_fields = {}
        
        # Try to fetch HHL if missing and needed for maps
        if maps_needed and hhl_field is None:
             try:
                # Try fetching HHL with current horizon
                req_hhl = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="HHL",
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                res_hhl = ogd_api.get_from_ogd(req_hhl)
                if res_hhl is not None:
                    hhl_field = res_hhl
                    # print("(Captured HHL)", end=" ")
             except Exception:
                pass

        # Inject HHL if available
        if hhl_field is not None:
            domain_fields["HHL"] = hhl_field

        for var in vars_to_fetch:
            # Print separate progress dot potentially or handle failure
            try:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                res = ogd_api.get_from_ogd(req)
                if res is not None:
                    domain_fields[var] = res
            except Exception as e:
                # print(f"({var} err)", end="")
                pass
        
        if domain_fields:
            # Check if we have enough for Traces
            # Traces need VARS_TRACES. HHL is NOT in VARS_TRACES usually (only for maps)
            # VARS_TRACES = ["T", "U", "V", "P", "QV"]
            
            if traces_needed:
                 if any(v in domain_fields for v in VARS_TRACES):
                     process_traces(domain_fields, locations, time_tag, h_int, ref_time)
            
            if maps_needed:
                 # Check HHL/Wind
                 if hhl_field is not None and "U" in domain_fields and "V" in domain_fields:
                     process_wind_maps(domain_fields, time_tag, h_int, ref_time)
            
            print("Done")
        else:
            print("Failed (No data)")

    # Cleanup after processing
    cleanup_old_runs()

def cleanup_old_runs():
    """
    Removes runs older than RETENTION_DAYS (env var).
    If RETENTION_DAYS is not set, NO cleanup is performed (local default).
    """
    try:
        days_str = os.environ.get("RETENTION_DAYS")
        if not days_str:
            # If not running in CI (or no env var), we might want a default or just return.
            # For now, let's look for a local default or return to avoid deleting user data unexpectedly.
            return 
        
        days_to_keep = int(days_str)
    except ValueError:
        print(f"Warning: Invalid RETENTION_DAYS '{days_str}', skipping cleanup.")
        return

    print(f"--- CLEANING UP OLD DATA (Retention: {days_to_keep} days) ---")
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    
    # We look at folder names YYYYMMDD_HHMM
    dirs_to_check = [CACHE_DIR_TRACES, CACHE_DIR_MAPS]
    
    for d in dirs_to_check:
        if not os.path.exists(d): continue
        
        for item in os.listdir(d):
            path = os.path.join(d, item)
            
            # 1. Handle Directories (Runs)
            if os.path.isdir(path):
                # Parse timestamp from folder name
                try:
                    # Expected format: YYYYMMDD_HHMM
                    dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M")
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                    
                    if dt < cutoff:
                        print(f"Deleting old run: {path}")
                        import shutil
                        shutil.rmtree(path)
                except ValueError:
                    # Not a timestamped folder, skip
                    pass
            
            # 2. Handle Orphaned Files (e.g. invalid downloads, old logs)
            elif os.path.isfile(path):
                 # Check modification time
                 mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path), tz=datetime.timezone.utc)
                 if mtime < cutoff:
                     print(f"Deleting old orphaned file: {path}")
                     try:
                        os.remove(path)
                     except Exception as e:
                        print(f"Failed to remove {path}: {e}")

if __name__ == "__main__":
    main()
