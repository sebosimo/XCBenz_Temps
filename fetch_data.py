import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def is_run_complete_locally(time_tag, locations, max_h):
    """Checks if the last file of a run exists locally."""
    last_loc = list(locations.keys())[-1]
    check_file = os.path.join(CACHE_DIR, f"{last_loc}_{time_tag}_H{max_h:02d}.nc")
    return os.path.exists(check_file)

def main():
    if not os.path.exists("locations.json"): return
    with open("locations.json", "r") as f: locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Target A (Latest) and Target B (Previous)
    targets = [latest_run, latest_run - datetime.timedelta(hours=3)]
    
    selected_run = None
    
    print("--- CHECKING FOR AVAILABLE RUNS ---")
    for run in targets:
        tag = run.strftime('%Y%m%d_%H%M')
        max_h = 45 if run.hour == 3 else 33
        
        # 1. Do we already have this run?
        if is_run_complete_locally(tag, locations, max_h):
            print(f"Run {tag}: ✅ Already fully cached.")
            # If we already have the latest target, we stop entirely
            if run == targets[0]:
                print("Everything is up to date.")
                return
            continue # If we have the old run but not the new one, keep checking the new one
            
        # 2. If we don't have it, is it ready on the server?
        try:
            req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                 reference_datetime=run, horizon="P0DT0H", perturbed=False)
            if ogd_api.get_from_ogd(req) is not None:
                print(f"Run {tag}: ✨ NEW DATA READY! Selecting this run.")
                selected_run = run
                break # We found the newest possible run that is ready to download
            else:
                print(f"Run {tag}: ❌ Files not found on server yet.")
        except (IndexError, Exception):
            print(f"Run {tag}: ❌ Server says 'not ready' (IndexError).")

    if not selected_run:
        print("RESULT: No new runs to download at this time.")
        return

    # --- START DOWNLOAD FOR SELECTED RUN ---
    ref_time = selected_run
    time_tag = ref_time.strftime('%Y%m%d_%H%M')
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    
    print(f"\n--- PROCESSING RUN: {time_tag} ---")
    cached_indices = None

    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        valid_time = ref_time + datetime.timedelta(hours=h_int)
        
        try:
            domain_fields = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                domain_fields[var] = ogd_api.get_from_ogd(req)
            
            if cached_indices is None:
                sample = domain_fields["T"]
                lat_n = 'latitude' if 'latitude' in sample.coords else 'lat'
                lon_n = 'longitude' if 'longitude' in sample.coords else 'lon'
                lats, lons = sample[lat_n].values, sample[lon_n].values
                cached_indices = {n: int(np.argmin((lats-c['lat'])**2+(lons-c['lon'])**2)) for n, c in locations.items()}

            for name, flat_idx in cached_indices.items():
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")
                if os.path.exists(cache_path): continue

                loc_vars = {}
                for var_name, ds_full in domain_fields.items():
                    lat_coord = 'latitude' if 'latitude' in ds_full.coords else 'lat'
                    spatial_dim = ds_full[lat_coord].dims[0]
                    profile = ds_full.squeeze().isel({spatial_dim: flat_idx}).compute()
                    loc_vars[var_name] = profile.drop_vars([c for c in profile.coords if c not in profile.dims])

                ds_final = xr.Dataset(loc_vars)
                ds_final.attrs = {"location": name, "HUM_TYPE": "QV", "ref_time": ref_time.isoformat(), 
                                 "horizon_h": h_int, "valid_time": valid_time.isoformat()}
                
                for v in ds_final.data_vars: ds_final[v].attrs = {}
                ds_final.to_netcdf(cache_path)
            
            print(f"  [OK] Horizon +{h_int}h")

        except Exception as e:
            print(f"  [WAIT] Horizon +{h_int}h not ready yet. Stopping here to resume later.")
            break 

if __name__ == "__main__":
    main()
