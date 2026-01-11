import os, sys, datetime, json, xarray as xr
import numpy as np
import traceback
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P"]
HUM_VARS = ["RELHUM", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    """
    Calculates the nearest grid indices for all locations at once.
    """
    try:
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        print(f"DIAGNOSTIC: Coordinate names found: lat='{lat_name}', lon='{lon_name}'")
        
        indices = {}
        grid_lat = ds[lat_name].values
        grid_lon = ds[lon_name].values
        
        print(f"DIAGNOSTIC: Grid shape: {grid_lat.shape}")
        
        for name, coords in locations.items():
            dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
            # unravel_index returns a tuple: (idx,) for 1D or (y, x) for 2D
            flat_min = np.argmin(dist)
            idx = np.unravel_index(flat_min, dist.shape)
            indices[name] = idx
        
        # Log the first location index as a sample
        sample_name = list(locations.keys())[0]
        print(f"DIAGNOSTIC: Sample index for {sample_name}: {indices[sample_name]} (type: {type(indices[sample_name])})")
        
        return indices
    except Exception as e:
        print(f"DIAGNOSTIC ERROR in get_location_indices: {e}")
        print(traceback.format_exc())
        raise e

def main():
    if not os.path.exists("locations.json"):
        print("Error: locations.json not found.")
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- ICON-CH1 Run: {time_tag} | Max Horizon: {max_h}h ---")

    cached_indices = None

    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        valid_time = ref_time + datetime.timedelta(hours=h_int)
        
        locations_to_process = [n for n in locations.keys() 
                               if not os.path.exists(os.path.join(CACHE_DIR, f"{n}_{time_tag}_H{h_int:02d}.nc"))]
        
        if not locations_to_process:
            continue

        print(f"\nHorizon +{h_int:02d}h: Fetching domain fields...")
        domain_fields = {}
        hum_type_found = None

        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                domain_fields[var] = ogd_api.get_from_ogd(req)
            
            for hv in HUM_VARS:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hv,
                                           reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                    res_h = ogd_api.get_from_ogd(req_h)
                    if res_h is not None:
                        domain_fields["HUM"], hum_type_found = res_h, hv
                        break
                except: continue

            if "HUM" not in domain_fields:
                print(f"  [!] Missing humidity for horizon {h_int}, skipping.")
                continue

            if cached_indices is None:
                cached_indices = get_location_indices(domain_fields[CORE_VARS[0]], locations)

            for name in locations_to_process:
                idx = cached_indices[name]
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")

                loc_data = {}
                for var_name, ds_field in domain_fields.items():
                    # --- CORE DIAGNOSTIC BLOCK ---
                    # We only log on failure, but we gather data beforehand
                    try:
                        possible_dims = ['cell', 'ncells', 'values', 'index', 'node']
                        spatial_dim = None
                        for d in possible_dims:
                            if d in ds_field.dims:
                                spatial_dim = d
                                break
                        
                        if spatial_dim and len(idx) == 1:
                            subset = ds_field.isel({spatial_dim: idx[0]})
                        elif not spatial_dim and len(idx) == 2:
                            subset = ds_field.isel(y=idx[0], x=idx[1])
                        else:
                            # If we hit the fallback, we want to know why
                            fallback_dim = max(ds_field.dims, key=lambda d: ds_field.dims[d])
                            subset = ds_field.isel({fallback_dim: idx[0]})

                        loc_data[var_name] = subset.squeeze().compute()
                    
                    except Exception as e:
                        print(f"\n!!! CRITICAL EXTRACTION ERROR !!!")
                        print(f"Location: {name}, Variable: {var_name}, Horizon: +{h_int}h")
                        print(f"Index Tuple (idx): {idx} (Length: {len(idx)})")
                        print(f"Dataset Dimensions: {dict(ds_field.dims)}")
                        print(f"Dataset Coordinates: {list(ds_field.coords)}")
                        print(f"Spatial Dimension Detected: {spatial_dim}")
                        print(f"Exception Type: {type(e).__name__}")
                        print(f"Exception Message: {e}")
                        raise e # Stop the script so we can see the log

                # Save if successful
                ds_final = xr.Dataset(loc_data)
                ds_final.attrs = {
                    "location": name, "HUM_TYPE": hum_type_found, 
                    "ref_time": ref_time.isoformat(), "horizon_h": h_int,
                    "valid_time": valid_time.isoformat()
                }
                for v in ds_final.data_vars: ds_final[v].attrs = {}
                for c in ds_final.coords: ds_final[c].attrs = {}
                ds_final.to_netcdf(cache_path)
                print(f"    -> Saved: {name}")

        except Exception as e:
            print(f"  [ERROR] Horizon +{h_int}h failed: {e}")
            # In diagnostic mode, we might want to exit immediately to see the log clearly
            sys.exit(1)

if __name__ == "__main__":
    main()
