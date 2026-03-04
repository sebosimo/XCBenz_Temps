import os
import sys
import shutil
import xarray as xr
import datetime
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_data
from meteodatalab import ogd_api

def test_integration_real_download_and_process():
    # Setup test cache dir
    test_cache = "tests/cache_real"
    fetch_data.CACHE_DIR_MAPS = test_cache
    if os.path.exists(test_cache):
        shutil.rmtree(test_cache)
    os.makedirs(test_cache, exist_ok=True)
    
    # Ensure static files
    fetch_data.download_static_files()
    hhl = fetch_data.load_static_hhl()
    grid = fetch_data.load_static_grid()
    assert hhl is not None, "Failed to load real HHL"
    assert grid is not None, "Failed to load real Grid"

    # Inject coords into HHL (Production Logic copy)
    n_grid = grid['lat'].size
    match_dim = next((d for d in hhl.dims if hhl.sizes[d] == n_grid), None)
    if match_dim:
        hhl = hhl.assign_coords({
            "latitude": (match_dim, grid['lat'].values),
            "longitude": (match_dim, grid['lon'].values)
        })

    # 1. Discovery (Try up to 3 runs)
    runs = fetch_data.get_latest_available_runs(limit=3)
    
    valid_run = None
    for r in runs:
        # Quick check if it has assets
        iso_h = fetch_data.get_iso_horizon(0)
        req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="U", reference_datetime=r, horizon=iso_h, perturbed=False)
        urls = ogd_api.get_asset_urls(req)
        print(f"Run {r}: URLs found: {len(urls) if urls else 0}")
        if urls:
            valid_run = r
            break
    
    if not valid_run:
        print("Skipping test: No valid runs with assets found.")
        return

    ref_time = valid_run
    tag = ref_time.strftime('%Y%m%d_%H%M')
    print(f"Testing with real run: {tag}")
    
    # 2. Download Real Data for H+00
    h = 0
    iso_h = fetch_data.get_iso_horizon(h)
    
    fields = {"HHL": hhl}
    vars_to_fetch = ["U", "V"] # We need U and V for wind maps
    
    # Unique temp dir to avoid Windows file locking issues
    import uuid
    run_id = str(uuid.uuid4())[:8]
    temp_dir = f"tests/temp_gribs_{run_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        for var in vars_to_fetch:
            print(f"Downloading real {var}...")
            # ... request logic ...
            req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                 reference_datetime=ref_time, horizon=iso_h, perturbed=False)
            urls = ogd_api.get_asset_urls(req)
            if not urls:
                print(f"Skipping test: No URL for {var}")
                return
            
            tmp = os.path.join(temp_dir, f"temp_{var}.grib2")
            if fetch_data.download_file(urls[0], tmp):
                 ds = xr.open_dataset(tmp, engine='cfgrib', backend_kwargs={'indexpath': ''})
                 # Standard logic from fetch_data
                 data = ds[next(iter(ds.data_vars))].load()
                 
                 # Inject coords
                 m_dim = next(d for d in data.dims if data.sizes[d] == grid['lat'].size)
                 data = data.assign_coords({"latitude": (m_dim, grid['lat'].values), "longitude": (m_dim, grid['lon'].values)})
                 
                 fields[var] = data
                 ds.close()
            else:
                 assert False, f"Failed to download {var}"
        
        # 3. Execution: process_wind_maps
        # Setup specific WIND_LEVELS for test
        fetch_data.WIND_LEVELS = [
            {"name": "10m_AGL",   "h": 10,   "type": "AGL"} # Just test one level for speed
        ]
        
        print(f"Running process_wind_maps with REAL data. Fields keys: {fields.keys()}")
        fetch_data.process_wind_maps(fields, tag, h, ref_time)
        
        # 4. Assertion
        out_dir = os.path.join(test_cache, tag)
        expected_file = os.path.join(out_dir, f"Wind_AGL_10m_AGL_{tag}_H{h:02d}.nc")
        
        assert os.path.exists(expected_file), f"Output file not created: {expected_file}"
        print(f"Success! Real file created: {expected_file}")
        
        # Verify content
        ds_out = xr.open_dataset(expected_file)
        print("Output file contents:", list(ds_out.data_vars))
        assert f"u_10m_AGL" in ds_out.data_vars
        ds_out.close()

    finally:
        # Cleanup temp
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir, ignore_errors=True)
            except: pass
        if os.path.exists(test_cache):
            shutil.rmtree(test_cache, ignore_errors=True)

if __name__ == "__main__":
    test_integration_real_download_and_process()
