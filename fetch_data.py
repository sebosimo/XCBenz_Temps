import os, sys
import datetime, json, xarray as xr
import numpy as np
import warnings
import requests
import time
import shutil

# Set GRIB definitions for COSMO/ICON
COSMO_DEFS = r"C:\Users\sebas\.conda\envs\weather_final\share\eccodes-cosmo-resources\definitions"
STANDARD_DEFS = os.path.join(sys.prefix, "Library", "share", "eccodes", "definitions")

defs_to_use = []
if os.path.exists(COSMO_DEFS):
    defs_to_use.append(COSMO_DEFS)
if os.path.exists(STANDARD_DEFS):
    defs_to_use.append(STANDARD_DEFS)

if defs_to_use:
    final_def_path = ":".join(defs_to_use)
    os.environ["GRIB_DEFINITION_PATH"] = final_def_path
    os.environ["ECCODES_DEFINITION_PATH"] = final_def_path

from meteodatalab import ogd_api

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
VARS_TRACES = ["T", "U", "V", "P", "QV"]
VARS_MAPS = ["U", "V", "HHL"] 
CACHE_DIR_TRACES = "cache_data"
CACHE_DIR_MAPS = "cache_wind"
STATIC_DIR = "static_data"
HHL_FILENAME = "vertical_constants_icon-ch1-eps.grib2"
HGRID_FILENAME = "horizontal_constants_icon-ch1-eps.grib2"
STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-forecasting-icon-ch1"
STAC_ASSETS_URL = f"{STAC_BASE_URL}/assets"

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

def sanitize_name(name):
    n = name.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae") \
            .replace("Ü", "Ue").replace("Ö", "Oe").replace("Ä", "Ae").replace("ß", "ss")
    clean = "".join(c for c in n if c.isalnum() or c in ('-', '_'))
    return clean if clean else "unnamed"

def log(msg, level="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("debug_log.txt", "a") as f:
        f.write(f"{timestamp} [{level}] {msg}\n")
    print(f"{timestamp} [{level}] {msg}", flush=True)

def download_file(url, target_path, max_retries=3):
    """Downloads a file with retries and exponential backoff."""
    backoff = 2
    for attempt in range(max_retries):
        try:
            log(f"Downloading {url} to {target_path}...")
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            log(f"Download complete: {target_path}")
            return True
        except Exception as e:
            log(f"Download attempt {attempt+1} failed: {e}", "ERROR")
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
    return False

def get_latest_available_runs(limit=1):
    """Discovers actual runs available on the server using Active Probing."""
    log("Discovering runs via Active Probing...")
    now = datetime.datetime.now(datetime.timezone.utc)
    hour = (now.hour // 3) * 3
    start = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    
    found_runs = []
    for i in range(16): # Check last 48 hours
        cand = start - datetime.timedelta(hours=i*3)
        ref = cand.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        params = {"limit": 1, "forecast:reference_datetime": ref}
        try:
            r = requests.get(f"{STAC_BASE_URL}/items", params=params, timeout=10)
            if r.status_code == 200 and r.json().get("features"):
                found_runs.append(cand)
                log(f"Found available run: {ref}")
                if len(found_runs) >= limit: break
        except: pass
    return found_runs

def download_static_files():
    os.makedirs(STATIC_DIR, exist_ok=True)
    for filename in [HHL_FILENAME, HGRID_FILENAME]:
        path = os.path.join(STATIC_DIR, filename)
        if not os.path.exists(path):
            log(f"Downloading static file {filename}...")
            try:
                resp = requests.get(STAC_ASSETS_URL, timeout=10)
                assets = resp.json()["assets"]
                url = next((a["href"] for a in assets if a.get("id") == filename), None)
                if url: download_file(url, path)
            except Exception as e: log(f"Failed to fetch static {filename}: {e}", "ERROR")

def load_static_hhl():
    path = os.path.join(STATIC_DIR, HHL_FILENAME)
    if not os.path.exists(path): return None
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        var = next((v for v in ds.data_vars if v.lower() in ['h', 'hhl']), list(ds.data_vars)[0])
        hhl = ds[var].load()
        ds.close()
        return hhl
    except Exception as e: log(f"Error loading HHL: {e}", "ERROR"); return None

def load_static_grid():
    path = os.path.join(STATIC_DIR, HGRID_FILENAME)
    if not os.path.exists(path): return None
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        grid = {}
        for key in ['lat', 'lon']:
            grid[key] = next((ds[k].load() for k in ds.coords if key in k.lower()), None)
        ds.close()
        return grid if grid['lat'] is not None else None
    except Exception as e: log(f"Error loading HGRID: {e}", "ERROR"); return None

def is_run_complete_locally(time_tag, locations, max_h):
    last_loc = sanitize_name(list(locations.keys())[-1])
    trace_path = os.path.join(CACHE_DIR_TRACES, time_tag, last_loc, f"H{max_h:02d}.nc")
    map_path = os.path.join(CACHE_DIR_MAPS, time_tag, f"wind_maps_H{max_h:02d}.nc")
    return os.path.exists(trace_path) and os.path.exists(map_path)

def process_traces(fields, locations, tag, h, ref):
    sample = list(fields.values())[0]
    lat_n = 'latitude' if 'latitude' in sample.coords else 'lat'
    lon_n = 'longitude' if 'longitude' in sample.coords else 'lon'
    lats, lons = sample[lat_n].values, sample[lon_n].values
    indices = {n: int(np.argmin((lats-c['lat'])**2+(lons-c['lon'])**2)) for n, c in locations.items()}

    for name, idx in indices.items():
        loc_dir = os.path.join(CACHE_DIR_TRACES, tag, sanitize_name(name))
        os.makedirs(loc_dir, exist_ok=True)
        path = os.path.join(loc_dir, f"H{h:02d}.nc")
        if os.path.exists(path): continue

        loc_vars = {}
        for var, ds in fields.items():
            if var == "HHL": continue
            s_dim = ds[lat_n].dims[0]
            profile = ds.squeeze().isel({s_dim: idx}).compute()
            if profile.dims: profile = profile.rename({profile.dims[0]: 'level'})
            loc_vars[var] = profile.drop_vars([c for c in profile.coords if c not in profile.dims])

        ds_out = xr.Dataset(loc_vars)
        ds_out.attrs = {"location": name, "ref_time": ref.isoformat(), "horizon": h}
        ds_out.to_netcdf(path)

def process_wind_maps(fields, tag, h_int, ref):
    output_path = os.path.join(CACHE_DIR_MAPS, tag, f"wind_maps_H{h_int:02d}.nc")
    if os.path.exists(output_path) or "U" not in fields or "V" not in fields or "HHL" not in fields: return

    from metpy.interpolate import interpolate_to_isosurface
    u, v, hhl = fields["U"].squeeze(), fields["V"].squeeze(), fields["HHL"].squeeze()
    
    try:
        z_dim = hhl.dims[0]
        z_f = (hhl.isel({z_dim: slice(0,-1)}).values + hhl.isel({z_dim: slice(1,None)}).values) / 2
        h_surf = hhl.isel({z_dim: -1})
        
        np_u, np_v = u.values, v.values
        np_z = z_f
        
        out_ds = {}
        for lvl in WIND_LEVELS:
            try:
                target_z = np_z - h_surf.values if lvl['type'] == 'AGL' else np_z
                res_u = interpolate_to_isosurface(target_z, np_u, lvl['h'])
                res_v = interpolate_to_isosurface(target_z, np_v, lvl['h'])
                
                spatial = u.dims[-1]
                coords = {spatial: u[spatial], "latitude": u.latitude, "longitude": u.longitude}
                out_ds[f"u_{lvl['name']}"] = xr.DataArray(res_u, dims=[spatial], coords=coords)
                out_ds[f"v_{lvl['name']}"] = xr.DataArray(res_v, dims=[spatial], coords=coords)
            except: pass
        
        if out_ds:
            xr.Dataset(out_ds).to_netcdf(output_path)
            log(f"Saved wind map: {output_path}")
    except Exception as e: log(f"Wind map error: {e}", "ERROR")

def main():
    log("Main start...")
    download_static_files()
    if not os.path.exists("locations.json"): return
    with open("locations.json", "r", encoding="utf-8") as f: locations = json.load(f)

    runs = get_latest_available_runs(limit=3)
    if not runs: log("No runs found."); return

    hhl = load_static_hhl()
    grid = load_static_grid()

    for ref_time in runs:
        tag = ref_time.strftime('%Y%m%d_%H%M')
        max_h = 45 if ref_time.hour == 3 else 33
        if is_run_complete_locally(tag, locations, max_h):
            log(f"Run {tag} complete locally."); break
        
        log(f"Processing run: {tag}")
        for h in range(max_h + 1):
            iso_h = get_iso_horizon(h)
            fields = {"HHL": hhl} if hhl is not None else {}
            
            for var in ["T", "U", "V", "P", "QV"]:
                try:
                    req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                         reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                    urls = ogd_api.get_asset_urls(req)
                    if urls:
                        tmp = f"temp_{var}_{tag}_{h:02d}.grib2"
                        if download_file(urls[0], tmp):
                            ds = xr.open_dataset(tmp, engine='cfgrib', backend_kwargs={'indexpath': ''})
                            data = ds[next(iter(ds.data_vars))].load()
                            if grid:
                                m_dim = next(d for d in data.dims if data.sizes[d] == grid['lat'].size)
                                data = data.assign_coords({"latitude": (m_dim, grid['lat'].values), "longitude": (m_dim, grid['lon'].values)})
                            fields[var] = data
                            ds.close()
                            os.remove(tmp)
                except Exception as e: log(f"{var} H+{h} err: {e}", "ERROR")
            
            if fields:
                process_traces(fields, locations, tag, h, ref_time)
                process_wind_maps(fields, tag, h, ref_time)
                log(f"H+{h:02d} done")
        break # Only process latest successful run

    cleanup_old_runs()

def cleanup_old_runs():
    ret = os.environ.get("RETENTION_DAYS")
    if not ret: return
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(ret))
    for d in [CACHE_DIR_TRACES, CACHE_DIR_MAPS]:
        if not os.path.exists(d): continue
        for item in os.listdir(d):
            path = os.path.join(d, item)
            try:
                dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M").replace(tzinfo=datetime.timezone.utc)
                if dt < cutoff: shutil.rmtree(path)
            except: pass

if __name__ == "__main__":
    main()

