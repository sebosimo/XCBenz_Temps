import os
import sys
import shutil
import xarray as xr
import numpy as np
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_data

def test_process_wind_maps_naming():
    fetch_data.CACHE_DIR_MAPS = "tests/cache_wind"
    
    # 1. Setup Mock Config & Data
    
    # Manually inject WIND_LEVELS for test
    fetch_data.WIND_LEVELS = [
        {"name": "10m_AGL",   "h": 10,   "type": "AGL"},
        {"name": "2000m_AMSL","h": 2000, "type": "AMSL"}
    ]
    
    tag = "20990101_1200"
    h_int = 3
    ref = datetime.datetime(2099, 1, 1, 12, 0)
    
    # Create simple grid (10x10)
    lats = np.linspace(46, 47, 10)
    lons = np.linspace(8, 9, 10)
    
    # Create 3D fields (Z=5, Y=10, X=10)
    data_u = xr.DataArray(np.random.rand(5, 10, 10), 
                          dims=("generalVerticalLayer", "latitude", "longitude"),
                          coords={"latitude": ("latitude", lats), "longitude": ("longitude", lons)})
    data_v = xr.DataArray(np.random.rand(5, 10, 10), 
                          dims=("generalVerticalLayer", "latitude", "longitude"),
                          coords={"latitude": ("latitude", lats), "longitude": ("longitude", lons)})
    
    # HHL Z=6
    data_hhl = xr.DataArray(np.linspace(0, 5000, 6).reshape(6,1,1) * np.ones((1,10,10)), 
                           dims=("generalVerticalLayer_plus1", "latitude", "longitude"),
                           coords={"latitude": ("latitude", lats), "longitude": ("longitude", lons)})

    fields = {"U": data_u, "V": data_v, "HHL": data_hhl}
    
    try:
        # 2. Execution
        print("Running process_wind_maps...")
        fetch_data.process_wind_maps(fields, tag, h_int, ref)
        
        # 3. Assertions
        out_dir = os.path.join(fetch_data.CACHE_DIR_MAPS, tag)
        assert os.path.exists(out_dir), f"Output directory not created: {out_dir}"
        
        # Check for specific files
        # 1. AGL file
        file_agl = os.path.join(out_dir, f"Wind_AGL_10m_AGL_{tag}_H{h_int:02d}.nc")
        assert os.path.exists(file_agl), f"AGL wind map missing: {file_agl}"
        
        # 2. AMSL file
        file_amsl = os.path.join(out_dir, f"Wind_AMSL_2000m_AMSL_{tag}_H{h_int:02d}.nc")
        assert os.path.exists(file_amsl), f"AMSL wind map missing: {file_amsl}"
        
        print("Wind map naming and splitting test passed!")
        
    finally:
        # Cleanup
        if os.path.exists("tests/cache_wind"):
            shutil.rmtree("tests/cache_wind")

if __name__ == "__main__":
    test_process_wind_maps_naming()
