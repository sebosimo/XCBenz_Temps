import os
import sys
import shutil
import xarray as xr
import numpy as np
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_data

def test_process_traces_naming_and_hhl():
    fetch_data.CACHE_DIR_TRACES = "tests/cache_data"
    
    # 1. Setup Mock Data
    tag = "20990101_1200"
    h = 0
    ref_time = datetime.datetime(2099, 1, 1, 12, 0)
    locations = {"TestLoc": {"lat": 46.0, "lon": 8.0}}
    
    # Create simple grid
    lats = np.array([46.0, 47.0])
    lons = np.array([8.0, 9.0])
    
    # Create T variable
    data_t = xr.DataArray(np.random.rand(10, 2), dims=("generalVerticalLayer", "values"), 
                          coords={"latitude": (("values",), lats), "longitude": (("values",), lons)})
    
    # Create HHL variable (Using same dim for test simplicity, although technically n+1)
    data_hhl = xr.DataArray(np.random.rand(10, 2), dims=("generalVerticalLayer", "values"),
                            coords={"latitude": (("values",), lats), "longitude": (("values",), lons)})
    
    fields = {"T": data_t, "HHL": data_hhl}
    
    try:
        # 2. Execution
        fetch_data.process_traces(fields, locations, tag, h, ref_time)
        
        # 3. Assertions
        # Check Directory Structure
        loc_dir = os.path.join(fetch_data.CACHE_DIR_TRACES, tag, "TestLoc")
        assert os.path.exists(loc_dir), f"Directory not created: {loc_dir}"
        
        # Check Filename
        expected_name = f"TestLoc_{tag}_H{h:02d}.nc"
        file_path = os.path.join(loc_dir, expected_name)
        assert os.path.exists(file_path), f"File name incorrect: {file_path}"
        
        # Check Content (HHL presence)
        ds_out = xr.open_dataset(file_path)
        assert "HHL" in ds_out.data_vars, "HHL variable missing from trace output!"
        ds_out.close()
        
        print("Traces naming and HHL inclusion test passed!")
        
    finally:
        # Cleanup
        if os.path.exists("tests/cache_data"):
            shutil.rmtree("tests/cache_data")

if __name__ == "__main__":
    test_process_traces_naming_and_hhl()
