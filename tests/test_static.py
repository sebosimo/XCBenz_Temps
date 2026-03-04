import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch_data

def test_static_download():
    # Define paths
    static_dir = fetch_data.STATIC_DIR
    hhl_path = os.path.join(static_dir, fetch_data.HHL_FILENAME)
    
    # 1. Setup: Backup existing file if it exists
    backup_path = hhl_path + ".bak"
    if os.path.exists(hhl_path):
        os.rename(hhl_path, backup_path)
    
    try:
        # 2. Execution: Run download_static_files
        print("Testing static file download from server...")
        fetch_data.download_static_files()
        
        # 3. Assertion: Check if file exists and has size > 0
        assert os.path.exists(hhl_path), "HHL file was not downloaded!"
        assert os.path.getsize(hhl_path) > 0, "HHL file is empty!"
        print("Static file download test passed!")
        
    finally:
        # 4. Teardown: Restore backup
        if os.path.exists(backup_path):
            if os.path.exists(hhl_path):
                os.remove(hhl_path) # Remove test download
            os.rename(backup_path, hhl_path)

if __name__ == "__main__":
    test_static_download()
