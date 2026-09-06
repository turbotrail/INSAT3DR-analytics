import json
import subprocess
from datetime import datetime, timedelta
import os

import sys

if len(sys.argv) > 1:
    config_path = sys.argv[1]
else:
    config_path = r"src/mdapi/config_aod_dly.json"
start_date_str = "2025-01-01"
end_date_str = "2025-12-31"

start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

current_start = start_dt

while current_start <= end_dt:
    current_end = current_start + timedelta(days=99)
    if current_end > end_dt:
        current_end = end_dt
        
    print(f"\n=======================================================")
    print(f"Downloading from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
    print(f"=======================================================\n")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    config["search_parameters"]["startTime"] = current_start.strftime("%Y-%m-%d")
    config["search_parameters"]["endTime"] = current_end.strftime("%Y-%m-%d")
    config["search_parameters"]["count"] = "100"
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    # Run the download command
    subprocess.run(["uv", "run", "src/mdapi/mdapi.py", config_path])
    
    current_start = current_end + timedelta(days=1)

print("\nFinished downloading all batches for the year!")
