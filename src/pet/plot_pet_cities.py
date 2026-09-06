# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "h5py",
#     "numpy",
#     "matplotlib",
#     "pandas",
#     "tqdm",
#     "pyproj",
# ]
# ///
import os
import glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from tqdm import tqdm
import warnings
from pyproj import Proj

# Define the bounding boxes for major cities (approx 1x1 degree box)
# Format: { 'CityName': (Lat_Min, Lat_Max, Lon_Min, Lon_Max) }
CITIES = {
    'New Delhi': (28.1, 29.1, 76.7, 77.7),
    'Mumbai': (18.5, 19.5, 72.3, 73.3),
    'Kolkata': (22.0, 23.0, 87.8, 88.8),
    'Chennai': (12.5, 13.5, 79.7, 80.7)
}

# Projection parameters from INSAT-3D PET data
p = Proj(proj='merc', lat_ts=17.75, lon_0=77.25, a=6378137.0, b=6356752.3142)

def extract_datetime(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            return datetime.strptime(parts[1].upper(), "%d%b%Y")
        except Exception:
            pass
    return datetime.min

def extract_city_pet(filepath):
    """Returns a dictionary mapping city name to its daily mean PET"""
    try:
        with h5py.File(filepath, 'r') as f:
            if 'PET_DLY' not in f:
                return None
                
            raw_data = f['PET_DLY'][:]
            if len(raw_data.shape) == 3:
                raw_data = raw_data[0, :, :]
                
            x_arr = f['X'][:]
            y_arr = f['Y'][:]
            
            data = raw_data.astype(float)
            data[data < 0] = np.nan # -999.0 is fill value
            
            city_pets = {}
            for city, (lat_min, lat_max, lon_min, lon_max) in CITIES.items():
                x_min_proj, y_min_proj = p(lon_min, lat_min)
                x_max_proj, y_max_proj = p(lon_max, lat_max)
                
                x_min, x_max = min(x_min_proj, x_max_proj), max(x_min_proj, x_max_proj)
                y_min, y_max = min(y_min_proj, y_max_proj), max(y_min_proj, y_max_proj)
                
                x_mask = (x_arr >= x_min) & (x_arr <= x_max)
                y_mask = (y_arr >= y_min) & (y_arr <= y_max)
                
                # Get the 2D slice
                city_data = data[np.ix_(y_mask, x_mask)]
                
                valid_pixels = np.count_nonzero(~np.isnan(city_data))
                total_pixels = city_data.size
                
                if total_pixels == 0:
                    city_pets[city] = np.nan
                    continue
                
                # DATA CLEANING: If less than 15% of the bounding box has valid pixels 
                if valid_pixels < (0.15 * total_pixels):
                    city_pets[city] = np.nan
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        city_pets[city] = np.nanmean(city_data)
                    
            return city_pets
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

if __name__ == "__main__":
    cache_dir = "data/processed_data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "pet_cities.csv")
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
    else:
        search_path = r"X:\mdapi_data\3RIMG_L3C_PET_DLY\**\*.h5"
        files = glob.glob(search_path, recursive=True)
        
        if not files:
            print("No PET files found!")
            exit(1)
            
        files.sort(key=extract_datetime)
        
        dates = []
        # Dictionary to hold lists of mean PETs per city
        city_data_lists = {city: [] for city in CITIES.keys()}
        
        for filepath in tqdm(files, desc="Calculating City PETs"):
            dt = extract_datetime(filepath)
            if dt != datetime.min:
                city_pets = extract_city_pet(filepath)
                if city_pets is not None:
                    dates.append(dt)
                    for city in CITIES.keys():
                        city_data_lists[city].append(city_pets[city])
                    
        if not dates:
            print("No valid data could be extracted.")
            exit(1)
            
        # Create pandas DataFrame
        df_data = {'Date': dates}
        df_data.update(city_data_lists)
        df = pd.DataFrame(df_data)
        df.set_index('Date', inplace=True)
        
        # Save to cache
        df.to_csv(cache_file)
        print(f"Saved processed data to {cache_file}")
    
    # Plotting
    fig, axes = plt.subplots(nrows=len(CITIES), ncols=1, figsize=(16, 14), sharex=True)
    plt.style.use('seaborn-v0_8-darkgrid')
    
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']
    
    for ax, (city, color) in zip(axes, zip(CITIES.keys(), colors)):
        rolling = df[city].rolling(window=7, min_periods=1).mean()
        
        ax.scatter(df.index, df[city], alpha=0.3, color=color, s=20)
        ax.plot(df.index, rolling, color=color, linewidth=2.5, label=city)
        
        ax.set_ylabel('Avg PET', fontsize=12)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('7-Day Rolling Average PET: Major Indian Cities', fontsize=18, fontweight='bold', y=0.96)
    axes[-1].set_xlabel('Date', fontsize=14, labelpad=10)
    
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = "outputs/pet_cities_trend.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCity trend graph successfully saved to: {output_path}")
