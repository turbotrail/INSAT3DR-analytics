# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "h5py",
#     "numpy",
#     "matplotlib",
#     "pandas",
#     "tqdm",
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

# Define the bounding boxes for major cities (approx 1x1 degree box)
# Format: { 'CityName': (Lat_Min, Lat_Max, Lon_Min, Lon_Max) }
CITIES = {
    'New Delhi': (28.1, 29.1, 76.7, 77.7),
    'Mumbai': (18.5, 19.5, 72.3, 73.3),
    'Kolkata': (22.0, 23.0, 87.8, 88.8),
    'Chennai': (12.5, 13.5, 79.7, 80.7)
}

def extract_datetime(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            return datetime.strptime(parts[1], "%Y%m%d")
        except Exception:
            pass
    return datetime.min

def extract_city_aod(filepath):
    """Returns a dictionary mapping city name to its daily mean AOD"""
    try:
        with h5py.File(filepath, 'r') as f:
            dataset_name = None
            for key in f.keys():
                if 'AOD' in key.upper() or 'AD' in key.upper():
                    dataset_name = key
                    break
            
            if not dataset_name:
                return None
                
            raw_data = f[dataset_name][:]
            if len(raw_data.shape) == 3:
                raw_data = raw_data[0, :, :]
                
            lat_arr = f['latitude'][:]
            lon_arr = f['longitude'][:]
            
            data = raw_data.astype(float)
            data[(data < 0) | (data >= 10)] = np.nan
            
            city_aods = {}
            for city, (lat_min, lat_max, lon_min, lon_max) in CITIES.items():
                # lat_arr is typically descending, so check bounds safely
                lat_mask = (lat_arr >= lat_min) & (lat_arr <= lat_max)
                lon_mask = (lon_arr >= lon_min) & (lon_arr <= lon_max)
                
                # Get the 2D slice
                city_data = data[np.ix_(lat_mask, lon_mask)]
                
                valid_pixels = np.count_nonzero(~np.isnan(city_data))
                total_pixels = city_data.size
                
                # DATA CLEANING: If less than 15% of the bounding box has valid clear-sky pixels 
                # (due to heavy clouds), the average is unreliable, so we filter it out.
                if valid_pixels < (0.15 * total_pixels):
                    city_aods[city] = np.nan
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        city_aods[city] = np.nanmean(city_data)
                    
            return city_aods
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

if __name__ == "__main__":
    cache_dir = "data/processed_data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "aod_cities.csv")
    
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
    else:
        search_path = r"X:\mdapi_data\E06OCM_L3_LAC_AD\**\*.nc"
        files = glob.glob(search_path, recursive=True)
        
        if not files:
            print("No AOD files found!")
            exit(1)
            
        files.sort(key=extract_datetime)
        
        dates = []
        # Dictionary to hold lists of mean AODs per city
        city_data_lists = {city: [] for city in CITIES.keys()}
        
        for filepath in tqdm(files, desc="Calculating City AODs"):
            dt = extract_datetime(filepath)
            if dt != datetime.min:
                city_aods = extract_city_aod(filepath)
                if city_aods is not None:
                    dates.append(dt)
                    for city in CITIES.keys():
                        city_data_lists[city].append(city_aods[city])
                    
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
        # Calculate 7-day rolling mean to smooth out noisy spikes (clouds, missing swaths, etc)
        rolling = df[city].rolling(window=7, min_periods=1).mean()
        
        # Plot raw daily scatter points (very faint)
        ax.scatter(df.index, df[city], alpha=0.3, color=color, s=20)
        
        # Plot the smooth trend line
        ax.plot(df.index, rolling, color=color, linewidth=2.5, label=city)
        
        ax.set_ylabel('Avg AOD', fontsize=12)
        ax.legend(fontsize=14, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    fig.suptitle('7-Day Rolling Average AOD: Major Indian Cities', fontsize=18, fontweight='bold', y=0.96)
    axes[-1].set_xlabel('Date', fontsize=14, labelpad=10)
    
    # Format X-axis to show months clearly
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Rotate ticks on the bottom axis
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = "outputs/aod_cities_trend.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCity trend graph successfully saved to: {output_path}")
