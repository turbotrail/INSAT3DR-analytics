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

def extract_datetime(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) >= 2:
        try:
            return datetime.strptime(parts[1].upper(), "%d%b%Y")
        except Exception:
            pass
    return datetime.min

def extract_mean_pet(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            if 'PET_DLY' not in f:
                return np.nan
                
            raw_data = f['PET_DLY'][:]
            
            if len(raw_data.shape) == 3:
                raw_data = raw_data[0, :, :]
                
            data = raw_data.astype(float)
            data[data < 0] = np.nan  # -999.0 is fill value
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(data)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.nan

if __name__ == "__main__":
    cache_dir = "data/processed_data"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "pet_annual.csv")
    
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
        mean_pets = []
        
        for filepath in tqdm(files, desc="Calculating Daily Average PET"):
            dt = extract_datetime(filepath)
            if dt != datetime.min:
                mean_pet = extract_mean_pet(filepath)
                if not np.isnan(mean_pet):
                    dates.append(dt)
                    mean_pets.append(mean_pet)
                    
        if not dates:
            print("No valid data could be extracted.")
            exit(1)
            
        # Create pandas DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Mean_PET': mean_pets
        })
        df.set_index('Date', inplace=True)
        
        # Save to cache
        df.to_csv(cache_file)
        print(f"Saved processed data to {cache_file}")
    
    # Calculate 7-day rolling mean to smooth out noisy spikes (clouds, missing swaths, etc)
    df['Rolling_7_Day'] = df['Mean_PET'].rolling(window=7, min_periods=1).mean()
    
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.style.use('ggplot')
    
    # Plot raw daily scatter points (semi-transparent)
    plt.scatter(df.index, df['Mean_PET'], alpha=0.4, color='dodgerblue', label='Daily Average PET', s=15)
    
    # Plot the smooth trend line
    plt.plot(df.index, df['Rolling_7_Day'], color='darkblue', linewidth=2.5, label='7-Day Rolling Trend')
    
    plt.title('Daily Average Potential Evapotranspiration (PET) Over India', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Date', fontsize=14, labelpad=10)
    plt.ylabel('Average PET', fontsize=14, labelpad=10)
    
    # Format X-axis to show months clearly
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    output_path = "outputs/pet_annual_trend.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nTrend graph successfully saved to: {output_path}")
