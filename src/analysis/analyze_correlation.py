# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pandas",
#     "scipy",
# ]
# ///
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
import os

def main():
    cache_dir = "data/processed_data"
    aod_file = os.path.join(cache_dir, "aod_annual.csv")
    pet_file = os.path.join(cache_dir, "pet_annual.csv")
    
    if not os.path.exists(aod_file) or not os.path.exists(pet_file):
        print("Required processed data files not found. Please run plot_aod_trends.py and plot_pet_trends.py first.")
        return
        
    print(f"Loading {aod_file}...")
    df_aod = pd.read_csv(aod_file, parse_dates=['Date'])
    df_aod.set_index('Date', inplace=True)
    
    print(f"Loading {pet_file}...")
    df_pet = pd.read_csv(pet_file, parse_dates=['Date'])
    df_pet.set_index('Date', inplace=True)
    
    # Merge the dataframes on Date (inner join to get only days where both exist)
    df_merged = pd.merge(df_aod, df_pet, left_index=True, right_index=True, how='inner')
    
    # Drop NaNs
    df_merged.dropna(subset=['Mean_AOD', 'Mean_PET'], inplace=True)
    
    if df_merged.empty:
        print("No overlapping data between AOD and PET.")
        return
        
    print(f"\nSuccessfully merged {len(df_merged)} overlapping days.")
    
    # Calculate Pearson and Spearman correlation
    pearson_corr = df_merged['Mean_AOD'].corr(df_merged['Mean_PET'], method='pearson')
    spearman_corr = df_merged['Mean_AOD'].corr(df_merged['Mean_PET'], method='spearman')
    
    print(f"Pearson Correlation Coefficient: {pearson_corr:.3f}")
    print(f"Spearman Correlation Coefficient: {spearman_corr:.3f}")
    
    # Time-Lagged Cross-Correlation (TLCC)
    # We will interpolate missing dates to have a continuous time series for accurate lag analysis
    df_resampled = df_merged.resample('D').mean().interpolate(method='linear')
    
    aod_series = df_resampled['Mean_AOD']
    pet_series = df_resampled['Mean_PET']
    
    # Normalize series to mean 0 and std 1
    aod_norm = (aod_series - aod_series.mean()) / aod_series.std()
    pet_norm = (pet_series - pet_series.mean()) / pet_series.std()
    
    # Calculate cross correlation
    cc = signal.correlate(aod_norm, pet_norm, mode='full')
    cc /= len(aod_norm)
    
    lags = signal.correlation_lags(len(aod_norm), len(pet_norm), mode='full')
    
    # Find the peak correlation and its lag
    max_corr_idx = np.argmax(np.abs(cc))
    best_lag = lags[max_corr_idx]
    best_corr = cc[max_corr_idx]
    
    print(f"\nPeak Cross-Correlation: {best_corr:.3f} at lag = {best_lag} days")
    if best_lag > 0:
        print(f"This indicates that AOD tends to lead PET by {best_lag} days.")
    elif best_lag < 0:
        print(f"This indicates that PET tends to lead AOD by {abs(best_lag)} days.")
    else:
        print("AOD and PET are strongly correlated simultaneously (lag = 0).")
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Scatter Plot
    ax1 = axes[0]
    scatter = ax1.scatter(df_merged['Mean_PET'], df_merged['Mean_AOD'], alpha=0.5, c=df_merged.index.month, cmap='twilight', s=25)
    ax1.set_xlabel('Mean PET', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean AOD', fontsize=12, fontweight='bold')
    ax1.set_title('Scatter Plot: AOD vs PET\n(Color=Month of Year)', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Trend line
    z = np.polyfit(df_merged['Mean_PET'], df_merged['Mean_AOD'], 1)
    p = np.poly1d(z)
    ax1.plot(df_merged['Mean_PET'], p(df_merged['Mean_PET']), "r--", alpha=0.8, linewidth=2, label=f'Trend (r={pearson_corr:.2f})')
    ax1.legend()
    
    # 2. Cross-Correlation Plot
    ax2 = axes[1]
    # Zoom in on lags between -100 and +100 days
    mask = (lags >= -100) & (lags <= 100)
    ax2.plot(lags[mask], cc[mask], color='purple', linewidth=2)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5, label='Zero Lag')
    ax2.axvline(best_lag, color='red', linestyle='-', alpha=0.7, label=f'Max Corr Lag ({best_lag} days)')
    ax2.set_xlabel('Lag (Days)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cross-Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('Time-Lagged Cross-Correlation (TLCC)\nAOD vs PET', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = "outputs/correlation_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCorrelation plots successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
