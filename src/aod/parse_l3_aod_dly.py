# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "h5py",
#     "numpy",
#     "pillow",
#     "matplotlib",
#     "scipy",
#     "tqdm",
# ]
# ///
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np
import glob
import subprocess
import sys
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import scipy.ndimage
from tqdm import tqdm

import argparse
import warnings

def extract_data_from_file(file_path):
    try:
        f = h5py.File(file_path, 'r')
    except Exception as e:
        print(f"-> Failed to open file: {e}")
        return None
        
    with f:
        # Check for common AOD keys
        dataset_name = None
        for key in f.keys():
            if 'AOD' in key.upper() or 'AD' in key.upper():
                dataset_name = key
                break
        
        if not dataset_name:
            for key in f.keys():
                if key not in ['Latitude', 'Longitude', 'time', 'X', 'Y', 'Projection_Information']:
                    dataset_name = key
                    break

        if not dataset_name:
            return None
            
        raw_data = f[dataset_name][0, :, :] if len(f[dataset_name].shape) == 3 else f[dataset_name][:, :]
        
        # Mask out invalid/fill values (e.g., negative values or very large fill values)
        # Set invalid values to np.nan for nanmean computation
        data = raw_data.astype(float)
        data[(data < 0) | (data >= 10)] = np.nan
        return data

def process_aod_window(file_paths):
    target_file = file_paths[-1]
    
    arrays = []
    for fp in file_paths:
        data = extract_data_from_file(fp)
        if data is not None:
            arrays.append(data)
            
    if not arrays:
        return None, None
        
    # Stack arrays and compute nanmax to fill in gaps across the window and preserve peak values
    stacked_data = np.stack(arrays, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        agg_data = np.nanmax(stacked_data, axis=0)
        
    # Fill nan with 0 for computation
    agg_data = np.nan_to_num(agg_data, nan=0.0)
    
    # Apply spatial smoothing filter (morphological dilation)
    # This replaces each pixel with the maximum value in a 5x5 neighborhood, effectively spreading the valid data to fill small gaps
    smoothed_data = scipy.ndimage.maximum_filter(agg_data, size=5)
    
    # Valid mask is where smoothed_data is > 0 (since invalid was filled with 0)
    valid_mask = smoothed_data > 0
    
    # AOD values typically range from 0.0 to 1.0 (sometimes up to 2.0 for high pollution)
    # We normalize between 0.0 and 1.0
    vmin, vmax = 0.0, 1.0
    
    norm_data = np.clip((smoothed_data - vmin) / (vmax - vmin), 0, 1)
    
    # 1. Grayscale Image 
    gray_intensity = norm_data * 255
    h, w = gray_intensity.shape
    img_gray = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(3):
        img_gray[:, :, i] = np.where(valid_mask, gray_intensity, 0)
        
    # 2. Color Image 
    cmap = plt.get_cmap('jet')
    color_rgba = cmap(norm_data) 
    img_color = (color_rgba[:, :, :3] * 255).astype(np.uint8)
    for i in range(3):
        img_color[:, :, i] = np.where(valid_mask, img_color[:, :, i], 0)
        
    # 3. Add Labels
    filename = os.path.basename(target_file)
    parts = filename.split('_')
    timestamp_str = ""
    if len(parts) >= 3:
        timestamp_str = f" | {parts[1]}"
        
    text = f" AOD{timestamp_str} "
    
    text_img = Image.new('RGB', (len(text) * 6 + 10, 15), color=(0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((5, 2), text, fill=(255, 255, 255))
    
    scale = 4
    text_img = text_img.resize((text_img.width * scale, text_img.height * scale), Image.NEAREST)
    
    for img in [img_gray, img_color]:
        pil_img = Image.fromarray(img)
        pil_img.paste(text_img, (40, 40))
        img[:] = np.array(pil_img)
        
    out_gray = target_file.replace('.nc', '_GRAY.ppm')
    out_color = target_file.replace('.nc', '_COLOR.ppm')
    
    header = f"P6\n{w} {h}\n255\n".encode('ascii')
    with open(out_gray, 'wb') as img_file:
        img_file.write(header)
        img_file.write(img_gray.tobytes())
        
    with open(out_color, 'wb') as img_file:
        img_file.write(header)
        img_file.write(img_color.tobytes())
        
    return out_gray, out_color


def create_movie(list_file, output_movie):
    print(f"\nCreating movie {output_movie}...")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-fps_mode", "vfr", "-pix_fmt", "yuv420p", output_movie
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"-> Successfully created movie: {output_movie}")
    except subprocess.CalledProcessError as e:
        print(f"-> Failed to create movie. Error:\n{e.stderr.decode()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AOD files into timeline videos.")
    parser.add_argument("--days", type=int, default=0, help="Number of recent days to process (0 for all)")
    parser.add_argument("--window", type=int, default=7, help="Rolling window size in days for composite (default: 7)")
    args = parser.parse_args()

    search_path = r"X:\mdapi_data\E06OCM_L3_LAC_AD\**\*.nc"
    
    def extract_datetime(filepath):
        # We need to extract datetime from filenames like E06OCML3AD_20260824_...
        # The date is the second part (index 1) and is formatted as %Y%m%d
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) >= 2:
            try:
                return datetime.strptime(parts[1], "%Y%m%d")
            except Exception:
                pass
        return datetime.min

    files = glob.glob(search_path, recursive=True)
    files.sort(key=extract_datetime)
    
    if not files:
        print("No L3 AOD .nc files found!")
        sys.exit(0)
        
    if args.days > 0:
        files = files[-args.days:]
        
    print(f"Found {len(files)} files to process with a {args.window}-day rolling window.")
    
    gray_images = []
    color_images = []
    
    for i in tqdm(range(len(files)), desc="Processing AOD windows"):
        start_idx = max(0, i - args.window + 1)
        window_files = files[start_idx:i+1]
        
        gray, color = process_aod_window(window_files)
        if gray and color:
            gray_images.append(gray)
            color_images.append(color)
            
    if gray_images:
        gray_list = "data/movie_list_aod_gray.txt"
        with open(gray_list, "w") as f:
            for img in gray_images:
                f.write(f"file '{img}'\n")
                f.write(f"duration 0.5\n")
            f.write(f"file '{gray_images[-1]}'\n")
        create_movie(gray_list, "outputs/aod_timeline_GRAY.mp4")
        
    if color_images:
        color_list = "data/movie_list_aod_color.txt"
        with open(color_list, "w") as f:
            for img in color_images:
                f.write(f"file '{img}'\n")
                f.write(f"duration 0.5\n")
            f.write(f"file '{color_images[-1]}'\n")
        create_movie(color_list, "outputs/aod_timeline_COLOR.mp4")
