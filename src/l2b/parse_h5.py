import h5py
import numpy as np
import glob
import os

def parse_and_summarize(file_path):
    print(f"\n--- Analyzing: {os.path.basename(file_path)} ---")
    with h5py.File(file_path, 'r') as f:
        
        # The variables available in this dataset
        lat = f['Latitude'][:]
        lon = f['Longitude'][:]
        
        # Extract Cloud Mask (CMK)
        # The shape is (1, y, x), so we take index 0
        cmk = f['CMK'][0, :, :]
        
        # flag values: 0: Clear, 1: Cloudy, 2: Probably Clear, 3: Probably Cloudy
        # 9 is the fill value (No data)
        fill_value = f['CMK'].attrs['_FillValue'][0]
        
        # The latitude and longitude data are stored as scaled integers.
        # We need to apply the scale_factor to convert them back to degrees.
        lat_scale = f['Latitude'].attrs.get('scale_factor', [1.0])[0]
        lon_scale = f['Longitude'].attrs.get('scale_factor', [1.0])[0]
        lat_fill = f['Latitude'].attrs.get('_FillValue', [32767])[0]
        lon_fill = f['Longitude'].attrs.get('_FillValue', [32767])[0]
        
        # Convert to float to handle scale factor
        lat = np.where(lat != lat_fill, lat * lat_scale, np.nan)
        lon = np.where(lon != lon_fill, lon * lon_scale, np.nan)
        
        # Regional Bounding Box (South India & Chennai)
        lat_min, lat_max = 10.0, 15.0
        lon_min, lon_max = 78.0, 82.0
        
        # Create a boolean mask for the bounding box
        # We use np.nan_to_num with False as fallback since nan comparisons evaluate to False
        bbox_mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        
        # Find the row and column indices where the mask is True
        rows = np.any(bbox_mask, axis=1)
        cols = np.any(bbox_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            print("No data found for the region in this file.")
            return
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop all arrays to the bounding box
        cmk = cmk[rmin:rmax+1, cmin:cmax+1]
        lat = lat[rmin:rmax+1, cmin:cmax+1]
        lon = lon[rmin:rmax+1, cmin:cmax+1]
        
        # Get only the valid data points in the cropped region
        valid_mask = cmk != fill_value
        valid_cmk = cmk[valid_mask]
        
        total_pixels = valid_cmk.size
        
        if total_pixels == 0:
            print("No valid data points found in this file.")
            return

        # Count occurrences of each category
        clear_count = np.sum(valid_cmk == 0)
        cloudy_count = np.sum(valid_cmk == 1)
        prob_clear = np.sum(valid_cmk == 2)
        prob_cloudy = np.sum(valid_cmk == 3)
        
        # Define RGB colors for each category
        # FillValue (9) -> Black
        # Clear (0) -> Dark Blue
        # Cloudy (1) -> White
        # Probably Clear (2) -> Light Blue
        # Probably Cloudy (3) -> Light Gray
        
        # Create an RGB image array initialized to black
        h, w = cmk.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors
        img[cmk == 0] = [0, 0, 139]       # Dark Blue
        img[cmk == 1] = [255, 255, 255]   # White
        img[cmk == 2] = [135, 206, 235]   # Light Blue
        img[cmk == 3] = [200, 200, 200]   # Light Gray
        
        # Scale up the image (e.g. 4x) so it's not a tiny rectangle on the screen
        scale_factor = 4
        img = np.repeat(np.repeat(img, scale_factor, axis=0), scale_factor, axis=1)
        h, w = img.shape[:2]
        
        # Save alongside the original file
        output_filename = file_path.replace('.h5', '.ppm')
        with open(output_filename, 'wb') as img_file:
            img_file.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
            img_file.write(img.tobytes())
            
        print(f"-> Saved visual image to {output_filename}")

if __name__ == "__main__":
    # Find all .h5 files and analyze them
    files = sorted(glob.glob("src/mdapi/data/3RIMG_L2B_CMK/**/*.h5", recursive=True))
    if files:
        for file in files:
            parse_and_summarize(file)
    else:
        print("No .h5 files found!")
