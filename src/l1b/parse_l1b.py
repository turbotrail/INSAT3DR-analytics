# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "h5py",
#     "numpy",
#     "pillow",
# ]
# ///
import h5py
import numpy as np
import glob
import os

def parse_and_summarize_l1b(file_path, channel="VIS"):
    print(f"\n--- Analyzing {channel.upper()} Data: {os.path.basename(file_path)} ---")
    try:
        f = h5py.File(file_path, 'r')
    except Exception as e:
        print(f"-> Failed to open file (might be incomplete/corrupted): {e}")
        return None
        
    with f:
        
        # 1. Read Latitude and Longitude (Standard Resolution: 2816 x 2805)
        # Note: We must apply the scale_factor and handle fill values.
        lat_scale = f['Latitude'].attrs.get('scale_factor', [1.0])[0]
        lon_scale = f['Longitude'].attrs.get('scale_factor', [1.0])[0]
        lat_fill = f['Latitude'].attrs.get('_FillValue', [32767])[0]
        lon_fill = f['Longitude'].attrs.get('_FillValue', [32767])[0]
        
        lat = f['Latitude'][:]
        lon = f['Longitude'][:]
        
        # Convert to float and apply scale factor
        lat = np.where(lat != lat_fill, lat * lat_scale, np.nan)
        lon = np.where(lon != lon_fill, lon * lon_scale, np.nan)
        
        # 2. Use Full Disk Data (No Bounding Box)
        
        # 3. Read Raw Data and map to Physical Values based on Channel
        channel_upper = channel.upper()
        if channel_upper == "VIS":
            # VIS resolution is 11264 x 11220. Downsample by 4 to match TIR size (2816 x 2805)
            raw_data = f['IMG_VIS'][0, ::4, ::4]
            lookup = f['IMG_VIS_ALBEDO'][:]
            physical_data = lookup[raw_data]
            valid_mask = (physical_data >= 0) & (physical_data <= 100)
            unit = "% Albedo"
            # Normalize for visualization (0 to 1) -> 0% to 100%
            norm_data = np.clip(physical_data / 100.0, 0, 1)
            intensity = norm_data * 255 # Bright clouds
        elif channel_upper in ["TIR1", "TIR2", "MIR", "WV"]:
            # Thermal/Infrared channels are 2816 x 2805
            raw_data = f[f'IMG_{channel_upper}'][0, :, :]
            lookup = f[f'IMG_{channel_upper}_TEMP'][:]
            physical_data = lookup[raw_data]
            valid_mask = (physical_data > 100) & (physical_data < 400)
            unit = "K"
            # Normalize between roughly 200K and 320K, invert for visualization (cold=bright)
            norm_data = np.clip((physical_data - 200) / (320 - 200), 0, 1)
            intensity = (1.0 - norm_data) * 255
        elif channel_upper == "SWIR":
            # SWIR is 2816 x 2805, uses radiance
            raw_data = f['IMG_SWIR'][0, :, :]
            lookup = f['IMG_SWIR_RADIANCE'][:]
            physical_data = lookup[raw_data]
            valid_mask = physical_data >= 0
            unit = "Radiance"
            # Normalize for visualization (0 to ~10, based on max expected radiance)
            norm_data = np.clip(physical_data / 10.0, 0, 1)
            intensity = norm_data * 255
        else:
            print(f"Unsupported channel: {channel}")
            return None
        
        valid_physical = physical_data[valid_mask]
        
        if valid_physical.size == 0:
            print(f"No valid data in this file for channel {channel_upper}.")
            return None
            
        avg_val = np.mean(valid_physical)
        min_val = np.min(valid_physical)
        max_val = np.max(valid_physical)
        
        print(f"\n{channel_upper} Statistics for Full Disk:")
        print(f"  Average: {avg_val:.2f} {unit}")
        print(f"  Minimum: {min_val:.2f} {unit}")
        print(f"  Maximum: {max_val:.2f} {unit}")

        # 5. Generate a Visual Image (PPM format)
        h, w = physical_data.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply to RGB channels (making it grayscale)
        # Mask out invalid pixels (make them dark blue)
        img[:, :, 0] = np.where(valid_mask, intensity, 0)
        img[:, :, 1] = np.where(valid_mask, intensity, 0)
        img[:, :, 2] = np.where(valid_mask, intensity, 100)
        
        # Add timestamp and channel label using Pillow
        try:
            from PIL import Image, ImageDraw
            
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            timestamp_str = ""
            if len(parts) >= 3:
                timestamp_str = f" | {parts[1]} {parts[2]} UTC"
                
            text = f" INSAT-3D {channel_upper}{timestamp_str} "
            
            # Draw text on a small image and resize it to make it legible
            text_img = Image.new('RGB', (len(text) * 6 + 10, 15), color=(0, 0, 0))
            text_draw = ImageDraw.Draw(text_img)
            text_draw.text((5, 2), text, fill=(255, 255, 255))
            
            # The main image is large (~2800x2800), scale text up by 6x
            scale = 6
            text_img = text_img.resize((text_img.width * scale, text_img.height * scale), Image.NEAREST)
            
            pil_img = Image.fromarray(img)
            pil_img.paste(text_img, (50, 50))
            img = np.array(pil_img)
        except ImportError:
            print("Pillow (PIL) not installed. Skipping timestamp overlay. Run with 'uv run parse_l1b.py' to auto-install.")
        
        h, w = img.shape[:2]
        
        # Save alongside the original file
        output_filename = file_path.replace('.h5', f'_{channel_upper}.ppm')
        with open(output_filename, 'wb') as img_file:
            img_file.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
            img_file.write(img.tobytes())
            
        print(f"-> Saved {channel_upper} image to {output_filename}")
        return output_filename

if __name__ == "__main__":
    import argparse
    import subprocess
    import json
    import sys
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser(description="Parse INSAT-3D L1B data")
    parser.add_argument("--channel", type=str, default="VIS", choices=["VIS", "TIR1", "TIR2", "MIR", "WV", "SWIR"], help="Channel to parse and visualize")
    args = parser.parse_args()
    
    config_path = "src/mdapi/config.json"
    data_dir = "/Volumes/Expansion/mdapi_data/3RIMG_L1B_STD"
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True))
    needs_download = True
    
    if os.path.exists(config_path):
        if files:
            latest_file = files[-1]
            filename = os.path.basename(latest_file)
            try:
                parts = filename.split('_')
                if len(parts) >= 3:
                    date_str = parts[1]
                    time_str = parts[2]
                    file_dt = datetime.strptime(f"{date_str}_{time_str}", "%d%b%Y_%H%M")
                    # Check if latest file is within the last 4 hours
                    if datetime.utcnow() - file_dt < timedelta(hours=4):
                        needs_download = False
            except Exception as e:
                print(f"Error parsing date from {filename}: {e}")
                
        if needs_download:
            print(f"\n--- Latest data not found or is too old. Invoking mdapi to download last 24 hours to {data_dir} ---")
            with open(config_path, "r") as f:
                config = json.load(f)
                
            now = datetime.utcnow()
            yesterday = now - timedelta(days=1)
            
            if "search_parameters" not in config:
                config["search_parameters"] = {}
            config["search_parameters"]["startTime"] = yesterday.strftime("%Y-%m-%d")
            config["search_parameters"]["endTime"] = now.strftime("%Y-%m-%d")
            config["search_parameters"]["count"] = "48"
            
            if "download_settings" not in config:
                config["download_settings"] = {}
            config["download_settings"]["skip_user_input"] = True
            config["download_settings"]["download_path"] = "/Volumes/Expansion/mdapi_data"
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
                
            try:
                subprocess.run([sys.executable, "mdapi.py"], cwd="src/mdapi", check=True)
                print("--- Download complete ---")
                
                # Re-evaluate files after download
                files = sorted(glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True))
            except subprocess.CalledProcessError as e:
                print(f"Error downloading data with mdapi: {e}")
        else:
            print("\n--- Data is up to date. Skipping download ---")
    else:
        print(f"\n--- Config file {config_path} not found. Skipping auto-download ---")

    # Limit to latest 48 files for a 24-hour timeline
    if len(files) > 48:
        files = files[-48:]

    output_images = []
    if files:
        for file in files:
            img = parse_and_summarize_l1b(file, channel=args.channel)
            if img:
                output_images.append(img)
                
        if output_images:
            list_file = "data/movie_list.txt"
            with open(list_file, "w") as f:
                for img in output_images:
                    f.write(f"file '{img}'\n")
                    f.write(f"duration 0.5\n")
            if output_images:
                with open(list_file, "a") as f:
                    f.write(f"file '{output_images[-1]}'\n")
                    
            output_movie = f"outputs/full_disk_timeline_{args.channel.upper()}.mp4"
            print(f"\nCreating movie {output_movie} from {len(output_images)} frames...")
            
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
    else:
        print("No L1B .h5 files found!")
