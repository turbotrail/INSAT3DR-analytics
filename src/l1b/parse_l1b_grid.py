# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "h5py",
#     "numpy",
#     "pillow",
# ]
# ///
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import numpy as np
import glob
import subprocess
import json
import sys
from datetime import datetime, timedelta

def get_channel_image(f, channel, file_path):
    channel_upper = channel.upper()
    
    # 1. Read Raw Data and Downsample to 704 x 701
    # Standard resolution is 2816x2805. We downsample by 4 to get 704x701.
    # VIS is 11264x11220. We downsample by 16 to get 704x701.
    
    if channel_upper == "VIS":
        raw_data = f['IMG_VIS'][0, ::16, ::16]
        lookup = f['IMG_VIS_ALBEDO'][:]
        physical_data = lookup[raw_data][:704, :701]
        valid_mask = (physical_data >= 0) & (physical_data <= 100)
        norm_data = np.clip(physical_data / 100.0, 0, 1)
        intensity = norm_data * 255
    elif channel_upper in ["TIR1", "TIR2", "MIR"]:
        raw_data = f[f'IMG_{channel_upper}'][0, ::4, ::4]
        lookup = f[f'IMG_{channel_upper}_TEMP'][:]
        physical_data = lookup[raw_data][:704, :701]
        valid_mask = (physical_data > 100) & (physical_data < 400)
        norm_data = np.clip((physical_data - 200) / (320 - 200), 0, 1)
        intensity = (1.0 - norm_data) * 255
    elif channel_upper == "WV":
        raw_data = f['IMG_WV'][0, ::2, ::2]
        lookup = f['IMG_WV_TEMP'][:]
        physical_data = lookup[raw_data][:704, :701]
        valid_mask = (physical_data > 100) & (physical_data < 400)
        norm_data = np.clip((physical_data - 200) / (320 - 200), 0, 1)
        intensity = (1.0 - norm_data) * 255
    elif channel_upper == "SWIR":
        raw_data = f['IMG_SWIR'][0, ::16, ::16]
        lookup = f['IMG_SWIR_RADIANCE'][:]
        physical_data = lookup[raw_data][:704, :701]
        valid_mask = physical_data >= 0
        norm_data = np.clip(physical_data / 10.0, 0, 1)
        intensity = norm_data * 255
    else:
        return None

    # 2. Generate Image Array
    h, w = physical_data.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.where(valid_mask, intensity, 0)
    img[:, :, 1] = np.where(valid_mask, intensity, 0)
    img[:, :, 2] = np.where(valid_mask, intensity, 100)
    
    # 3. Add timestamp and channel label using Pillow
    from PIL import Image, ImageDraw
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    timestamp_str = ""
    if len(parts) >= 3:
        timestamp_str = f" | {parts[1]} {parts[2]} UTC"
        
    text = f" {channel_upper}{timestamp_str} "
    
    # Draw text
    text_img = Image.new('RGB', (len(text) * 6 + 10, 15), color=(0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((5, 2), text, fill=(255, 255, 255))
    
    # Scale text by 2x for this smaller image size
    scale = 2
    text_img = text_img.resize((text_img.width * scale, text_img.height * scale), Image.NEAREST)
    
    pil_img = Image.fromarray(img)
    pil_img.paste(text_img, (20, 20))
    img = np.array(pil_img)
        
    return img

def parse_and_summarize_l1b_grid(file_path):
    print(f"\n--- Processing Grid for: {os.path.basename(file_path)} ---")
    try:
        f = h5py.File(file_path, 'r')
    except Exception as e:
        print(f"-> Failed to open file: {e}")
        return None
        
    with f:
        channels = ["VIS", "SWIR", "MIR", "WV", "TIR1", "TIR2"]
        images = []
        
        for ch in channels:
            img = get_channel_image(f, ch, file_path)
            if img is not None:
                images.append(img)
            else:
                # Fallback blank image
                images.append(np.zeros((704, 701, 3), dtype=np.uint8))
                
        # We have exactly 6 channels (VIS, SWIR, MIR, WV, TIR1, TIR2).
        # Create a 2x3 grid (2 rows, 3 columns).
        
        # Row 1: VIS, SWIR, MIR
        row1 = np.concatenate(images[0:3], axis=1)
        # Row 2: WV, TIR1, TIR2
        row2 = np.concatenate(images[3:6], axis=1)
        
        # Combine rows (2x3 grid)
        grid_img = np.concatenate([row1, row2], axis=0)
        
        h, w = grid_img.shape[:2]
        output_filename = file_path.replace('.h5', '_GRID.ppm')
        
        with open(output_filename, 'wb') as img_file:
            img_file.write(f"P6\n{w} {h}\n255\n".encode('ascii'))
            img_file.write(grid_img.tobytes())
            
        print(f"-> Saved grid image to {output_filename}")
        return output_filename

if __name__ == "__main__":
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
                files = sorted(glob.glob(os.path.join(data_dir, "**/*.h5"), recursive=True))
            except subprocess.CalledProcessError as e:
                print(f"Error downloading data with mdapi: {e}")
        else:
            print("\n--- Data is up to date. Skipping download ---")
    else:
        print(f"\n--- Config file {config_path} not found. Skipping auto-download ---")

    if len(files) > 48:
        files = files[-48:]

    output_images = []
    if files:
        for file in files:
            img = parse_and_summarize_l1b_grid(file)
            if img:
                output_images.append(img)
                
        if output_images:
            list_file = "data/movie_list_grid.txt"
            with open(list_file, "w") as f:
                for img in output_images:
                    f.write(f"file '{img}'\n")
                    f.write(f"duration 0.5\n")
            if output_images:
                with open(list_file, "a") as f:
                    f.write(f"file '{output_images[-1]}'\n")
                    
            output_movie = "outputs/full_disk_timeline_GRID.mp4"
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
