# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "h5py",
#     "numpy",
#     "pillow",
#     "matplotlib",
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

def process_pet_file(file_path):
    print(f"\n--- Processing PET: {os.path.basename(file_path)} ---")
    try:
        f = h5py.File(file_path, 'r')
    except Exception as e:
        print(f"-> Failed to open file: {e}")
        return None, None
        
    with f:
        # Assuming the dataset name is 'PET' or similar. Check for common names.
        dataset_name = None
        for key in f.keys():
            if 'PET' in key.upper():
                dataset_name = key
                break
        
        if not dataset_name:
            print(f"-> PET dataset not found in file! Available keys: {list(f.keys())}")
            return None, None
            
        raw_data = f[dataset_name][0, :, :] if len(f[dataset_name].shape) == 3 else f[dataset_name][:, :]
        
        # STEP 1: Define limits and mask invalid data
        # PET values usually range from 0 to 15 mm/day. 
        # We mask out invalid/fill values (like -999) using a boolean mask.
        valid_mask = (raw_data >= 0) & (raw_data < 100)
        
        # STEP 2: Normalize the data
        # We map the scientific values (0.0 to 15.0) down to a float between 0.0 and 1.0. 
        # np.clip ensures anything above 15.0 stays at 1.0, and below 0.0 stays at 0.0.
        vmin, vmax = 0.0, 15.0
        norm_data = np.clip((raw_data - vmin) / (vmax - vmin), 0, 1)
        
        # STEP 3: Create Grayscale Image
        # Multiply the 0.0-1.0 floats by 255 to map them to standard 8-bit image pixels (0-255).
        gray_intensity = norm_data * 255
        h, w = gray_intensity.shape
        img_gray = np.zeros((h, w, 3), dtype=np.uint8) # Create a blank 3-channel (RGB) image matrix
        for i in range(3):
            # Apply the intensity to R, G, and B channels equally to get gray. 
            # If the pixel was invalid in the mask, set it to 0 (black).
            img_gray[:, :, i] = np.where(valid_mask, gray_intensity, 0)
            
        # STEP 4: Create Color Image 
        # Pass the 0.0-1.0 float data through a Matplotlib Colormap ('jet').
        # This converts the single float value into 4 floats representing an RGBA color.
        cmap = plt.get_cmap('jet')
        color_rgba = cmap(norm_data) 
        
        # Discard the Alpha channel, leaving RGB, and scale to 0-255 uint8 integers.
        img_color = (color_rgba[:, :, :3] * 255).astype(np.uint8)
        for i in range(3):
            # Mask out invalid points to black
            img_color[:, :, i] = np.where(valid_mask, img_color[:, :, i], 0)
            
        # 3. Add Labels
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        timestamp_str = ""
        if len(parts) >= 3:
            timestamp_str = f" | {parts[1]} {parts[2]} UTC"
            
        text = f" PET{timestamp_str} "
        
        text_img = Image.new('RGB', (len(text) * 6 + 10, 15), color=(0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((5, 2), text, fill=(255, 255, 255))
        
        scale = 4
        text_img = text_img.resize((text_img.width * scale, text_img.height * scale), Image.NEAREST)
        
        for img in [img_gray, img_color]:
            pil_img = Image.fromarray(img)
            pil_img.paste(text_img, (40, 40))
            img[:] = np.array(pil_img)
            
        out_gray = file_path.replace('.h5', '_GRAY.ppm')
        out_color = file_path.replace('.h5', '_COLOR.ppm')
        
        header = f"P6\n{w} {h}\n255\n".encode('ascii')
        with open(out_gray, 'wb') as img_file:
            img_file.write(header)
            img_file.write(img_gray.tobytes())
            
        with open(out_color, 'wb') as img_file:
            img_file.write(header)
            img_file.write(img_color.tobytes())
            
        print(f"-> Saved: {os.path.basename(out_gray)}")
        print(f"-> Saved: {os.path.basename(out_color)}")
        
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
    search_path = r"X:\mdapi_data\3RIMG_L3C_PET_DLY\**\*.h5"
    
    def extract_datetime(filepath):
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        if len(parts) >= 3:
            try:
                return datetime.strptime(f"{parts[1]}_{parts[2]}", "%d%b%Y_%H%M")
            except Exception:
                pass
        return datetime.min

    files = glob.glob(search_path, recursive=True)
    files.sort(key=extract_datetime)
    
    if not files:
        print("No L3C PET DLY .h5 files found!")
        sys.exit(0)
        
    print(f"Found {len(files)} files to process.")
    
    gray_images = []
    color_images = []
    
    for file in files:
        gray, color = process_pet_file(file)
        if gray and color:
            gray_images.append(gray)
            color_images.append(color)
            
    if gray_images:
        gray_list = "data/movie_list_pet_gray.txt"
        with open(gray_list, "w") as f:
            for img in gray_images:
                f.write(f"file '{img}'\n")
                f.write(f"duration 0.5\n")
            f.write(f"file '{gray_images[-1]}'\n")
        create_movie(gray_list, "outputs/pet_timeline_GRAY.mp4")
        
    if color_images:
        color_list = "data/movie_list_pet_color.txt"
        with open(color_list, "w") as f:
            for img in color_images:
                f.write(f"file '{img}'\n")
                f.write(f"duration 0.5\n")
            f.write(f"file '{color_images[-1]}'\n")
        create_movie(color_list, "outputs/pet_timeline_COLOR.mp4")
