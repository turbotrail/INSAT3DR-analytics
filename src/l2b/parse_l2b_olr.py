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
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def process_olr_file(file_path):
    print(f"\n--- Processing OLR: {os.path.basename(file_path)} ---")
    try:
        f = h5py.File(file_path, 'r')
    except Exception as e:
        print(f"-> Failed to open file: {e}")
        return None, None
        
    with f:
        if 'OLR' not in f:
            print("-> OLR dataset not found in file!")
            return None, None
            
        raw_data = f['OLR'][0, :, :]
        
        # OLR values typically range from 100 to 350 W/m^2
        # -999.0 is the _FillValue
        valid_mask = (raw_data > 0)
        
        # Normalize between 100 and 320 for visualization
        vmin, vmax = 100.0, 320.0
        norm_data = np.clip((raw_data - vmin) / (vmax - vmin), 0, 1)
        
        # 1. Grayscale Image (Inverted: Cold/Low OLR = White, Hot/High OLR = Black)
        gray_intensity = (1.0 - norm_data) * 255
        h, w = gray_intensity.shape
        img_gray = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            img_gray[:, :, i] = np.where(valid_mask, gray_intensity, 0)
            
        # 2. Color Image (Low OLR = Blue, High OLR = Red using 'jet' colormap)
        # Apply colormap to normalized data
        cmap = plt.get_cmap('jet')
        color_rgba = cmap(norm_data) # returns shape (h, w, 4) with values 0-1
        img_color = (color_rgba[:, :, :3] * 255).astype(np.uint8)
        # Apply mask
        for i in range(3):
            img_color[:, :, i] = np.where(valid_mask, img_color[:, :, i], 0)
            
        # 3. Add Labels
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        timestamp_str = ""
        if len(parts) >= 3:
            timestamp_str = f" | {parts[1]} {parts[2]} UTC"
            
        text = f" OLR{timestamp_str} "
        
        # Draw text
        text_img = Image.new('RGB', (len(text) * 6 + 10, 15), color=(0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((5, 2), text, fill=(255, 255, 255))
        
        # Scale text by 4x for this full size image (2816x2805)
        scale = 4
        text_img = text_img.resize((text_img.width * scale, text_img.height * scale), Image.NEAREST)
        
        # Apply to both
        for img in [img_gray, img_color]:
            pil_img = Image.fromarray(img)
            pil_img.paste(text_img, (40, 40))
            img[:] = np.array(pil_img)
            
        # Save to PPM
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
    search_path = "/Volumes/Expansion/mdapi_data/3RIMG_L2B_OLR/**/*.h5"
    files = sorted(glob.glob(search_path, recursive=True))
    
    if not files:
        print("No L2B OLR .h5 files found!")
        sys.exit(0)
        
    print(f"Found {len(files)} files to process.")
    
    gray_images = []
    color_images = []
    
    for file in files:
        gray, color = process_olr_file(file)
        if gray and color:
            gray_images.append(gray)
            color_images.append(color)
            
    if gray_images:
        gray_list = "data/movie_list_olr_gray.txt"
        with open(gray_list, "w") as f:
            for img in gray_images:
                f.write(f"file '{img}'\n")
                f.write(f"duration 0.5\n")
            f.write(f"file '{gray_images[-1]}'\n")
        create_movie(gray_list, "outputs/olr_timeline_GRAY.mp4")
        
    if color_images:
        color_list = "data/movie_list_olr_color.txt"
        with open(color_list, "w") as f:
            for img in color_images:
                f.write(f"file '{img}'\n")
                f.write(f"duration 0.5\n")
            f.write(f"file '{color_images[-1]}'\n")
        create_movie(color_list, "outputs/olr_timeline_COLOR.mp4")
