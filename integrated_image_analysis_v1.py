import os
#import sys

#sys.path.insert(0, os.path.abspath("/Users/natalie/projects/integrated_image_analysis"))

# === SET YOUR INPUT DIRECTORY HERE ===
input_folder = "//nuc-fs1/Engineering/Grant/DASH/General Cartridge Run Videos/QC testing cartridge pics/RDCE_NEG_23JUL25"

# === CATEGORIZE IMAGES ===
images = {
    "pre_coins": [],
    "post_coins": [],
    "pre_buffers": [],
    "post_buffers": []
}

# Categorize based on keywords in filenames or folder names
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            continue
        fpath = os.path.join(root, file)
        lower_path = fpath.lower()

        if "pre coins" in lower_path:
            images["pre_coins"].append(fpath)
        elif "post coins" in lower_path:
            images["post_coins"].append(fpath)
        elif "pre buffers" in lower_path:
            images["pre_buffers"].append(fpath)
        elif "post buffers" in lower_path:
            images["post_buffers"].append(fpath)

# === ANALYSIS FUNCTIONS ===
from coin_position_analysis import process_coin_position_images
from laminate_position_analysis import process_laminate_images
from pmps_analysis import process_pmps_images
from wax_melt_analysis import process_wax_melt_images
from buffer_analysis_pre import process_pre_buffer_images
from buffer_analysis_post import process_post_buffer_images

# === RUN ANALYSES ON GROUPED IMAGES ===
if images["pre_coins"]:
    process_coin_position_images(images["pre_coins"], input_folder)
    process_laminate_images(images["pre_coins"], input_folder)
if images["post_coins"]:
    process_pmps_images(images["post_coins"], input_folder)
    process_wax_melt_images(images["post_coins"], input_folder)
if images["pre_buffers"]:
    process_pre_buffer_images(images["pre_buffers"], input_folder)
if images["post_buffers"]:
    process_post_buffer_images(images["post_buffers"], input_folder)
