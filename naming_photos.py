import os
import re
import easyocr
from pathlib import Path
import shutil


#iterates over all images in  folder, groups them into groups of four, gets serial number from first in group
#saves serial number/checks if repeat, then names photos pre/post label, beads, coins, buffers


image_folder = "//nuc-fs1/Engineering/Grant/DASH/General Cartridge Run Videos/QC testing cartridge pics/QCBA-06AUG25/G4"
image_extensions = ('.jpg', '.jpeg', '.png')       #these are saved as .jpg

# Initialize OCR reader
reader = easyocr.Reader(['en'])

#track and store serial numbers we have seen
seen_serials = set()

#sort images by name - works with default names from camera
def sorted_image_list(folder):
    return sorted(
        [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]
    )

# Rename images in groups of 4
def process_image_groups(folder):
    image_files = sorted_image_list(folder)
    
    #define each group of four where first pic is the label
    for i in range(0, len(image_files), 4):
        group = image_files[i:i+4]
        if len(group) < 4:
            print(f"Skipping incomplete group at end: {group}")
            continue
        
        first_image_path = os.path.join(folder, group[0])
        result = reader.readtext(first_image_path)

        #find four character strings from first image of the group of four
        four_letter_words = []
        for _, text, _ in result:
            cleaned = text.upper()
            matches = re.findall(r'\b[A-Z]{4}\b', cleaned)
            four_letter_words.extend(matches)
        
        if not four_letter_words:
            print(f"No 4-letter words found in: {group[0]}")
            continue
        
        serial = four_letter_words[-1]  #use the last 4-letter "word" found in pic as serial number
        
        #have we seen this serial number before
        is_repeat = serial in seen_serials
        if not is_repeat:
            seen_serials.add(serial)

        print(f"{'Repeat' if is_repeat else 'First'} use of serial: {serial}")

        time = "post " if is_repeat else "pre "    #post if repeated, pre if not
        labels = ["label", "beads", "coins", "buffers"]    #need to change order depending on how we take pics
        for filename, label in zip(group, labels):
            old_path = os.path.join(folder, filename)
            ext = Path(filename).suffix
            new_filename = f"{serial} {time}{label}{ext}"    #serial num, pre/post, label, filetype
            new_path = os.path.join(folder, new_filename)
            shutil.move(old_path, new_path)
            print(f"Renamed: {filename} → {new_filename}")     #prints all old/new names to check


process_image_groups(image_folder)

