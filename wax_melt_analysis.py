import cv2
import numpy as np
import os
import csv

def process_wax_melt_images(image_paths, input_folder):
    if not image_paths:
        print("No images provided for wax melt analysis.")
        return

    # Setup output directories
    output_folder = os.path.join(input_folder, "wax_melt_analysis")
    mask_output_folder = os.path.join(input_folder, "threshold_wax")
    csv_output_path = os.path.join(input_folder, "wax_analysis.csv")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_output_folder, exist_ok=True)

    # ROI box coordinates
    LEFT, RIGHT, TOP, BOTTOM = 1150, 1450, 550, 800
    chamber_min_radius, chamber_max_radius = 80, 93
    chamber_radius_mm = 3.0
    threshold_value = 127

    # Start CSV
    with open(csv_output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'filename',
            'rect1_white_percent', 'rect1_dark_percent',
            'rect2_white_percent', 'rect2_dark_percent',
            'total_white_percent', 'total_dark_percent'
        ])

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {filename}. Skipping.")
                continue

            # Crop ROI for chamber detection
            roi = image[TOP:BOTTOM, LEFT:RIGHT]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Detect chamber
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30,
                minRadius=chamber_min_radius, maxRadius=chamber_max_radius
            )

            if circles is None:
                print(f"{filename} - Chamber not detected, skipping.")
                continue

            circle = max(np.uint16(np.around(circles[0, :])), key=lambda c: c[2])
            cx, cy, cr = circle
            cx_full, cy_full = cx + LEFT, cy + TOP
            px_per_mm = cr / chamber_radius_mm

            # Rectangle 1
            br_x = int(cx_full - 1.5 * px_per_mm)
            br_y = int(cy_full - 3.15 * px_per_mm)
            width1 = int(1.5 * px_per_mm)
            height1 = int(5.0 * px_per_mm)
            tl_x = br_x - width1
            tl_y = br_y - height1

            # Rectangle 2
            width2 = int(14.4 * px_per_mm)
            height2 = int(1.5 * px_per_mm)
            tr_x, tr_y = tl_x, tl_y
            bl_x, bl_y = tr_x - width2, tr_y + height2

            # Grayscale full image
            gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            def analyze_roi(x1, y1, x2, y2, name):
                roi = gray_full[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
                _, binary = cv2.threshold(roi, threshold_value, 255, cv2.THRESH_BINARY)
                total = binary.size
                white = cv2.countNonZero(binary)
                dark = total - white
                white_pct = white / total * 100
                dark_pct = dark / total * 100
                mask_path = os.path.join(mask_output_folder, f"{os.path.splitext(filename)[0]}_{name}.png")
                cv2.imwrite(mask_path, binary)
                return white_pct, dark_pct

            w1, d1 = analyze_roi(tl_x, tl_y, br_x, br_y, "rect1_thresh")
            w2, d2 = analyze_roi(bl_x, bl_y, tr_x, tr_y, "rect2_thresh")
            total_white = w1 + w2
            total_dark = d1 + d2

            # Annotate
            cv2.circle(image, (cx_full, cy_full), cr, (0, 255, 0), 2)
            cv2.circle(image, (cx_full, cy_full), 5, (0, 255, 0), -1)
            cv2.rectangle(image, (tl_x, tl_y), (br_x, br_y), (255, 255, 0), 2)
            cv2.rectangle(image, (bl_x, bl_y), (tr_x, tr_y), (0, 255, 255), 2)

            annotated_path = os.path.join(output_folder, filename)
            cv2.imwrite(annotated_path, image)

            csv_writer.writerow([
                filename,
                f"{w1:.2f}", f"{d1:.2f}",
                f"{w2:.2f}", f"{d2:.2f}",
                f"{total_white:.2f}", f"{total_dark:.2f}"
            ])

            print(f"{filename} - Rect1: {w1:.1f}%, Rect2: {w2:.1f}% white")

    print(f"\nWax melt analysis done! CSV: {csv_output_path}")
