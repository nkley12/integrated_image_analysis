import cv2
import numpy as np
import os
import csv

def process_post_buffer_images(image_paths, input_folder):
    if not image_paths:
        print("No images provided for post-buffer analysis.")
        return

    output_folder = os.path.join(input_folder, "post_buffer_levels")
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(input_folder, "post_buffer_levels.csv")

    # ROI definitions: (y_start, y_end, x_start, x_end)
    left_mid_roi = (450, 700, 630, 780)
    right_mid_roi = (90, 280, 1250, 1460)
    feature_roi = (950, 1070, 800, 1100)  # white feature ROI for normalization

    true_feature_height_mm = 1.369  # known height of white feature in mm

    results = [
        ("filename",
         "C2_liquid_y", "C1_liquid_y",
         "feature_top_y", "feature_bottom_y", "feature_height_px",
         "C2_height", "C2_height_mm",
         "C1_height", "C1_height_mm")
    ]

    def detect_horizontal_line(gray, roi):
        y1, y2, x1, x2 = roi
        crop = gray[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(crop, (5, 5), 0)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)
        row_strength = np.sum(sobel_y_abs, axis=1)
        local_y = np.argmax(row_strength)
        return y1 + local_y

    def detect_feature_height(gray, roi):
        y1, y2, x1, x2 = roi
        crop = gray[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(crop, (5, 5), 0)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, dx=0, dy=1, ksize=3)
        sobel_y_abs = cv2.convertScaleAbs(sobel_y)
        row_strength = np.sum(sobel_y_abs, axis=1)

        top_local = np.argmax(row_strength)
        # suppress values near top_local to find second peak
        window = 10
        start = max(top_local - window, 0)
        end = min(top_local + window, len(row_strength))
        row_strength[start:end] = 0
        bottom_local = np.argmax(row_strength)

        top_y = y1 + min(top_local, bottom_local)
        bottom_y = y1 + max(top_local, bottom_local)
        height_px = bottom_y - top_y
        return height_px, top_y, bottom_y

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {filename}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        annotated = img.copy()

        left_liquid_y = detect_horizontal_line(gray, left_mid_roi)
        right_liquid_y = detect_horizontal_line(gray, right_mid_roi)
        feature_height_px, feature_top_y, feature_bottom_y = detect_feature_height(gray, feature_roi)

        pixels_per_mm = feature_height_px / true_feature_height_mm if feature_height_px else None

        left_delta_y = feature_top_y - left_liquid_y
        right_delta_y = feature_top_y - right_liquid_y
        left_delta_mm = left_delta_y / pixels_per_mm if pixels_per_mm else "NA"
        right_delta_mm = right_delta_y / pixels_per_mm if pixels_per_mm else "NA"

        # Draw ROIs
        for roi in [left_mid_roi, right_mid_roi, feature_roi]:
            y1, y2, x1, x2 = roi
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw detected lines
        line_specs = [
            (left_liquid_y, left_mid_roi, (0, 0, 255)),       # Red
            (right_liquid_y, right_mid_roi, (0, 255, 255)),   # Yellow
            (feature_top_y, feature_roi, (0, 165, 255)),      # Orange top
            (feature_bottom_y, feature_roi, (0, 165, 255))    # Orange bottom
        ]
        for y, roi, color in line_specs:
            x1, x2 = roi[2], roi[3]
            cv2.line(annotated, (x1, y), (x2, y), color, 2)

        out_path = os.path.join(output_folder, f"annotated_{filename}")
        cv2.imwrite(out_path, annotated)

        results.append((
            filename,
            left_liquid_y, right_liquid_y,
            feature_top_y, feature_bottom_y, feature_height_px,
            left_delta_y, left_delta_mm,
            right_delta_y, right_delta_mm
        ))

        print(f"{filename} processed.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"Post-buffer analysis done! Results saved to {csv_path}")
