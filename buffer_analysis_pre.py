import cv2
import numpy as np
import os
import csv

def process_pre_buffer_images(image_paths, input_folder):
    if not image_paths:
        print("No images provided for pre-buffer analysis.")
        return

    # Assume images are from the same folder
    output_folder = os.path.join(input_folder, "pre_buffer_levels")
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(input_folder, "pre_buffer_levels.csv")

    # ROIs and constants
    mid_chamber_roi = (200, 400, 630, 780)    # (y1, y2, x1, x2)
    feature_roi = (950, 1070, 800, 1100)
    true_feature_height_mm = 1.369

    results = [(
        "filename",
        "liquid_y",
        "feature_top_y", "feature_bottom_y", "feature_height_px",
        "delta_y", "delta_mm"
    )]

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
        # zero out around top_local to find second strongest peak
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

        liquid_y = detect_horizontal_line(gray, mid_chamber_roi)
        feature_height_px, feature_top_y, feature_bottom_y = detect_feature_height(gray, feature_roi)

        delta_y = feature_top_y - liquid_y
        pixels_per_mm = feature_height_px / true_feature_height_mm if feature_height_px else None
        delta_mm = delta_y / pixels_per_mm if pixels_per_mm else "NA"

        # Annotate image
        annotated = img.copy()
        cv2.rectangle(annotated, (mid_chamber_roi[2], mid_chamber_roi[0]), (mid_chamber_roi[3], mid_chamber_roi[1]), (0, 255, 0), 2)
        cv2.rectangle(annotated, (feature_roi[2], feature_roi[0]), (feature_roi[3], feature_roi[1]), (0, 255, 0), 2)
        cv2.line(annotated, (mid_chamber_roi[2], liquid_y), (mid_chamber_roi[3], liquid_y), (0, 0, 255), 2)
        cv2.line(annotated, (feature_roi[2], feature_top_y), (feature_roi[3], feature_top_y), (0, 165, 255), 2)
        cv2.line(annotated, (feature_roi[2], feature_bottom_y), (feature_roi[3], feature_bottom_y), (0, 165, 255), 2)

        out_path = os.path.join(output_folder, f"annotated_{filename}")
        cv2.imwrite(out_path, annotated)

        results.append((
            filename,
            liquid_y,
            feature_top_y, feature_bottom_y, feature_height_px,
            delta_y, delta_mm
        ))

        print(f"{filename} processed.")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"Pre-buffer analysis done! Results saved to {csv_path}")
