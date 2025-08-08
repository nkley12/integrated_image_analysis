import cv2
import numpy as np
import os
import csv

def process_pmps_images(image_paths, base_folder):
    if not image_paths:
        print("No images provided for PMPS analysis.")
        return

    output_csv = os.path.join(base_folder, "pmp_analysis.csv")
    mask_output_folder = os.path.join(base_folder, "thresholded_pmps")
    annotated_output_folder = os.path.join(base_folder, "pmps_in_chamber")
    os.makedirs(mask_output_folder, exist_ok=True)
    os.makedirs(annotated_output_folder, exist_ok=True)

    LEFT, RIGHT, TOP, BOTTOM = 1150, 1450, 550, 850

    chamber_min_radius = 83
    chamber_max_radius = 100
    dp = 1.2
    min_dist = 50
    param1 = 50
    param2 = 30

    lower_bound = np.array([0, 79, 72])
    upper_bound = np.array([255, 255, 142])

    results = [(
        "Filename",
        "Chamber center X", "Chamber center Y", "Chamber radius", "Chamber detected?",
        "Total chamber area (px)",
        "PMPs in chamber (px)",
        "Percent PMP in chamber area",
        "Percent total PMP area"
    )]

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {filename}. Skipping.")
            continue

        cv2.rectangle(image, (LEFT, TOP), (RIGHT, BOTTOM), (255, 0, 0), 2)
        roi = image[TOP:BOTTOM, LEFT:RIGHT]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        chamber_detected = False
        chamber_mask = None
        cx_full, cy_full, cr = -1, -1, -1

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
            param1=param1, param2=param2,
            minRadius=chamber_min_radius, maxRadius=chamber_max_radius
        )

        if circles is not None:
            chamber_detected = True
            circle = np.uint16(np.around(circles[0, 0]))
            cx, cy, cr = circle
            cx_full = cx + LEFT
            cy_full = cy + TOP

            cv2.circle(image, (cx_full, cy_full), cr, (0, 255, 0), 2)
            cv2.circle(image, (cx_full, cy_full), 4, (0, 0, 255), -1)

            chamber_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(chamber_mask, (cx_full, cy_full), cr, 255, -1)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        threshold_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        mask_filename = os.path.splitext(filename)[0] + "_mask.png"
        cv2.imwrite(os.path.join(mask_output_folder, mask_filename), threshold_mask)

        if chamber_detected:
            threshold_in_chamber = cv2.bitwise_and(threshold_mask, threshold_mask, mask=chamber_mask)
            chamber_area_px = int(np.count_nonzero(chamber_mask))
            thresholded_px = int(np.count_nonzero(threshold_in_chamber))
            percent_area = (thresholded_px / chamber_area_px) * 100 if chamber_area_px > 0 else 0

            threshold_roi_rect = threshold_mask[TOP:BOTTOM, LEFT:RIGHT]
            roi_threshold_px = int(np.count_nonzero(threshold_roi_rect))
            roi_vs_chamber_ratio = (roi_threshold_px / chamber_area_px) * 100 if chamber_area_px > 0 else 0
        else:
            chamber_area_px = 0
            thresholded_px = 0
            percent_area = 0
            roi_vs_chamber_ratio = 0

        annotated_path = os.path.join(annotated_output_folder, filename)
        cv2.imwrite(annotated_path, image)

        results.append((
            filename,
            cx_full, cy_full, cr, chamber_detected,
            chamber_area_px,
            thresholded_px,
            percent_area,
            roi_vs_chamber_ratio
        ))

    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"\nPMPS analysis done!")
    print(f"CSV saved to: {output_csv}")
    print(f"Masks saved in: {mask_output_folder}")
    print(f"Annotated images saved in: {annotated_output_folder}")