import cv2
import numpy as np
import os
import csv

def process_coin_position_images(image_paths, input_folder):
    if not image_paths:
        print("No images to process for coin position analysis.")
        return

    output_folder = os.path.join(input_folder, "annotated_coin_position")
    csv_path = os.path.join(input_folder, "coin_positions.csv")

    # ROI box coordinates
    LEFT, RIGHT, TOP, BOTTOM = 1150, 1450, 550, 800

    # Marker drawing params
    marker_radius = 5
    marker_thickness = -1

    # Colors in BGR
    chamber_color = (0, 255, 0)  # Green
    coin_color = (0, 0, 255)     # Red
    box_color = (255, 0, 0)      # Blue
    center_marker_color = (0, 0, 255)

    # Circle detection ranges
    chamber_min_radius = 80
    chamber_max_radius = 92
    coin_min_radius = 55
    coin_max_radius = 65

    os.makedirs(output_folder, exist_ok=True)

    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'filename',
            'chamber_center_x', 'chamber_center_y', 'chamber_radius', 'chamber_detected',
            'coin_center_x', 'coin_center_y', 'coin_radius', 'coin_detected',
            'X diff (px)', 'Y diff (px)', 'mm/px', 'X diff (mm)', 'Y diff (mm)'
        ])

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read {filename}. Skipping.")
                continue

            roi = image[TOP:BOTTOM, LEFT:RIGHT]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            cv2.rectangle(image, (LEFT, TOP), (RIGHT, BOTTOM), box_color, 2)

            # Detect chamber
            chamber_circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30, minRadius=chamber_min_radius, maxRadius=chamber_max_radius
            )

            chamber_detected = False
            if chamber_circles is not None:
                chamber_circles = np.around(chamber_circles[0, :]).astype(int)
                cx, cy, cr = max(chamber_circles, key=lambda c: c[2])
                cx_full, cy_full = cx + LEFT, cy + TOP
                chamber_detected = True
                cv2.circle(image, (cx_full, cy_full), cr, chamber_color, 2)
                cv2.circle(image, (cx_full, cy_full), marker_radius, center_marker_color, marker_thickness)
            else:
                cx_full, cy_full, cr = -1, -1, -1

            # Detect coin
            coin_circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30, minRadius=coin_min_radius, maxRadius=coin_max_radius
            )

            coin_detected = False
            if coin_circles is not None:
                coin_circles = np.around(coin_circles[0, :]).astype(int)
                x2, y2, r2 = max(coin_circles, key=lambda c: c[2])
                x2_full, y2_full = x2 + LEFT, y2 + TOP
                coin_detected = True
                cv2.circle(image, (x2_full, y2_full), r2, coin_color, 2)
                cv2.circle(image, (x2_full, y2_full), marker_radius, center_marker_color, marker_thickness)
            else:
                x2_full, y2_full, r2 = -1, -1, -1

            x_diff_px = x2 - cx
            y_diff_px = y2 - cy

            mm_px = 3.0 / cr

            x_diff_mm = x_diff_px * mm_px
            y_diff_mm = y_diff_px * mm_px

            csv_writer.writerow([
                filename,
                cx_full, cy_full, cr, chamber_detected,
                x2_full, y2_full, r2, coin_detected,
                x_diff_px, y_diff_px, mm_px, x_diff_mm, y_diff_mm
            ])

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)

            print(f"{filename}: Chamber detected={chamber_detected}, Coin detected={coin_detected}")

    print(f"\nProcessing complete. CSV saved to: {csv_path}")
    print(f"Annotated images saved in: {output_folder}")
