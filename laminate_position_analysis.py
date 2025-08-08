import cv2
import numpy as np
import os
import csv

def process_laminate_images(image_paths, base_folder):
    if not image_paths:
        print("No images provided for laminate analysis.")
        return

    output_folder = os.path.join(base_folder, "laminate_position")
    csv_output_path = os.path.join(base_folder, "laminate_position.csv")
    os.makedirs(output_folder, exist_ok=True)

    rois = [
        {"name": "ROI 1 - Horizontal", "left": 1300, "right": 1375, "top": 510, "bottom": 560, "orientation": "horizontal", "smooth": 9, "clahe": 4.5},
        {"name": "ROI 2 - Horizontal", "left": 320, "right": 450, "top": 950, "bottom": 1010, "orientation": "horizontal", "smooth": 5, "clahe": 3.0},
        {"name": "ROI 3 - Vertical", "left": 1100, "right": 1170, "top": 790, "bottom": 870, "orientation": "vertical", "smooth": 5, "clahe": 3.0},
        {"name": "ROI 4 - Vertical", "left": 750, "right": 810, "top": 790, "bottom": 870, "orientation": "vertical", "smooth": 5, "clahe": 3.0}
    ]

    chamber_roi = {
        "left": 1150, "right": 1450, "top": 550, "bottom": 800,
        "min_radius": 85, "max_radius": 93
    }

    box_color = (255, 0, 0)
    line_color = (0, 255, 255)
    dot_color = (0, 0, 255)
    chamber_color = (0, 255, 0)
    center_marker_color = (0, 0, 255)

    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["filename"]
        for roi in rois:
            header.append(f"{roi['name']} x")
            header.append(f"{roi['name']} y")
        header.extend([
            "chamber_center_x", "chamber_center_y", "chamber_radius",
            "pixels_per_mm",
            "vertical_distance_px", "vertical_distance_mm",
            "horizontal_distance_px", "horizontal_distance_mm"
        ])
        writer.writerow(header)

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load {filename}, skipping.")
                continue

            output = image.copy()
            row = [filename]

            for roi in rois:
                LEFT, RIGHT, TOP, BOTTOM = roi["left"], roi["right"], roi["top"], roi["bottom"]
                gray = cv2.cvtColor(image[TOP:BOTTOM, LEFT:RIGHT], cv2.COLOR_BGR2GRAY)

                bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
                clahe = cv2.createCLAHE(clipLimit=roi["clahe"], tileGridSize=(4, 4))
                contrast = clahe.apply(bilateral)

                if roi["orientation"] == "horizontal":
                    sobel = cv2.Sobel(contrast, cv2.CV_64F, 0, 1, ksize=3)
                else:
                    sobel = cv2.Sobel(contrast, cv2.CV_64F, 1, 0, ksize=3)

                abs_sobel = np.uint8(np.absolute(sobel))
                projection = np.sum(abs_sobel, axis=1 if roi["orientation"] == "horizontal" else 0).astype(np.float32)

                if roi["orientation"] == "horizontal":
                    smoothed = cv2.GaussianBlur(projection[:, np.newaxis], (1, roi["smooth"]), 0).flatten()
                    y = int(np.argmax(smoothed))
                    x = (LEFT + RIGHT) // 2
                    center = (x, y + TOP)
                    start = (LEFT, y + TOP)
                    end = (RIGHT, y + TOP)
                else:
                    smoothed = cv2.GaussianBlur(projection[np.newaxis, :], (roi["smooth"], 1), 0).flatten()
                    x = int(np.argmax(smoothed))
                    y = (TOP + BOTTOM) // 2
                    center = (x + LEFT, y)
                    start = (x + LEFT, TOP)
                    end = (x + LEFT, BOTTOM)

                cv2.rectangle(output, (LEFT, TOP), (RIGHT, BOTTOM), box_color, 1)
                cv2.line(output, start, end, line_color, 2)
                cv2.circle(output, center, 3, dot_color, -1)

                row.extend([center[0], center[1]])

            # Chamber Detection
            c_left, c_right, c_top, c_bottom = chamber_roi["left"], chamber_roi["right"], chamber_roi["top"], chamber_roi["bottom"]
            roi_crop = image[c_top:c_bottom, c_left:c_right]
            gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            cv2.rectangle(output, (c_left, c_top), (c_right, c_bottom), box_color, 1)

            chamber_circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=50, param2=30,
                minRadius=chamber_roi["min_radius"],
                maxRadius=chamber_roi["max_radius"]
            )

            if chamber_circles is not None:
                chamber_circles = np.uint16(np.around(chamber_circles[0, :]))
                cx, cy, cr = max(chamber_circles, key=lambda c: c[2])
                cx_full = cx + c_left
                cy_full = cy + c_top
                cv2.circle(output, (cx_full, cy_full), cr, chamber_color, 2)
                cv2.circle(output, (cx_full, cy_full), 3, center_marker_color, -1)
            else:
                cx_full, cy_full, cr = -1, -1, -1

            row.extend([cx_full, cy_full, cr])

            # Distance Calculations
            pixels_per_mm = cr / 3.0 if cr > 0 else -1
            roi1_y, roi2_y = row[2], row[4]
            roi3_x, roi4_x = row[5], row[7]

            vertical_distance_px = abs(roi1_y - roi2_y)
            vertical_distance_mm = vertical_distance_px / pixels_per_mm if pixels_per_mm > 0 else -1

            horizontal_distance_px = abs(roi3_x - roi4_x)
            horizontal_distance_mm = horizontal_distance_px / pixels_per_mm if pixels_per_mm > 0 else -1

            row.extend([
                pixels_per_mm,
                vertical_distance_px, vertical_distance_mm,
                horizontal_distance_px, horizontal_distance_mm
            ])

            writer.writerow(row)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_annotated.jpg")
            cv2.imwrite(output_path, output)

    print(f"Laminate analysis complete. CSV saved to: {csv_output_path}")
