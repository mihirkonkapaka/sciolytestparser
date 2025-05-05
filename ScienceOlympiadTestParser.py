!pip install pdf2image opencv-python-headless
!apt-get install -y poppler-utils

import os
from google.colab import drive
import numpy as np
import cv2
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
from PIL import Image

drive.mount('/content/drive')

pdf_path = '/content/drive/MyDrive/geomap.pdf'

output_folder = '/content/drive/MyDrive/Extracted'  os.makedirs(output_folder, exist_ok=True)

viz_folder = os.path.join(output_folder, 'visualizations')
diag_folder = os.path.join(output_folder, 'diagnostics')
os.makedirs(viz_folder, exist_ok=True)
os.makedirs(diag_folder, exist_ok=True)

def create_merged_pdf_image():
    """Merge all pages of the PDF into a single tall image with improved handling"""
    print("Converting PDF to images for merging...")
        pages = convert_from_path(pdf_path, 300)
    print(f"Converted {len(pages)} pages")

        width = pages[0].width
    total_height = sum(page.height for page in pages)

        merged_image = Image.new('RGB', (width, total_height + 20 * (len(pages) - 1)), (255, 255, 255))

        y_offset = 0
    for i, page in enumerate(pages):
                page.save(f"{diag_folder}/original_page_{i+1}.png")

                merged_image.paste(page, (0, y_offset))

                y_offset += page.height + 20

        merged_path = os.path.join(output_folder, "merged_pdf.png")
    merged_image.save(merged_path)
    print(f"Created merged image at {merged_path}")

    return merged_image, merged_path

def detect_and_save_boxes(image, page_identifier):
    """Detect rectangular boxes in an image and save them"""
        img_np = np.array(image)

        img_visual = img_np.copy()

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

        kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        binary_filename = f"{diag_folder}/{page_identifier}_binary.png"
    cv2.imwrite(binary_filename, binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        height, width = img_np.shape[:2]

        extracted_boxes = []

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                x, y, w, h = cv2.boundingRect(contour)

                                        if (area < 15000 or
            w < width * 0.3 or
            h < 50):
            continue

                cv2.rectangle(img_visual, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(img_visual, f"{i+1}", (x+10, y+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                box_img = img_np[y:y+h, x:x+w]
        box_filename = f"{output_folder}/question_box{i+1}.png"
        cv2.imwrite(box_filename, cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR))

        extracted_boxes.append((box_filename, x, y, w, h, i+1))

        viz_filename = f"{viz_folder}/{page_identifier}_detected.png"
    cv2.imwrite(viz_filename, cv2.cvtColor(img_visual, cv2.COLOR_RGB2BGR))

    return extracted_boxes

print("Starting question box extraction process...")

print("Creating merged PDF image...")
merged_img, merged_path = create_merged_pdf_image()

print("Processing merged image to extract question boxes...")
merged_boxes = detect_and_save_boxes(merged_img, "merged")
print(f"Extracted {len(merged_boxes)} question boxes from merged image")

summary_file = f"{output_folder}/extraction_summary.txt"
with open(summary_file, 'w') as f:
    f.write("QUESTION BOX EXTRACTION SUMMARY\n")
    f.write("==============================\n\n")
    f.write(f"Total question boxes extracted: {len(merged_boxes)}\n\n")

    for i, (filename, x, y, w, h, box_num) in enumerate(merged_boxes):
        f.write(f"Question Box {box_num}:\n")
        f.write(f"  File: {os.path.basename(filename)}\n")
        f.write(f"  Position: x={x}, y={y}, width={w}, height={h}\n\n")

print(f"\nExtraction complete!")
print(f"Question boxes saved to: {output_folder}")
print(f"Visualizations saved to: {viz_folder}")
print(f"Diagnostic images saved to: {diag_folder}")
print(f"Summary saved to: {summary_file}")