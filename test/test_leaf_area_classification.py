# src/leaf_area_classification.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name: Leaf Area Classification

Description:
------------
This module processes scanned images of leaves to estimate their area.
It applies a series of image processing techniques to detect and measure leaf regions,
and outputs the results both as annotated images and as a CSV summary file.

Features:
---------
- Configurable cropping based on percentage inputs.
- Optional contrast adjustment using CLAHE.
- Adaptive thresholding in RGB space to exclude near-white regions.
- Noise reduction with Gaussian Blur.
- Adaptive thresholding and morphological operations for binary image creation.
- Contour detection to identify individual leaves.
- Area calculation in square millimeters.
- Mean RGB value calculation within each leaf contour.
- Annotation of leaves with unique identifiers.
- Generation of a comprehensive results table.
- Logging of processing steps and errors.

Usage:
------
1. Configure the parameters in `config/config.ini`.
2. Run the script:
       python leaf_area_classification.py

Dependencies:
-------------
- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- pandas
- Pillow (`PIL`)
- configparser
- logging
- threading

Configuration:
--------------
The script reads configuration parameters from `config/config.ini`. Ensure this file is properly
configured before running the script. Key parameters include:

- `image_directory`: Path to the directory containing leaf images.
- `area_threshold`: Minimum area (in mm²) to consider a contour as a valid leaf.
- `skip_contrast_adjustment`: Boolean flag to skip contrast adjustment.
- `crop_left`, `crop_right`, `crop_top`, `crop_bottom`: Percentage values for cropping.
- `filename`: Specific filename to process (optional).
- `img_debug`: Boolean flag to save intermediate processing images for debugging.
- `adaptive_threshold`: Boolean flag to enable adaptive thresholding.
- `adaptive_window_size`: Window size for adaptive thresholding (must be odd).
- `adaptive_C`: Constant subtracted from mean in adaptive thresholding.
- `color_threshold`: RGB value threshold to identify near-white pixels.

Author:
-------
Rasmus Jensen
raje at ecos.au.dk

Date:
-----
Refined on October 22, 2024

Version:
--------
0.2.0

License:
--------
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""

import os
import time
import logging
import configparser
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import re
import threading

# Configure logging for debugging and information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Configuration dataclass to hold all necessary parameters.

    Attributes:
        image_directory (str): Path to the directory containing leaf images.
        area_threshold (float): Minimum area (in mm²) to consider a contour as a valid leaf.
        skip_contrast_adjustment (bool): Flag to skip contrast adjustment.
        crop_left_pct (float): Percentage to crop from the left.
        crop_right_pct (float): Percentage to crop from the right.
        crop_top_pct (float): Percentage to crop from the top.
        crop_bottom_pct (float): Percentage to crop from the bottom.
        filename (str): Specific filename to process (optional).
        img_debug (bool): Flag to save intermediate processing images for debugging.
        adaptive_threshold (bool): Flag to enable adaptive thresholding.
        adaptive_window_size (int): Size of the neighborhood window for adaptive thresholding (must be odd).
        adaptive_C (int): Constant subtracted from the local mean in adaptive thresholding.
        color_threshold (int): RGB value threshold to identify near-white pixels.
    """
    image_directory: str
    area_threshold: float
    skip_contrast_adjustment: bool
    crop_left_pct: float
    crop_right_pct: float
    crop_top_pct: float
    crop_bottom_pct: float
    filename: str
    img_debug: bool
    adaptive_threshold: bool
    adaptive_window_size: int
    adaptive_C: int
    color_threshold: int


def read_config(config_file: str = 'config.ini') -> Config:
    """
    Read configuration parameters from a config file.

    Args:
        config_file (str, optional): Path to the configuration file. Defaults to 'config.ini'.

    Returns:
        Config: Config dataclass with all parameters.
    """
    # Determine the path to the config.ini file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', config_file)

    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)
    config = config_parser['DEFAULT']

    return Config(
        image_directory=config.get('image_directory', ''),
        area_threshold=config.getfloat('area_threshold', fallback=10.0),
        skip_contrast_adjustment=config.getboolean('skip_contrast_adjustment', fallback=False),
        crop_left_pct=config.getfloat('crop_left', fallback=20.0) / 100.0,
        crop_right_pct=config.getfloat('crop_right', fallback=3.0) / 100.0,
        crop_top_pct=config.getfloat('crop_top', fallback=3.0) / 100.0,
        crop_bottom_pct=config.getfloat('crop_bottom', fallback=3.0) / 100.0,
        filename=config.get('filename', fallback=''),
        img_debug=config.getboolean('img_debug', fallback=False),
        adaptive_threshold=config.getboolean('adaptive_threshold', fallback=True),
        adaptive_window_size=config.getint('adaptive_window_size', fallback=15),
        adaptive_C=config.getint('adaptive_C', fallback=2),
        color_threshold=config.getint('color_threshold', fallback=240)
    )


def create_directories(config: Config) -> Tuple[str, str]:
    """
    Create necessary directories for intermediate and result images.

    Args:
        config (Config): Configuration dataclass.

    Returns:
        Tuple[str, str]: Paths to intermediate and result folders.
    """
    intermediate_folder = os.path.join(config.image_directory, 'intermediate_img')
    result_folder = os.path.join(config.image_directory, 'result_img')
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    logger.debug(f"Intermediate folder: {intermediate_folder}, Result folder: {result_folder}")
    return intermediate_folder, result_folder


def get_image_files(config: Config) -> List[str]:
    """
    Get the list of image files to process.

    Args:
        config (Config): Configuration dataclass.

    Returns:
        List[str]: List of image filenames.
    """
    if config.filename:
        logger.debug(f"Processing single file: {config.filename}")
        return [config.filename]
    else:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
        image_files = [f for f in os.listdir(config.image_directory)
                       if f.lower().endswith(valid_extensions)]
        logger.debug(f"Found {len(image_files)} image files.")
        return image_files


def process_image(image_path: str, config: Config, intermediate_folder: str, result_folder: str, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
    """
    Process a single image and return the analysis results along with RGB statistics.

    Args:
        image_path (str): Path to the image file.
        config (Config): Configuration dataclass with processing parameters.
        intermediate_folder (str): Path to save intermediate images.
        result_folder (str): Path to save the final result image.
        stop_event (threading.Event): Event to signal stopping of processing.

    Returns:
        Optional[Dict[str, Any]]: Analysis results for the image or None if an error occurs or processing is stopped.
    """
    if stop_event.is_set():
        logger.info("Processing was stopped before starting.")
        return None

    start_time = time.time()
    filename = os.path.basename(image_path)

    try:
        logger.info(f"Processing {filename}")

        # Open the image and get its properties
        pil_image = Image.open(image_path)
        width, height = pil_image.size
        dpi = pil_image.info.get('dpi', (600, 600))
        x_dpi = dpi[0] if dpi else 600
        pixels_per_mm = x_dpi / 25.4  # Convert DPI to pixels per millimeter

        logger.debug(f"Image size: {width}x{height} pixels, DPI: {dpi}, Pixels/mm: {pixels_per_mm:.2f}")

        # Convert to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Apply image preprocessing steps
        processed_image, cropped_cv, top_offset, left_offset = preprocess_image(
            image_cv, config, intermediate_folder, filename, pixels_per_mm, stop_event
        )

        if stop_event.is_set():
            logger.info(f"Processing was stopped after preprocessing {filename}.")
            return None

        # Find and process contours, including RGB stats
        total_count, total_area_mm2, table, leaf_rgb_stats = find_and_process_contours(
            processed_image, cropped_cv, pixels_per_mm, config, result_folder,
            filename, top_offset, left_offset, stop_event
        )

        if stop_event.is_set():
            logger.info(f"Processing was stopped after contour processing {filename}.")
            return None

        processing_time = time.time() - start_time
        logger.info(f"Finished processing {filename}. "
                    f"Leaf Count: {total_count}, Total Leaf Area: {total_area_mm2:.2f} mm², "
                    f"Processing Time: {processing_time:.2f} s")

        # Perform regex searches once and reuse
        pct_match = re.search(r"_(HF|CF)_", filename)
        leaf_stem_match = re.search(r"_(L|S)_", filename)

        # Print table content to console
        if table:
            print(f"\n--- Leaf Analysis Table for {filename} ---")
            print(f"{'ID':<10}{'Area (mm²)':<15}")
            for row in table:
                print(f"{row[0]:<10}{row[1]:<15}")
            print(f"---------------------------------------------\n")
        else:
            print(f"No leaves detected in {filename}.")

        return {
            'Filename': filename,
            'PCT': pct_match.group(1) if pct_match else None,
            'Leaf_stem': leaf_stem_match.group(1) if leaf_stem_match else None,
            'Leaf Count': total_count,
            'Total Leaf Area (mm²)': total_area_mm2,
            'Pixels_per_mm': pixels_per_mm,
            'Processing Time (s)': round(processing_time, 2),
            'Leaf RGB Stats': leaf_rgb_stats  # Include RGB stats
        }

    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
        return None


def preprocess_image(image_cv: np.ndarray, config: Config, intermediate_folder: str, filename: str, pixels_per_mm: float, stop_event: threading.Event) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Apply preprocessing steps to the image, including adaptive thresholding in RGB space.

    Args:
        image_cv (np.ndarray): OpenCV image in BGR format.
        config (Config): Configuration dataclass.
        intermediate_folder (str): Path to save intermediate images.
        filename (str): Name of the image file.
        pixels_per_mm (float): Pixels per millimeter.
        stop_event (threading.Event): Event to signal stopping of processing.

    Returns:
        Tuple[np.ndarray, np.ndarray, int, int]: Processed binary image, cropped OpenCV image, top and left offsets.
    """
    # Step 1: Cropping based on configured percentages
    height, width = image_cv.shape[:2]
    left = int(width * config.crop_left_pct)
    right = int(width * (1 - config.crop_right_pct))
    top = int(height * config.crop_top_pct)
    bottom = int(height * (1 - config.crop_bottom_pct))

    cropped_image = image_cv[top:bottom, left:right]
    logger.debug(f"Cropped image dimensions: {cropped_image.shape}")

    save_image(cropped_image, intermediate_folder, filename, '1_cropped_image', config.img_debug)

    if stop_event.is_set():
        return None, None, 0, 0

    # Convert cropped image to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Step 2: Conditional Contrast Adjustment using CLAHE
    if not config.skip_contrast_adjustment:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_image = clahe.apply(gray_image)
        logger.debug("Applied CLAHE for contrast adjustment.")
    else:
        contrast_image = gray_image
        logger.debug("Skipped contrast adjustment as per configuration.")

    save_image(contrast_image, intermediate_folder, filename, '2_contrast_image', config.img_debug)

    if stop_event.is_set():
        return None, None, 0, 0

    # Step 3: Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(contrast_image, (5, 5), 0)
    logger.debug("Applied Gaussian Blur.")
    save_image(blurred_image, intermediate_folder, filename, '3_blurred_image', config.img_debug)

    if stop_event.is_set():
        return None, None, 0, 0

    # Step 4: Adaptive Thresholding to binarize the image
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2  # Adjusted blockSize and C
    )
    save_image(binary_image, intermediate_folder, filename, '4_binary_image', config.img_debug)

    if stop_event.is_set():
        return None, None, 0, 0

    # Step 5: Morphological Closing to close small gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    save_image(closed_image, intermediate_folder, filename, '5_closed_image', config.img_debug)

    if stop_event.is_set():
        return None, None, 0, 0

    # Step 6: Adaptive Thresholding in RGB Space
    if config.adaptive_threshold:
        logger.debug("Applying adaptive thresholding in RGB space.")

        # Convert image to RGB (from BGR)
        image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Define near-white mask based on color_threshold
        near_white_mask = np.all(image_rgb > config.color_threshold, axis=2).astype(np.uint8) * 255

        # Apply adaptive thresholding using a Gaussian blur in RGB space
        # Calculate the mean RGB values in local windows
        mean_r = cv2.blur(image_rgb[:, :, 0], (config.adaptive_window_size, config.adaptive_window_size))
        mean_g = cv2.blur(image_rgb[:, :, 1], (config.adaptive_window_size, config.adaptive_window_size))
        mean_b = cv2.blur(image_rgb[:, :, 2], (config.adaptive_window_size, config.adaptive_window_size))

        # Create a mask where pixel RGB values are significantly higher than local means
        adaptive_mask = (
            (image_rgb[:, :, 0] > (mean_r + config.adaptive_C)) &
            (image_rgb[:, :, 1] > (mean_g + config.adaptive_C)) &
            (image_rgb[:, :, 2] > (mean_b + config.adaptive_C))
        ).astype(np.uint8) * 255

        # Combine with near-white mask to exclude near-white regions
        combined_mask = cv2.bitwise_or(near_white_mask, adaptive_mask)

        # Invert mask to get non-near-white regions
        combined_mask = cv2.bitwise_not(combined_mask)

        # Update the binary_image to exclude near-white regions
        binary_image = cv2.bitwise_and(binary_image, combined_mask)

        logger.debug("Adaptive thresholding in RGB space applied.")
        save_image(binary_image, intermediate_folder, filename, 'X_adaptive_threshold', config.img_debug)

    # Step 7: Remove Small Objects based on Area Threshold
    area_threshold_pixels = config.area_threshold * (pixels_per_mm ** 2)
    cleaned_image = remove_small_objects(closed_image, area_threshold_pixels)
    save_image(cleaned_image, intermediate_folder, filename, '6_cleaned_image', config.img_debug)

    if stop_event.is_set():
        return None, None, 0, 0

    return cleaned_image, cropped_image, top, left


def remove_small_objects(closed_image: np.ndarray, area_threshold_pixels: float) -> np.ndarray:
    """
    Remove small objects from binary image based on area threshold.

    Args:
        closed_image (np.ndarray): Binary image after morphological closing.
        area_threshold_pixels (float): Area threshold in pixels.

    Returns:
        np.ndarray: Cleaned binary image.
    """
    # Find connected components
    num_labels, labels_im = cv2.connectedComponents(closed_image)

    # Count the size of each component
    sizes = np.bincount(labels_im.ravel())

    # Create a mask for components that meet the area threshold
    mask_sizes = sizes >= area_threshold_pixels
    mask_sizes[0] = 0  # Ensure background is not included

    # Apply the mask to the labels image
    output_image = mask_sizes[labels_im]

    # Convert the boolean mask to uint8
    output_image = (output_image * 255).astype(np.uint8)

    logger.debug(f"Removed small objects below {area_threshold_pixels} pixels.")
    return output_image


def save_image(image: np.ndarray, folder: str, filename: str, step_name: str, img_debug: bool) -> None:
    """
    Save intermediate images if img_debug is True.

    Args:
        image (np.ndarray): Image to save.
        folder (str): Directory to save the image.
        filename (str): Original filename.
        step_name (str): Step identifier for the image.
        img_debug (bool): Flag to control saving of intermediate images.
    """
    if img_debug:
        image_name = f"{os.path.splitext(filename)[0]}_{step_name}.png"
        image_path = os.path.join(folder, image_name)
        cv2.imwrite(image_path, image)
        logger.debug(f"Saved intermediate image: {image_path}")
    else:
        logger.debug(f"Skipping saving intermediate image: {filename}_{step_name}.png")


def create_annotation_table(annotations: List[Dict[str, Any]], areas_mm2: List[float]) -> List[Tuple[str, str]]:
    """
    Create a table of annotations with contour numbers and areas.

    Args:
        annotations (List[Dict[str, Any]]): List of annotation dictionaries.
        areas_mm2 (List[float]): List of leaf areas in mm².

    Returns:
        List[Tuple[str, str]]: List of tuples representing table rows.
    """
    table = []
    for ann, area in zip(annotations, areas_mm2):
        table.append((ann['text'], f"{area:.2f} mm²"))
    logger.debug("Created annotation table.")
    return table


def add_table_to_image(image: np.ndarray, table: List[Tuple[str, str]], table_position: Tuple[int, int] = (50, 50)) -> np.ndarray:
    """
    Add a table to the image at the specified position with enhanced aesthetics.

    Args:
        image (np.ndarray): Image to draw the table on.
        table (List[Tuple[str, str]]): List of tuples representing table rows.
        table_position (Tuple[int, int], optional): (x, y) position to place the table. Defaults to (50, 50).

    Returns:
        np.ndarray: Image with the table added.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = 0.6  # Base font scale for visibility
    base_font_thickness = 2
    base_line_height = 30
    base_margin = 15

    # Determine table size based on image width to ensure at least 10% width
    image_height, image_width = image.shape[:2]
    min_table_width = int(0.1 * image_width)
    current_table_width = 300  # Increased width for better visibility
    table_width = max(current_table_width, min_table_width)

    # Calculate the number of rows and columns
    num_rows = len(table) + 1  # Including header
    num_cols = 2
    table_height = num_rows * base_line_height + 2 * base_margin

    x_start, y_start = table_position

    # Dynamic Font Scaling based on image width
    base_width = 1920  # Reference width for scaling
    scale_factor = image_width / base_width
    font_scale = base_font_scale * scale_factor
    font_thickness = max(1, int(base_font_thickness * scale_factor))
    line_height = int(base_line_height * scale_factor)
    margin = int(base_margin * scale_factor)

    # Adjust table position if it exceeds image boundaries
    if x_start + table_width > image_width:
        x_start = image_width - table_width - 20  # 20 pixels padding from the edge
    if y_start + table_height > image_height:
        y_start = image_height - table_height - 20  # 20 pixels padding from the edge

    logger.debug(f"Table position: ({x_start}, {y_start}), Width: {table_width}px, Height: {table_height}px")

    # Draw semi-transparent table background
    overlay = image.copy()
    cv2.rectangle(overlay, (x_start, y_start),
                  (x_start + table_width, y_start + table_height),
                  (255, 255, 255), -1)  # White background
    alpha = 0.8  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw table border
    cv2.rectangle(image, (x_start, y_start),
                  (x_start + table_width, y_start + table_height),
                  (0, 0, 0), 2)  # Black border

    # Define column widths
    col1_width = int(table_width * 0.3)  # 30% for ID
    col2_width = table_width - col1_width  # 70% for Area

    # Add header
    header = ("ID", "Area (mm²)")
    for i, text in enumerate(header):
        x_offset = 10 + i * col1_width
        y_offset = y_start + margin + line_height
        cv2.putText(image, text, (x_start + x_offset, y_start + margin + line_height),
                    font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Add rows with alternating background colors for better readability
    for idx, row in enumerate(table):
        y = y_start + margin + (idx + 2) * line_height  # +2 to account for header
        # Alternate row color
        if idx % 2 == 0:
            row_color = (240, 240, 240)  # Light gray
            cv2.rectangle(image, (x_start, y - line_height + 5),
                          (x_start + table_width, y + 5),
                          row_color, -1)
        for col_idx, item in enumerate(row):
            x_offset = 10 + col_idx * col1_width
            cv2.putText(image, item, (x_start + x_offset, y),
                        font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Draw vertical lines for column separators
    cv2.line(image, (x_start + col1_width, y_start),
             (x_start + col1_width, y_start + table_height),
             (0, 0, 0), 2)

    logger.debug("Added table to image with enhanced aesthetics.")
    return image


def find_and_process_contours(processed_image: np.ndarray, cropped_cv: np.ndarray, pixels_per_mm: float,
                              config: Config, result_folder: str, filename: str,
                              top_offset: int, left_offset: int, stop_event: threading.Event) -> Tuple[int, float, List[Tuple[str, str]], List[Dict[str, Any]]]:
    """
    Find contours, calculate leaf areas, annotate the image, and compute mean RGB values within contours.

    Args:
        processed_image (np.ndarray): Binary image after preprocessing.
        cropped_cv (np.ndarray): Cropped original image for background.
        pixels_per_mm (float): Pixels per millimeter.
        config (Config): Configuration dataclass.
        result_folder (str): Path to save the final result image.
        filename (str): Name of the image file.
        top_offset (int): Top offset used during cropping.
        left_offset (int): Left offset used during cropping.
        stop_event (threading.Event): Event to signal stopping of processing.

    Returns:
        Tuple[int, float, List[Tuple[str, str]], List[Dict[str, Any]]]: Total count, total area in mm², annotation table, list of leaf RGB stats.
    """
    if stop_event.is_set():
        logger.info("Processing was stopped before contour processing.")
        return 0, 0.0, [], []

    # Ensure binary image is in uint8 format
    if processed_image.dtype != np.uint8:
        processed_image = processed_image.astype(np.uint8)

    # Find contours using OpenCV
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} total contours.")

    areas_mm2 = []
    total_count = 0
    pixels_per_mm2 = pixels_per_mm ** 2

    output_image = cropped_cv.copy()  # Use cropped image as background

    # Generate unique colors for each contour
    np.random.seed(42)  # For reproducibility
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(contours))]

    annotations = []
    leaf_rgb_stats = []

    # Iterate through each contour to calculate area, annotate, and compute mean RGB
    for idx, contour in enumerate(contours, 1):
        if stop_event.is_set():
            logger.info("Processing was stopped during contour iteration.")
            break

        area_pixels = cv2.contourArea(contour)
        area_mm2 = area_pixels / pixels_per_mm2

        if area_mm2 >= config.area_threshold:
            areas_mm2.append(area_mm2)
            total_count += 1

            # Draw contour with unique color
            color = colors[idx - 1]
            cv2.drawContours(output_image, [contour], -1, color, 2)

            # Calculate centroid for placing the annotation
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Fallback if contour area is zero
                x, y, w, h = cv2.boundingRect(contour)
                cX = x + w // 2
                cY = y + h // 2

            # Prepare the annotation text
            text = f"{total_count}"
            annotations.append({
                'text': text,
                'position': (cX, cY),
                'color': color
            })

            # Create a mask for the current contour
            contour_mask = np.zeros(processed_image.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)  # Filled contour

            # Extract RGB values within the contour
            image_rgb = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB)
            masked_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=contour_mask)

            # Calculate mean R, G, B values, ignoring zero pixels
            mean_r = cv2.mean(masked_rgb[:, :, 0], mask=contour_mask)[0]
            mean_g = cv2.mean(masked_rgb[:, :, 1], mask=contour_mask)[0]
            mean_b = cv2.mean(masked_rgb[:, :, 2], mask=contour_mask)[0]

            leaf_rgb_stats.append({
                'Leaf ID': total_count,
                'Mean R': round(mean_r, 2),
                'Mean G': round(mean_g, 2),
                'Mean B': round(mean_b, 2)
            })

    if not annotations and not stop_event.is_set():
        logger.warning("No valid contours detected after applying area threshold.")

    # Annotate the leaves with numbers, adding a white background behind each number
    for ann in annotations:
        # Define the size of the text background
        (text_width, text_height), baseline = cv2.getTextSize(ann['text'], cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        x, y = ann['position']
        # Draw white rectangle behind the text for better visibility
        cv2.rectangle(output_image, (x, y - text_height - baseline),
                      (x + text_width, y + baseline), (255, 255, 255), -1)  # White background
        # Put the text on top of the white rectangle
        cv2.putText(output_image, ann['text'], (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, ann['color'], 2, cv2.LINE_AA)  # Bigger font size

    # Create a table/list on the image
    table = create_annotation_table(annotations, areas_mm2)
    table_position = (50, 50)  # Initial table position; will adjust dynamically
    output_image = add_table_to_image(output_image, table, table_position)

    # Save the final output image
    result_image_path = os.path.join(result_folder, filename)
    cv2.imwrite(result_image_path, output_image)
    logger.debug(f"Saved final annotated image: {result_image_path}")

    total_area_mm2 = sum(areas_mm2)
    return total_count, total_area_mm2, table, leaf_rgb_stats


def save_results(results_list: List[Dict[str, Any]], csv_file_path: str) -> None:
    """
    Save the analysis results, including mean RGB values, to a CSV file.

    Args:
        results_list (List[Dict[str, Any]]): List of dictionaries with analysis results.
        csv_file_path (str): Path to the CSV file.
    """
    # Prepare data for CSV
    rows = []
    for result in results_list:
        base_info = {
            'Filename': result.get('Filename'),
            'PCT': result.get('PCT'),
            'Leaf_stem': result.get('Leaf_stem'),
            'Leaf Count': result.get('Leaf Count'),
            'Total Leaf Area (mm²)': result.get('Total Leaf Area (mm²)'),
            'Pixels_per_mm': result.get('Pixels_per_mm'),
            'Processing Time (s)': result.get('Processing Time (s)')
        }
        # Add RGB stats
        for leaf_stat in result.get('Leaf RGB Stats', []):
            row = base_info.copy()
            row['Leaf ID'] = leaf_stat['Leaf ID']
            row['Mean R'] = leaf_stat['Mean R']
            row['Mean G'] = leaf_stat['Mean G']
            row['Mean B'] = leaf_stat['Mean B']
            rows.append(row)

    df_new = pd.DataFrame(rows)

    if os.path.exists(csv_file_path):
        try:
            df_existing = pd.read_csv(csv_file_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(csv_file_path, index=False)
    logger.info(f"Results saved to {csv_file_path}")


def main(stop_event: Optional[threading.Event] = None) -> None:
    """
    Main function to orchestrate the leaf classification process.

    Args:
        stop_event (Optional[threading.Event], optional): Event to signal stopping of processing. Defaults to None.
    """
    config = read_config()
    intermediate_folder, result_folder = create_directories(config)
    csv_file_path = os.path.join(config.image_directory, 'leaf_analysis_results.csv')
    image_files = get_image_files(config)
    results_list = []

    for image_file in image_files:
        if stop_event and stop_event.is_set():
            logger.info("Processing was stopped before processing all images.")
            break

        image_path = os.path.join(config.image_directory, image_file)
        result = process_image(image_path, config, intermediate_folder, result_folder, stop_event)
        if result:
            results_list.append(result)

    if results_list and not (stop_event and stop_event.is_set()):
        save_results(results_list, csv_file_path)
    elif stop_event and stop_event.is_set():
        logger.info("Processing was interrupted. Partial results may not be saved.")


if __name__ == "__main__":
    main()
