"""
image_preprocessor.py — SwiftGrade OCR Image Pre-processing Pipeline
====================================================================
Implements the image pre-processing requirements defined in Module B:
  - Resolution Check (min 480p)
  - Lighting Normalization (CLAHE)
  - Orientation Correction (Deskewing)
  - Binarization (Otsu Thresholding)
  - Noise Reduction (Morphological Operations)
"""

import cv2
import numpy as np
from PIL import Image

def load_image_with_dpi_check(image_path, min_dpi=300):
    """Loads an image and warns if DPI is below the required threshold."""
    pil_img = Image.open(image_path)
    dpi = pil_img.info.get('dpi', (None, None))
    
    if dpi[0] is None or dpi[0] < min_dpi:
        print(f"⚠️ Warning: Image {image_path} has DPI {dpi[0]}. Recommended is {min_dpi}+")
        
    # Convert PIL to cv2 format (BGR)
    open_cv_image = np.array(pil_img) 
    # Convert RGB to BGR 
    if len(open_cv_image.shape) == 3 and open_cv_image.shape[2] == 3:
        open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

def enforce_minimum_resolution(image, min_short_edge=480):
    """Resizes the image if its shortest edge is below the minimum requirement."""
    h, w = image.shape[:2]
    short_edge = min(h, w)
    
    if short_edge < min_short_edge:
        scale_factor = min_short_edge / short_edge
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        print(f"🔄 Resized image from {w}x{h} to {new_w}x{new_h} to meet 480p min requirement.")
    return image

def deskew(image):
    """Corrects the orientation (skew) of the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Invert colors: text is white, background is black
    gray = cv2.bitwise_not(gray)
    
    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Find all non-zero points (text pixels)
    coords = np.column_stack(np.where(thresh > 0))
    
    # Get the bounding box of these points
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust angle to correct range
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # If angle is minimal, skip rotation to avoid blurring
    if abs(angle) < 0.5:
        return image
        
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization for uniform lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def binarize_and_denoise(gray_image):
    """Applies Otsu binarization and morphological noise reduction."""
    # Gaussian Blur to smooth noise before binarization
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Otsu's thresholding
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological opening to remove salt noise
    kernel = np.ones((2, 2), np.uint8)
    denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return denoised

def preprocess_for_ocr(image_path):
    """Runs the full pre-processing pipeline on an image."""
    img = load_image_with_dpi_check(image_path)
    img = enforce_minimum_resolution(img)
    img = deskew(img)
    gray = apply_clahe(img)
    final_binary = binarize_and_denoise(gray)
    
    return img, final_binary # Return original (deskewed) and binary for comparison/different engine needs
