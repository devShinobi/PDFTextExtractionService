import cv2
import numpy as np
import os
import tempfile
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import pytesseract
from pdf2image import convert_from_path

# Configure Tesseract path for Windows
# Update this path to match your Tesseract installation
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Alternative paths if the above doesn't work:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Users\YourUsername\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class ImageProcessor:
    def __init__(self):
        """Initialize the image processor."""
        pass

    def pdf_to_images(self, pdf_path: str, dpi: int = 200) -> List[np.ndarray]:
        """Convert PDF to images.
        
        Args:
            pdf_path: Path to the PDF file
            dpi: Resolution for image conversion
            
        Returns:
            List of images as numpy arrays
        """
        print(f"Converting PDF to images: {pdf_path}")
        
        images = convert_from_path(pdf_path, dpi=dpi)
        
        numpy_images = []
        for i, image in enumerate(images):
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            numpy_images.append(img_array)
            print(f"Page {i+1}: {img_array.shape}")
        
        return numpy_images

    def detect_paragraphs(self, image: np.ndarray, exclude_bottom_percent: float = 0.07, exclude_top_percent: float = 0.07, ignore_vertical_text: bool = False, debug: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Detects text blocks using morphological analysis with adaptive kernels."""
        print("Detecting text blocks with adaptive morphological analysis...")

        height, width, _ = image.shape
        top_cutoff = int(height * exclude_top_percent)
        bottom_cutoff = int(height * (1 - exclude_bottom_percent))

        # 1. Create a binary "ink" map
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

        # 2. Get word-level data to determine adaptive kernel sizes
        try:
            data = pytesseract.image_to_data(image[top_cutoff:bottom_cutoff, :], output_type=pytesseract.Output.DICT, config='--psm 6')
            
            # --- Character Height Analysis (for vertical kernel) ---
            words_data = [{'text': data['text'][i], 'conf': int(data['conf'][i]), 'height': data['height'][i], 'x': data['left'][i], 'y': data['top'][i], 'w': data['width'][i]} for i in range(len(data['text'])) if int(data['conf'][i]) > 30 and data['text'][i].strip()]
            
            if words_data:
                avg_char_height = np.median([w['height'] for w in words_data])
            else:
                avg_char_height = 12 # Fallback
            
            # --- Word Spacing Analysis (for horizontal kernel) ---
            # First, group words into lines based on their y-coordinate
            lines = {}
            for word in words_data:
                line_key = round(word['y'] / avg_char_height) # Group by line
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append(word)
            
            # Now, calculate the horizontal gap between adjacent words on each line
            word_gaps = []
            for line_key in lines:
                line = sorted(lines[line_key], key=lambda w: w['x'])
                for i in range(len(line) - 1):
                    curr_word = line[i]
                    next_word = line[i+1]
                    gap = next_word['x'] - (curr_word['x'] + curr_word['w'])
                    # Only consider plausible gaps
                    if gap > 0 and gap < (avg_char_height * 3):
                        word_gaps.append(gap)
            
            if word_gaps:
                median_word_gap = np.median(word_gaps)
            else:
                median_word_gap = avg_char_height * 0.5 # Fallback if no gaps found

        except Exception as e:
            print(f"Warning: Could not perform word-level analysis, using fallback. Error: {e}")
            avg_char_height = 12
            median_word_gap = 6

        print(f"Adaptive kernel calculation: Median char height: {avg_char_height:.2f}px, Median word gap: {median_word_gap:.2f}px")

        # 3. Create adaptive kernels based on the analysis
        h_kernel_size = int(median_word_gap * 1.5) # Kernel must be > gap to connect words
        v_kernel_size = int(avg_char_height * 1.5) # Kernel for connecting lines
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        
        print(f"Using adaptive kernels: Horizontal({h_kernel_size}, 1), Vertical(1, {v_kernel_size})")

        # 4. Apply morphological closing
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, v_kernel)

        # 5. Find contours of the resulting text blocks
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        coarse_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < avg_char_height or h < avg_char_height: continue
            if y < top_cutoff or (y + h) > bottom_cutoff: continue
            coarse_regions.append({'x': x, 'y': y, 'width': w, 'height': h})

        # 6. Refinement Pass: Redraw boxes based on actual word content
        print("Refining initial boxes to fit content tightly...")
        refined_regions = []
        for region in coarse_regions:
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            roi = image[y:y+h, x:x+w]
            
            try:
                data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT, config='--psm 6')
                
                words_in_box = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30 and data['text'][i].strip():
                        word_x = x + data['left'][i]
                        word_y = y + data['top'][i]
                        word_w = data['width'][i]
                        word_h = data['height'][i]
                        word_text = data['text'][i]
                        words_in_box.append({'x': word_x, 'y': word_y, 'w': word_w, 'h': word_h, 'text': word_text})
                
                if not words_in_box: continue

                tight_x1 = min(w['x'] for w in words_in_box)
                tight_y1 = min(w['y'] for w in words_in_box)
                tight_x2 = max(w['x'] + w['w'] for w in words_in_box)
                tight_y2 = max(w['y'] + w['h'] for w in words_in_box)
                
                refined_regions.append({
                    'x': tight_x1, 'y': tight_y1, 
                    'width': tight_x2 - tight_x1, 'height': tight_y2 - tight_y1
                })
            except:
                refined_regions.append(region)

        # 7. Sort regions using a robust, row-based algorithm
        print("Sorting regions using robust row-based algorithm...")
        
        regions = refined_regions
        
        final_sorted_regions = []
        while len(regions) > 0:
            # Sort remaining regions by top edge to find the next row's seed
            regions.sort(key=lambda r: r['y'])
            
            # Start a new row with the topmost region
            current_row = [regions.pop(0)]
            
            # Establish the initial vertical span of the row
            row_y_min = current_row[0]['y']
            row_y_max = current_row[0]['y'] + current_row[0]['height']

            # Iteratively find all other regions belonging to this row
            row_changed = True
            while row_changed:
                row_changed = False
                remaining_after_pass = []
                for region in regions:
                    # Check for any vertical overlap with the current row's span
                    if (region['y'] < row_y_max) and ((region['y'] + region['height']) > row_y_min):
                        current_row.append(region)
                        # Expand the row's vertical span to include the new region
                        row_y_min = min(row_y_min, region['y'])
                        row_y_max = max(row_y_max, region['y'] + region['height'])
                        # Mark that the row has changed, so we need to re-scan
                        row_changed = True
                    else:
                        remaining_after_pass.append(region)
                
                regions = remaining_after_pass

            # Once the row is complete, sort it by horizontal position
            current_row.sort(key=lambda r: r['x'])
            final_sorted_regions.extend(current_row)

        regions = final_sorted_regions
        
        # Only create result image if debug mode is enabled
        if debug:
            result_image = image.copy()
            for i, region in enumerate(regions):
                region['id'] = i + 1
                cv2.rectangle(result_image, (region['x'], region['y']), (region['x'] + region['width'], region['y'] + region['height']), (0, 255, 0), 2)
                cv2.putText(result_image, f"Block {region['id']}", (region['x'], region['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.line(result_image, (0, top_cutoff), (width, top_cutoff), (255, 0, 0), 2)
            cv2.line(result_image, (0, bottom_cutoff), (width, bottom_cutoff), (0, 0, 255), 2)
        else:
            # Create a minimal result image (just the original) when not in debug mode
            result_image = image.copy()
            # Still assign IDs to regions for consistency
            for i, region in enumerate(regions):
                region['id'] = i + 1

        print(f"Detected {len(regions)} text blocks.")
        return result_image, regions

    def extract_text_from_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts text by running OCR on each detected region."""
        print("Extracting text from final bounding boxes...")
        
        extracted_texts = set()
        final_paragraphs = []

        for region in regions:
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            
            # Add a small padding to the ROI to avoid cutting off text at the edges
            padding = 5
            roi = image[max(0, y - padding):min(image.shape[0], y + h + padding), 
                        max(0, x - padding):min(image.shape[1], x + w + padding)]
            
            # Use PSM 4 which assumes a single column of text of variable sizes.
            # This is generally more robust for OCRing paragraph blocks.
            text = pytesseract.image_to_string(roi, config='--psm 4').strip()

            if text and text not in extracted_texts:
                region['text'] = text
                region['text_length'] = len(text)
                extracted_texts.add(text)
                final_paragraphs.append(region)
        
        print(f"Extracted {len(final_paragraphs)} unique text blocks.")
        return final_paragraphs

    def process_pdf(self, pdf_path: str, output_dir: str = "test_output", final_output_dir: str = None, exclude_bottom_percent: float = 0.07, exclude_top_percent: float = 0.07, ignore_vertical_text: bool = False, debug: bool = False) -> List[Dict[str, Any]]:
        """Process a PDF file: convert to images, detect paragraphs, and extract text."""
        print(f"Processing PDF: {pdf_path}")
        print(f"Excluding top {exclude_top_percent*100}% and bottom {exclude_bottom_percent*100}% of each page.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Use final_output_dir if specified, otherwise use output_dir
        text_output_dir = final_output_dir if final_output_dir is not None else output_dir
        if text_output_dir != output_dir:
            os.makedirs(text_output_dir, exist_ok=True)
            print(f"Intermediate outputs will be saved to: {output_dir}")
            print(f"Final text output will be saved to: {text_output_dir}")
        
        start_time = time.time()
        images = self.pdf_to_images(pdf_path)
        all_paragraphs = []
        
        for page_num, image in enumerate(images):
            print(f"\nProcessing page {page_num + 1}")
            
            result_image, regions = self.detect_paragraphs(
                image, 
                exclude_bottom_percent=exclude_bottom_percent, 
                exclude_top_percent=exclude_top_percent,
                ignore_vertical_text=ignore_vertical_text,
                debug=debug
            )
            
            # Only save debug images if debug mode is enabled
            if debug:
                output_path = os.path.join(output_dir, f"page_{page_num + 1}_detected.png")
                cv2.imwrite(output_path, result_image)
                print(f"Saved detected blocks to: {output_path}")
            
            paragraphs_with_text = self.extract_text_from_regions(image, regions)
            
            for paragraph in paragraphs_with_text:
                paragraph['page'] = page_num + 1
            
            all_paragraphs.extend(paragraphs_with_text)
        
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f} seconds")
        
        self.save_text_to_file(all_paragraphs, text_output_dir, pdf_path)
        
        return all_paragraphs

    def save_text_to_file(self, paragraphs: List[Dict[str, Any]], output_dir: str, pdf_path: str):
        """Saves extracted text to a file, sorted by page and reading order."""
        if not paragraphs:
            print("No text was extracted, skipping file save.")
            return

        base_name = os.path.basename(pdf_path)
        file_name = os.path.splitext(base_name)[0]
        output_filepath = os.path.join(output_dir, f"{file_name}_extracted_text.txt")

        # Sort paragraphs by page, then by their ID, which respects the column-aware sort.
        paragraphs.sort(key=lambda p: (p.get('page', 0), p.get('id', 0)))

        with open(output_filepath, 'w', encoding='utf-8') as f:
            current_page = -1
            for para in paragraphs:
                if para.get('page', -1) != current_page:
                    current_page = para.get('page', -1)
                    f.write(f"\n{'='*20} Page {current_page} {'='*20}\n\n")
                
                f.write(para.get('text', ''))
                f.write('\n\n' + '-'*40 + '\n\n')

        print(f"Successfully saved extracted text to {output_filepath}") 