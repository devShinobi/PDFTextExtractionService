import pytesseract
from PIL import Image
import cv2
import numpy as np
import logging
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Protocol, Any
import multiprocessing
from scipy.signal import find_peaks
import re
import tempfile

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional GPU dependencies
try:
    import torch
    import torch.cuda
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU detected, falling back to CPU")
except ImportError:
    HAS_GPU = False
    logger.info("GPU dependencies not installed, falling back to CPU")

# Thread-local storage for OpenCV operations
thread_local = threading.local()

def get_thread_local_cv2():
    """Get thread-local OpenCV objects to avoid conflicts in parallel processing."""
    if not hasattr(thread_local, 'cv2'):
        thread_local.cv2 = cv2
    return thread_local.cv2

@lru_cache(maxsize=32)
def get_optimal_kernel_size(image_size):
    """Calculate optimal kernel size based on image dimensions."""
    width, height = image_size
    return (max(3, width // 100), max(3, height // 100))

class OCRProcessor(Protocol):
    """Protocol for OCR processing."""
    def process_image(self, image: Image.Image) -> str:
        """Process an image and return extracted text."""
        ...

class TesseractOCRProcessor:
    """Tesseract OCR implementation."""
    def __init__(self, tesseract_cmd: str = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        # Create temp directory for debug images
        self.temp_dir = os.path.join(tempfile.gettempdir(), 'ocr_debug_images')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def process_image(self, image: Image.Image) -> str:
        """Process an image using Tesseract OCR."""
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
        
        # Scale image if too small
        min_height = 1000
        if image.height < min_height:
            scale = min_height / image.height
            new_width = int(image.width * scale)
            new_height = int(image.height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save original grayscale image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image.save(os.path.join(self.temp_dir, f'1_original_gray_{timestamp}.png'))
        
        # Enhance image for better OCR
        image_np = np.array(image)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image_np)
        cv2.imwrite(os.path.join(self.temp_dir, f'2_clahe_enhanced_{timestamp}.png'), enhanced)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        cv2.imwrite(os.path.join(self.temp_dir, f'3_adaptive_threshold_{timestamp}.png'), binary)
        
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(binary)
        cv2.imwrite(os.path.join(self.temp_dir, f'4_denoised_{timestamp}.png'), denoised)
        
        # Convert back to PIL Image
        enhanced_image = Image.fromarray(denoised)
        
        # Configure Tesseract for better accuracy
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5 -c textord_parallel_baselines=1 -c textord_parallel_rows=1 -c textord_tabfind_find_tables=0 -c textord_tablefind_recognize_tables=0'
        
        # Perform OCR with enhanced configuration
        text = pytesseract.image_to_string(
            enhanced_image,
            config=custom_config,
            lang='eng'  # Specify language
        )
        
        # Clean up the text
        text = text.replace('\r', '')  # Remove carriage returns
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)  # Fix word breaks
        text = re.sub(r'([a-z])\s+([a-z])', r'\1 \2', text)  # Fix multiple spaces
        text = text.strip()  # Remove leading/trailing whitespace
        
        # Save the final image that was passed to Tesseract
        enhanced_image.save(os.path.join(self.temp_dir, f'5_final_tesseract_input_{timestamp}.png'))
        
        # Save OCR result for reference
        with open(os.path.join(self.temp_dir, f'6_ocr_result_{timestamp}.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"Debug images saved to: {self.temp_dir}")
        return text

@dataclass
class TextBlock:
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    confidence: float = 0.0
    block_type: str = "text"  # text, header, footer, title, etc.
    column: int = 0
    reading_order: int = 0
    area: int = 0

class LayoutAnalyzer:
    """Analyzes document layout to detect columns and reading order."""
    
    def __init__(self, image_shape: Tuple[int, int]):
        self.image_width = image_shape[1]
        self.image_height = image_shape[0]
    
    def detect_columns(self, block_positions: Tuple[Tuple[float, float], ...]) -> int:
        """Detect number of columns in the document."""
        if not block_positions:
            return 1
            
        # Extract x-coordinates of block centers
        x_coords = [x for x, _ in block_positions]
        
        # Use histogram to find column centers
        hist, bin_edges = np.histogram(x_coords, bins=50)
        peaks, _ = find_peaks(hist, height=max(hist)/3)
        
        # If no clear peaks found, assume single column
        if len(peaks) <= 1:
            return 1
            
        # Calculate distances between peaks
        peak_distances = np.diff(bin_edges[peaks])
        mean_distance = np.mean(peak_distances)
        
        # If peaks are too close, they're probably part of the same column
        if mean_distance < self.image_width * 0.1:  # 10% of image width
            return 1
            
        return len(peaks)
    
    def assign_columns(self, blocks: List[Dict[str, Any]], num_columns: int) -> List[Dict[str, Any]]:
        """Assign blocks to columns."""
        if num_columns == 1:
            for block in blocks:
                block['column'] = 0
            return blocks
            
        # Calculate column boundaries
        column_width = self.image_width / num_columns
        column_boundaries = [i * column_width for i in range(num_columns + 1)]
        
        # Assign blocks to columns
        for block in blocks:
            block_center = block['x'] + block['width'] / 2
            # Find which column the block belongs to
            for i in range(num_columns):
                if column_boundaries[i] <= block_center < column_boundaries[i + 1]:
                    block['column'] = i
                    break
            else:
                # If block doesn't fit in any column, assign to nearest
                block['column'] = min(range(num_columns), 
                                   key=lambda i: abs(block_center - (column_boundaries[i] + column_width/2)))
        
        return blocks
    
    def determine_reading_order(self, blocks: List[Dict[str, Any]], num_columns: int) -> List[Dict[str, Any]]:
        """Determine the correct reading order of blocks."""
        if num_columns == 1:
            # For single column, just sort by y-coordinate
            return sorted(blocks, key=lambda b: b['y'])
            
        # Group blocks by column
        column_blocks = [[] for _ in range(num_columns)]
        for block in blocks:
            column_blocks[block['column']].append(block)
            
        # Sort blocks within each column by y-coordinate
        for column in column_blocks:
            column.sort(key=lambda b: b['y'])
            
        # Interleave blocks from different columns
        result = []
        column_indices = [0] * num_columns
        
        while any(i < len(column_blocks[j]) for j, i in enumerate(column_indices)):
            # Find the column with the highest block
            next_column = min(
                (j for j in range(num_columns) if column_indices[j] < len(column_blocks[j])),
                key=lambda j: column_blocks[j][column_indices[j]]['y']
            )
            
            result.append(column_blocks[next_column][column_indices[next_column]])
            column_indices[next_column] += 1
            
        return result

class ImageProcessor:
    """Handles image preprocessing and text block detection."""
    def __init__(self, ocr_processor: OCRProcessor):
        self.ocr_processor = ocr_processor
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for better text block detection."""
        logger.info("Starting image preprocessing")
        
        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Scale image if too small (improves OCR accuracy)
        min_height = 1000
        if gray.shape[0] < min_height:
            scale = min_height / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Get optimal kernel size
        kernel_size = get_optimal_kernel_size(image.size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        # Perform morphological operations
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        # Apply additional noise removal
        final = cv2.fastNlMeansDenoising(eroded)
        
        logger.info("Image preprocessing completed")
        return final
    
    def detect_text_blocks(self, binary_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text blocks in the binary image."""
        logger.info("Starting text block detection")
        
        # Ensure we have a binary image
        if len(binary_image.shape) > 2:
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(binary_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        blocks = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Reduced minimum size to catch smaller text
            if w >= 5 and h >= 5:  # Reduced from 10 to 5
                blocks.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'text': ''
                })
        
        # Sort blocks by vertical position
        blocks.sort(key=lambda b: b['y'])
        
        # Merge blocks in the same line
        merged_blocks = []
        current_line = []
        
        for block in blocks:
            if not current_line:
                current_line.append(block)
            else:
                # Increased vertical threshold for line merging
                if abs(block['y'] - current_line[0]['y']) <= 10:  # Increased from 5 to 10
                    current_line.append(block)
                else:
                    # Merge blocks in current line
                    if len(current_line) > 1:
                        merged_block = self._merge_blocks(current_line)
                        merged_blocks.append(merged_block)
                    else:
                        merged_blocks.append(current_line[0])
                    current_line = [block]
        
        # Handle last line
        if current_line:
            if len(current_line) > 1:
                merged_block = self._merge_blocks(current_line)
                merged_blocks.append(merged_block)
            else:
                merged_blocks.append(current_line[0])
        
        # Merge horizontally close blocks
        final_blocks = []
        i = 0
        while i < len(merged_blocks):
            current = merged_blocks[i]
            j = i + 1
            while j < len(merged_blocks):
                next_block = merged_blocks[j]
                # Increased horizontal distance threshold
                if (next_block['x'] - (current['x'] + current['width'])) <= 20:  # Increased from 10 to 20
                    current = self._merge_blocks([current, next_block])
                    j += 1
                else:
                    break
            final_blocks.append(current)
            i = j
        
        logger.info(f"Detected {len(final_blocks)} text blocks")
        return final_blocks

    def _merge_blocks(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple blocks into one."""
        if not blocks:
            return {}
        
        x = min(b['x'] for b in blocks)
        y = min(b['y'] for b in blocks)
        width = max(b['x'] + b['width'] for b in blocks) - x
        height = max(b['y'] + b['height'] for b in blocks) - y
        
        # Add small padding to prevent clipping
        padding = 2
        return {
            'x': max(0, x - padding),
            'y': max(0, y - padding),
            'width': width + (2 * padding),
            'height': height + (2 * padding),
            'text': ''
        }

def format_paragraphs(paragraphs: List[str]) -> str:
    """Format paragraphs into a single column of text.
    
    Args:
        paragraphs: List of text paragraphs
        
    Returns:
        Formatted text with proper paragraph breaks
    """
    if not paragraphs:
        return ""
        
    # Join paragraphs with proper spacing
    formatted_text = []
    current_paragraph = []
    
    for para in paragraphs:
        # Skip empty paragraphs
        if not para.strip():
            continue
            
        # Split into lines and clean up
        lines = [line.strip() for line in para.split('\n') if line.strip()]
        
        # If this looks like a column break (very short line), merge with next line
        if len(lines) > 1 and len(lines[0]) < 20:  # Adjust threshold as needed
            lines[0] = lines[0] + " " + lines[1]
            lines.pop(1)
            
        # Join lines with spaces, preserving intentional line breaks
        cleaned_para = " ".join(lines)
        
        # Remove multiple spaces
        cleaned_para = re.sub(r'\s+', ' ', cleaned_para)
        
        # Add to current paragraph if it's a continuation
        if current_paragraph and not cleaned_para[0].isupper():
            current_paragraph.append(cleaned_para)
        else:
            # Start new paragraph
            if current_paragraph:
                formatted_text.append(" ".join(current_paragraph))
            current_paragraph = [cleaned_para]
    
    # Add the last paragraph
    if current_paragraph:
        formatted_text.append(" ".join(current_paragraph))
    
    # Join paragraphs with double newlines
    return "\n\n".join(formatted_text)

class TextProcessor:
    """Handles text extraction and processing."""
    def __init__(self, ocr_processor: OCRProcessor):
        self.ocr_processor = ocr_processor
        self.image_processor = ImageProcessor(ocr_processor)
    
    def process_image_with_layout(self, image: Image.Image, debug: bool = False) -> Tuple[List[str], Optional[Dict]]:
        """Process an image and return extracted text with layout analysis."""
        # Convert image to numpy array for OpenCV
        image_np = np.array(image)
        
        # Detect text blocks
        blocks = self.image_processor.detect_text_blocks(image_np)
        
        # Analyze layout
        layout_analyzer = LayoutAnalyzer(image_np.shape)
        num_columns = layout_analyzer.detect_columns(tuple((b['x'] + b['width']/2, b['width']) for b in blocks))
        
        # Assign blocks to columns and determine reading order
        blocks = layout_analyzer.assign_columns(blocks, num_columns)
        blocks = layout_analyzer.determine_reading_order(blocks, num_columns)
        
        # Extract text from blocks in reading order
        paragraphs = []
        current_column = -1
        
        for block in blocks:
            # Extract text from block
            block_image = image.crop((block['x'], block['y'], block['x'] + block['width'], block['y'] + block['height']))
            text = self.ocr_processor.process_image(block_image)
            
            if text.strip():
                # Add paragraph break when column changes
                if current_column != block['column']:
                    if paragraphs:  # Don't add extra newline at start
                        paragraphs.append("")
                    current_column = block['column']
                paragraphs.append(text)
        
        # Format paragraphs
        formatted_text = format_paragraphs(paragraphs)
        
        if debug:
            # Create debug visualization
            debug_image = image_np.copy()
            colors = {
                "text": (0, 255, 0),    # Green
                "header": (255, 0, 0),  # Red
                "footer": (0, 0, 255),  # Blue
                "title": (255, 255, 0), # Cyan
                "side_note": (0, 255, 255) # Yellow
            }
            
            # Draw rectangles around detected blocks
            for block in blocks:
                color = (0, 255, 0)  # Default to green for all blocks
                cv2.rectangle(debug_image, (block['x'], block['y']), 
                             (block['x'] + block['width'], block['y'] + block['height']), 
                             color, 2)
                
                # Add column info
                cv2.putText(debug_image, f"col {block['column']}",
                           (block['x'], block['y'] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 1)
            
            debug_info = {
                'num_columns': num_columns,
                'blocks': [{'x': b['x'], 'y': b['y'], 'width': b['width'], 'height': b['height'], 
                           'column': b['column']} for b in blocks]
            }
            return [formatted_text], debug_info
            
        return [formatted_text], None

def create_text_processor(tesseract_cmd: str = None) -> TextProcessor:
    """Factory function to create a TextProcessor instance."""
    ocr_processor = TesseractOCRProcessor(tesseract_cmd)
    return TextProcessor(ocr_processor)

def save_debug_image(image, blocks, output_dir='debug_outputs'):
    """Save debug image with detected blocks and layout information."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create a copy of the image for drawing
    debug_image = image.copy()
    
    # Define colors for different block types
    colors = {
        'text': (0, 255, 0),    # Green
        'header': (255, 0, 0),  # Blue
        'footer': (0, 0, 255),  # Red
        'title': (255, 255, 0), # Cyan
        'side_note': (0, 255, 255) # Yellow
    }
    
    # Draw rectangles around detected blocks
    for block in blocks:
        color = colors.get(block['block_type'], (0, 255, 0))
        cv2.rectangle(debug_image, (block['x'], block['y']), 
                     (block['x'] + block['width'], block['y'] + block['height']), 
                     color, 2)
        
        # Add block type and column info
        cv2.putText(debug_image, f"{block['block_type']} (col {block['column']})",
                   (block['x'], block['y'] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, color, 1)
    
    # Save the debug image
    output_path = os.path.join(output_dir, f'debug_{timestamp}.jpg')
    cv2.imwrite(output_path, debug_image)
    return output_path

def fix_word_breaks(text: str) -> str:
    """Fix word breaks and improve text formatting.
    
    Args:
        text: Input text with potential word breaks
        
    Returns:
        Formatted text with fixed word breaks
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix hyphenated words at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Fix common OCR errors
    text = re.sub(r'(\w+)\s*\n\s*(\w+)', r'\1 \2', text)  # Remove single line breaks between words
    
    # Fix single character lines
    text = re.sub(r'\n\s*(\w)\s*\n', r' \1 ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])(?=[^\s])', r'\1 ', text)
    
    # Fix spacing around parentheses
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    # Fix common OCR line break issues
    text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', text)  # Join words split across lines
    text = re.sub(r'(\w)\s*\n\s*([a-z])', r'\1\2', text)  # Join words split across lines (lowercase)
    
    # Fix spacing after periods that aren't sentence endings
    text = re.sub(r'\.\s+([a-z])', r'.\1', text)  # No space after period in abbreviations
    
    return text.strip()

def format_paragraphs(paragraphs: List[str]) -> str:
    """Format paragraphs with proper spacing and structure.
    
    Args:
        paragraphs: List of paragraph texts
        
    Returns:
        Formatted text with proper paragraph structure
    """
    formatted_text = []
    current_paragraph = []
    
    for para in paragraphs:
        # Skip empty paragraphs
        if not para.strip():
            continue
            
        # Fix word breaks in the paragraph
        fixed_para = fix_word_breaks(para)
        
        # Handle special block types
        if para.startswith('\n---') and para.endswith('---\n'):
            # Header - keep as is
            formatted_text.append(fixed_para)
        elif para.startswith('\n') and para.endswith('\n'):
            # Title or footer - keep as is
            formatted_text.append(fixed_para)
        else:
            # Regular paragraph - add to current paragraph
            current_paragraph.append(fixed_para)
            
            # If we have a complete paragraph, add it to formatted text
            if len(current_paragraph) > 0:
                formatted_text.append(' '.join(current_paragraph))
                current_paragraph = []
    
    # Add any remaining text
    if current_paragraph:
        formatted_text.append(' '.join(current_paragraph))
    
    # Join paragraphs with proper spacing
    return '\n\n'.join(formatted_text) 