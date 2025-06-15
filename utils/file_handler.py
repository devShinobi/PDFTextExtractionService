import os
import tempfile
from pdf2image import convert_from_path
from PIL import Image
import logging
from functools import lru_cache
import threading
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-local storage for temporary files
thread_local = threading.local()

def get_thread_local_temp():
    """Get thread-local temporary file storage."""
    if not hasattr(thread_local, 'temp_files'):
        thread_local.temp_files = []
    return thread_local.temp_files

def ensure_directories():
    """Ensure required directories exist."""
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('text_outputs', exist_ok=True)
    os.makedirs('debug_outputs', exist_ok=True)

def save_uploaded_file(file, filename):
    """Save an uploaded file.
    
    Args:
        file: File object (BytesIO or file-like object)
        filename: Name of the file
        
    Returns:
        str: Path to saved file
    """
    filepath = os.path.join('uploads', filename)
    
    # Handle BytesIO objects
    if isinstance(file, io.BytesIO):
        with open(filepath, 'wb') as f:
            f.write(file.getvalue())
    else:
        # Handle regular file objects
        file.save(filepath)
    
    return filepath

@lru_cache(maxsize=32)
def get_pdf_conversion_params():
    """Get optimized parameters for PDF conversion."""
    return {
        'fmt': 'jpeg',
        'thread_count': os.cpu_count() or 4,
        'use_pdftocairo': True,
        'grayscale': False,
        'size': (None, None),  # Maintain original size
        'dpi': 300  # Optimal DPI for OCR
    }

def convert_pdf_to_images(pdf_path):
    """Convert PDF to images.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        tuple: (list of PIL Images, temp directory path)
    """
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        return images, None
    except Exception as e:
        logger.error(f"Error converting PDF: {str(e)}")
        raise

def cleanup_temp_file(filepath):
    """Clean up temporary file.
    
    Args:
        filepath: Path to file to clean up
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")

def cleanup_all_temp_files():
    """Clean up all temporary files for the current thread."""
    temp_files = get_thread_local_temp()
    for filepath in temp_files[:]:  # Copy list to avoid modification during iteration
        cleanup_temp_file(filepath)
    temp_files.clear()

def save_text_to_file(text, original_filename):
    """Save text to file.
    
    Args:
        text: Text to save
        original_filename: Original filename (used to generate output filename)
        
    Returns:
        tuple: (filepath, filename)
    """
    # Generate output filename
    base_name = os.path.splitext(original_filename)[0]
    filename = f"{base_name}.txt"
    filepath = os.path.join('text_outputs', filename)
    
    # Save text
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return filepath, filename 