import pytest
import os
import shutil
from PIL import Image
import io
from utils.file_handler import (
    ensure_directories,
    save_uploaded_file,
    convert_pdf_to_images,
    cleanup_temp_file,
    save_text_to_file
)
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def sample_pdf(temp_dir):
    """Create a valid PDF file for testing."""
    pdf_path = temp_dir / "test.pdf"
    
    # Create a PDF with some content
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Test PDF Content")
    c.save()
    
    return str(pdf_path)

def test_ensure_directories():
    """Test directory creation."""
    # Test directory creation
    ensure_directories()
    assert os.path.exists('uploads')
    assert os.path.exists('text_outputs')
    assert os.path.exists('debug_outputs')

def test_save_uploaded_file():
    """Test saving uploaded file."""
    # Create test file content
    content = b"Test file content"
    file_obj = BytesIO(content)
    file_obj.filename = "test.txt"
    
    # Create uploads directory
    os.makedirs('uploads', exist_ok=True)
    
    # Save file
    filepath = save_uploaded_file(file_obj, "test.txt")
    
    # Verify file was saved
    assert os.path.exists(filepath)
    with open(filepath, 'rb') as f:
        assert f.read() == content

def test_cleanup_temp_file(temp_dir):
    """Test temporary file cleanup."""
    # Create a temporary file
    temp_file = temp_dir / "temp.txt"
    temp_file.write_text("test content")
    
    # Clean up
    cleanup_temp_file(str(temp_file))
    
    # Verify file was removed
    assert not os.path.exists(temp_file)

def test_save_text_to_file():
    """Test saving text to file."""
    # Create text_outputs directory
    os.makedirs('text_outputs', exist_ok=True)
    
    text = "Test text content"
    filename = "test.txt"
    
    # Save text
    filepath, saved_filename = save_text_to_file(text, filename)
    
    # Verify file was saved
    assert os.path.exists(filepath)
    with open(filepath, 'r') as f:
        assert f.read() == text
    assert saved_filename == "test.txt"

@pytest.mark.integration
def test_convert_pdf_to_images(temp_dir, sample_pdf):
    """Test PDF to image conversion."""
    # Test conversion
    images, temp_path = convert_pdf_to_images(sample_pdf)
    
    # Verify results
    assert len(images) > 0
    assert all(isinstance(img, Image.Image) for img in images)
    
    # Clean up
    if temp_path:
        cleanup_temp_file(temp_path) 