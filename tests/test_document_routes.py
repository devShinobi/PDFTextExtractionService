import pytest
from flask import Flask
from io import BytesIO
from utils.text_processor import create_text_processor
from routes.document_routes import document_bp
import os
import pytesseract
import shutil

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@pytest.fixture(scope="function")
def app():
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = 'test_uploads'
    app.config['TEXT_OUTPUT_FOLDER'] = 'test_text_outputs'
    app.config['DEBUG_OUTPUT_FOLDER'] = 'test_debug_outputs'
    
    # Create test directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEXT_OUTPUT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DEBUG_OUTPUT_FOLDER'], exist_ok=True)
    
    # Initialize text processor with Tesseract path
    text_processor = create_text_processor(tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    # Register blueprint
    app.register_blueprint(document_bp)
    
    yield app
    
    # Cleanup after tests
    for directory in [app.config['UPLOAD_FOLDER'], 
                     app.config['TEXT_OUTPUT_FOLDER'], 
                     app.config['DEBUG_OUTPUT_FOLDER']]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def sample_pdf():
    # Create a simple PDF file
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.drawString(100, 750, "Test PDF Content")
    c.save()
    buffer.seek(0)
    return buffer

def test_upload_file_no_file(client):
    """Test upload endpoint with no file."""
    response = client.post('/api/upload')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_upload_file_empty_filename(client):
    """Test upload endpoint with empty filename."""
    response = client.post('/api/upload', data={'file': (BytesIO(b''), '')})
    assert response.status_code == 400
    assert b'No selected file' in response.data

def test_upload_file_success(client, sample_pdf):
    """Test successful file upload."""
    response = client.post('/api/upload', 
                          data={'file': (sample_pdf, 'test.pdf')},
                          content_type='multipart/form-data')
    assert response.status_code == 200
    assert b'File uploaded successfully' in response.data

def test_process_file_no_file(client):
    """Test process endpoint with no file."""
    response = client.post('/api/process')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_process_file_success(client, sample_pdf):
    """Test successful file processing."""
    response = client.post('/api/process',
                          data={'file': (sample_pdf, 'test.pdf')},
                          content_type='multipart/form-data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'paragraphs' in data
    assert 'count' in data

def test_extract_text_no_file(client):
    """Test extract-text endpoint with no file."""
    response = client.post('/api/extract-text')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_extract_text_success(client, sample_pdf):
    """Test successful text extraction."""
    response = client.post('/api/extract-text',
                          data={'file': (sample_pdf, 'test.pdf')},
                          content_type='multipart/form-data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'text' in data
    assert 'paragraph_count' in data
    
    # Add assertions to check text formatting
    text = data['text']
    # Check that text is not empty
    assert len(text) > 0
    # Check that text doesn't have single characters per line
    lines = text.split('\n')
    assert not any(len(line.strip()) == 1 for line in lines if line.strip())
    # Check that text contains the expected content
    assert "Test PDF Content" in text

def test_extract_text_with_real_pdf(client):
    """Test text extraction with a real PDF file."""
    # Path to your PDF file
    pdf_path = "C:/backend/Moonlighter.pdf"  # Replace with your PDF path
    
    # Open and read the PDF file
    with open(pdf_path, 'rb') as pdf_file:
        response = client.post('/api/extract-text',
                             data={'file': (pdf_file, 'real_test.pdf')},
                             content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'text' in data
    assert 'paragraph_count' in data
    
    # Print the extracted text for inspection
    print("\nExtracted Text:")
    print(data['text'])
    print(f"\nParagraph Count: {data['paragraph_count']}")
    
    # Basic assertions
    text = data['text']
    assert len(text) > 0
    lines = text.split('\n')
    assert not any(len(line.strip()) == 1 for line in lines if line.strip())

def test_debug_processing_no_file(client):
    """Test debug endpoint with no file."""
    response = client.post('/api/debug')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_debug_processing_success(client, sample_pdf):
    """Test successful debug processing."""
    response = client.post('/api/debug',
                          data={'file': (sample_pdf, 'test.pdf')},
                          content_type='multipart/form-data')
    assert response.status_code == 200
    data = response.get_json()
    assert 'results' in data
    assert 'total_pages' in data

def test_convert_to_txt_no_file(client):
    """Test convert-to-txt endpoint with no file."""
    response = client.post('/api/convert-to-txt')
    assert response.status_code == 400
    assert b'No file part' in response.data

def test_convert_to_txt_success(client, sample_pdf):
    """Test successful conversion to text file."""
    response = client.post('/api/convert-to-txt',
                          data={'file': (sample_pdf, 'test.pdf')},
                          content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.mimetype == 'text/plain'
    assert 'attachment' in response.headers['Content-Disposition']
    
    # Verify the response content
    content = response.get_data()
    assert len(content) > 0
    assert b"Test PDF Content" in content 