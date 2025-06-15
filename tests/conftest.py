import pytest
from flask import Flask
from PIL import Image
import io
import os
import shutil
from routes.document_routes import document_bp

@pytest.fixture
def app():
    """Create a Flask app for testing."""
    app = Flask(__name__)
    app.register_blueprint(document_bp)
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path

@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    # Create test directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('text_outputs', exist_ok=True)
    os.makedirs('debug_outputs', exist_ok=True)
    
    yield
    
    # Cleanup test directories
    shutil.rmtree('uploads', ignore_errors=True)
    shutil.rmtree('text_outputs', ignore_errors=True)
    shutil.rmtree('debug_outputs', ignore_errors=True) 