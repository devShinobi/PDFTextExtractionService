from flask import Flask
import pytesseract
from utils.file_handler import ensure_directories
from routes.document_routes import document_bp

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def create_app():
    app = Flask(__name__)
    
    # Ensure required directories exist
    ensure_directories()
    
    # Register blueprints
    app.register_blueprint(document_bp, url_prefix='/api')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000) 