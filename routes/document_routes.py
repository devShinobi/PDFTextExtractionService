from flask import Blueprint, request, jsonify, send_file
from PIL import Image
import os
import logging
from utils.text_processor import create_text_processor
from utils.file_handler import (
    ensure_directories,
    save_uploaded_file,
    convert_pdf_to_images,
    cleanup_temp_file,
    save_text_to_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

document_bp = Blueprint('document', __name__, url_prefix='/api')

# Create text processor instance
text_processor = create_text_processor()

def process_document(file, debug=False):
    """Common function to process documents (PDF or image).
    
    Args:
        file: File object from request
        debug: Whether to return debug information
        
    Returns:
        tuple: (results, temp_path) where results is either text or debug info
    """
    filename = file.filename
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    logger.info(f"Processing file: {filename}")
    
    temp_path = None
    try:
        if filename.lower().endswith('.pdf'):
            logger.info("Converting PDF to images")
            images, temp_path = convert_pdf_to_images(filepath)
            results = []
            for i, img in enumerate(images):
                logger.info(f"Processing page {i+1}")
                if debug:
                    paragraphs, debug_info = text_processor.process_image_with_layout(img, debug=True)
                    results.append({
                        'page': i + 1,
                        'paragraphs': paragraphs,
                        'initial_blocks': debug_info['initial_blocks'],
                        'merged_blocks': debug_info['merged_blocks'],
                        'initial_blocks_image': debug_info['initial_blocks_image'],
                        'merged_blocks_image': debug_info['merged_blocks_image']
                    })
                else:
                    paragraphs, _ = text_processor.process_image_with_layout(img)
                    results.append(f"\n--- Page {i+1} ---\n")
                    results.extend(paragraphs)
        else:
            logger.info("Processing image file")
            img = Image.open(filepath)
            if debug:
                paragraphs, debug_info = text_processor.process_image_with_layout(img, debug=True)
                results = [{
                    'paragraphs': paragraphs,
                    'initial_blocks': debug_info['initial_blocks'],
                    'merged_blocks': debug_info['merged_blocks'],
                    'initial_blocks_image': debug_info['initial_blocks_image'],
                    'merged_blocks_image': debug_info['merged_blocks_image']
                }]
            else:
                paragraphs, _ = text_processor.process_image_with_layout(img)
                results = paragraphs
        
        return results, temp_path
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)

@document_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload a file for processing."""
    logger.info("Received file upload request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        logger.info(f"File saved successfully: {filename}")
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})

@document_bp.route('/process', methods=['POST'])
def process_file():
    """Process an uploaded file and return structured text."""
    logger.info("Received process request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        results, _ = process_document(file)
        return jsonify({
            'paragraphs': results,
            'count': len(results)
        })

@document_bp.route('/extract-text', methods=['POST'])
def extract_text():
    """Extract text from an uploaded file and return formatted text."""
    logger.info("Received text extraction request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        results, _ = process_document(file)
        formatted_text = "\n\n".join(results)
        return jsonify({
            'text': formatted_text,
            'paragraph_count': len(results)
        })

@document_bp.route('/debug', methods=['POST'])
def debug_processing():
    """Debug endpoint to visualize text block detection."""
    logger.info("Received debug request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        results, _ = process_document(file, debug=True)
        return jsonify({
            'results': results,
            'total_pages': len(results)
        })

@document_bp.route('/convert-to-txt', methods=['POST'])
def convert_to_txt():
    """Convert uploaded file to text file."""
    logger.info("Received convert to txt request")
    
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        results, _ = process_document(file)
        txt_path, txt_filename = save_text_to_file('\n\n'.join(results), file.filename)
        return send_file(txt_path, as_attachment=True, download_name=txt_filename) 