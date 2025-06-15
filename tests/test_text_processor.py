import pytest
from PIL import Image
import numpy as np
from utils.text_processor import (
    LayoutAnalyzer,
    TextBlock,
    ImageProcessor,
    TextProcessor,
    TesseractOCRProcessor,
    create_text_processor
)

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a white image with some black text
    img = Image.new('RGB', (800, 600), color='white')
    return img

@pytest.fixture
def layout_analyzer():
    """Create a LayoutAnalyzer instance for testing."""
    return LayoutAnalyzer((600, 800))

@pytest.fixture
def mock_ocr_processor():
    """Create a mock OCR processor for testing."""
    class MockOCRProcessor:
        def process_image(self, image):
            return "Test text"
    return MockOCRProcessor()

@pytest.fixture
def image_processor(mock_ocr_processor):
    """Create an ImageProcessor instance for testing."""
    return ImageProcessor(mock_ocr_processor)

@pytest.fixture
def text_processor(mock_ocr_processor):
    """Create a TextProcessor instance for testing."""
    return TextProcessor(mock_ocr_processor)

def test_layout_analyzer_initialization():
    """Test LayoutAnalyzer initialization."""
    analyzer = LayoutAnalyzer((600, 800))
    assert analyzer.height == 600
    assert analyzer.width == 800
    assert analyzer.header_height == 60  # 10% of height
    assert analyzer.footer_height == 60  # 10% of height
    assert analyzer.margin == 40  # 5% of width

def test_is_side_note():
    """Test side note detection."""
    analyzer = LayoutAnalyzer((600, 800))
    
    # Test a side note (narrow block)
    narrow_block = TextBlock(x=10, y=100, width=50, height=100)
    assert analyzer.is_side_note(narrow_block)
    
    # Test a main content block
    main_block = TextBlock(x=100, y=100, width=300, height=100)
    assert not analyzer.is_side_note(main_block)

def test_detect_columns():
    """Test column detection."""
    analyzer = LayoutAnalyzer((600, 800))
    
    # Test single column
    single_col_blocks = [(400, 300), (400, 300)]  # (x_center, width)
    assert analyzer.detect_columns(tuple(single_col_blocks)) == 1
    
    # Test two columns with more distinct positions
    two_col_blocks = [(200, 300), (600, 300), (200, 300), (600, 300)]  # (x_center, width)
    assert analyzer.detect_columns(tuple(two_col_blocks)) == 2

def test_assign_columns():
    """Test column assignment."""
    analyzer = LayoutAnalyzer((600, 800))
    blocks = [
        TextBlock(x=100, y=100, width=300, height=100),  # Main content
        TextBlock(x=10, y=100, width=50, height=100),    # Side note
        TextBlock(x=500, y=100, width=300, height=100)   # Main content
    ]
    
    result = analyzer.assign_columns(blocks, 2)
    assert result[0].column == 0  # First main content block
    assert result[1].column == -1  # Side note
    assert result[2].column == 1  # Second main content block

def test_identify_block_types():
    """Test block type identification."""
    analyzer = LayoutAnalyzer((600, 800))
    blocks = [
        TextBlock(x=100, y=50, width=300, height=100),   # Header
        TextBlock(x=100, y=500, width=300, height=100),  # Footer
        TextBlock(x=100, y=100, width=300, height=100)   # Regular text
    ]
    
    result = analyzer.identify_block_types(blocks)
    assert result[0].block_type == "header"
    assert result[1].block_type == "footer"
    assert result[2].block_type == "text"

def test_image_processor_preprocess(sample_image, image_processor):
    """Test image preprocessing."""
    processed = image_processor.preprocess_image(sample_image)
    assert isinstance(processed, np.ndarray)
    assert len(processed.shape) == 2  # Should be grayscale

def test_image_processor_detect_blocks(sample_image, image_processor):
    """Test text block detection."""
    processed = image_processor.preprocess_image(sample_image)
    blocks = image_processor.detect_text_blocks(processed)
    assert isinstance(blocks, list)
    # Since we're using a blank image, there should be no text blocks
    assert len(blocks) == 0

def test_text_processor_process_image(sample_image, text_processor):
    """Test text processing with layout."""
    paragraphs, debug_info = text_processor.process_image_with_layout(sample_image)
    assert isinstance(paragraphs, list)
    assert debug_info is None

def test_text_processor_process_image_debug(sample_image, text_processor):
    """Test text processing with debug info."""
    paragraphs, debug_info = text_processor.process_image_with_layout(sample_image, debug=True)
    assert isinstance(paragraphs, list)
    assert isinstance(debug_info, dict)
    assert 'initial_blocks' in debug_info
    assert 'merged_blocks' in debug_info

def test_create_text_processor():
    """Test text processor factory function."""
    processor = create_text_processor()
    assert isinstance(processor, TextProcessor)
    assert isinstance(processor.ocr_processor, TesseractOCRProcessor) 