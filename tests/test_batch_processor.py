import os
import shutil
import pytest
from utils.batch_processor import BatchProcessor
from utils.image_processor import ImageProcessor # To check instance
import sys

# Add project root to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_pdf_path():
    """
    Provides the path to a sample PDF file for testing.
    Skips the test if the file doesn't exist.
    """
    # Assuming Moonlighter.pdf is in the project root
    pdf_name = "Moonlighter.pdf"
    # Go up two directories from tests/ to reach the project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(project_root, pdf_name)
    
    if not os.path.exists(path):
        pytest.skip(f"Sample PDF for testing not found at '{path}'. Place it in the project root.")
    return path

def test_batch_processing(tmp_path, sample_pdf_path):
    """
    Tests the batch processing of a folder containing multiple PDF files.
    """
    # 1. Set up temporary directory structure
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # 2. Prepare test files
    shutil.copy(sample_pdf_path, input_dir / "doc1.pdf")
    shutil.copy(sample_pdf_path, input_dir / "doc2.pdf")
    (input_dir / "should_be_ignored.txt").touch()

    assert len(os.listdir(input_dir)) == 3, "Test setup failed: incorrect number of files in input_dir"

    # 3. Run the batch processor
    batch_processor = BatchProcessor()
    batch_processor.process_folder(
        input_folder=str(input_dir), 
        output_folder=str(output_dir)
    )

    # 4. Assert the output directory structure
    processed_folders = os.listdir(output_dir)
    assert "doc1" in processed_folders, "Output folder for doc1 was not created"
    assert "doc2" in processed_folders, "Output folder for doc2 was not created"
    assert "should_be_ignored" not in processed_folders, "A non-PDF file was incorrectly processed"
    assert len(processed_folders) == 2, "Incorrect number of folders in the output directory"

    # 5. Assert that key output files were created for each PDF
    doc1_outputs = os.listdir(output_dir / "doc1")
    assert "doc1_extracted_text.txt" in doc1_outputs, "Extracted text file for doc1 is missing"
    assert any(f.startswith("page_") and f.endswith("_detected.png") for f in doc1_outputs), "Detected image for doc1 is missing"

    doc2_outputs = os.listdir(output_dir / "doc2")
    assert "doc2_extracted_text.txt" in doc2_outputs, "Extracted text file for doc2 is missing"
    assert any(f.startswith("page_") and f.endswith("_detected.png") for f in doc2_outputs), "Detected image for doc2 is missing"

def test_batch_processing_with_final_output_dir(tmp_path, sample_pdf_path):
    """
    Tests the batch processing with separate final output directory for text files.
    """
    # 1. Set up temporary directory structure
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    final_output_dir = tmp_path / "final_output"
    input_dir.mkdir()
    output_dir.mkdir()
    final_output_dir.mkdir()

    # 2. Prepare test files
    shutil.copy(sample_pdf_path, input_dir / "doc1.pdf")
    shutil.copy(sample_pdf_path, input_dir / "doc2.pdf")

    # 3. Run the batch processor with final_output_dir
    batch_processor = BatchProcessor()
    batch_processor.process_folder(
        input_folder=str(input_dir), 
        output_folder=str(output_dir),
        final_output_dir=str(final_output_dir)
    )

    # 4. Assert the output directory structure
    processed_folders = os.listdir(output_dir)
    assert "doc1" in processed_folders, "Output folder for doc1 was not created"
    assert "doc2" in processed_folders, "Output folder for doc2 was not created"

    # 5. Assert that intermediate outputs are in the output directory
    doc1_outputs = os.listdir(output_dir / "doc1")
    assert any(f.startswith("page_") and f.endswith("_detected.png") for f in doc1_outputs), "Detected image for doc1 is missing"
    assert "doc1_extracted_text.txt" not in doc1_outputs, "Text file should not be in intermediate output directory"

    doc2_outputs = os.listdir(output_dir / "doc2")
    assert any(f.startswith("page_") and f.endswith("_detected.png") for f in doc2_outputs), "Detected image for doc2 is missing"
    assert "doc2_extracted_text.txt" not in doc2_outputs, "Text file should not be in intermediate output directory"

    # 6. Assert that final text outputs are in the final_output_dir
    final_outputs = os.listdir(final_output_dir)
    assert "doc1_extracted_text.txt" in final_outputs, "Extracted text file for doc1 is missing from final output directory"
    assert "doc2_extracted_text.txt" in final_outputs, "Extracted text file for doc2 is missing from final output directory"

def test_batch_processing_parallel_with_debug(tmp_path, sample_pdf_path):
    """
    Tests the parallel batch processing with debug mode enabled.
    """
    # 1. Set up temporary directory structure
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    final_output_dir = tmp_path / "final_output"
    input_dir.mkdir()
    output_dir.mkdir()
    final_output_dir.mkdir()

    # 2. Prepare test files (create multiple copies to test parallel processing)
    for i in range(3):
        shutil.copy(sample_pdf_path, input_dir / f"doc{i+1}.pdf")

    # 3. Run the batch processor with parallel processing and debug mode
    batch_processor = BatchProcessor()
    result = batch_processor.process_folder(
        input_folder=str(input_dir), 
        output_folder=str(output_dir),
        final_output_dir=str(final_output_dir),
        max_workers=2,  # Use 2 workers for testing
        debug=True      # Enable debug mode
    )

    # 4. Assert the result structure
    assert result is not None, "Batch processing should return a result dictionary"
    assert 'total_files' in result, "Result should contain total_files"
    assert 'successful_files' in result, "Result should contain successful_files"
    assert 'failed_files' in result, "Result should contain failed_files"
    assert 'total_time' in result, "Result should contain total_time"
    assert 'success_rate' in result, "Result should contain success_rate"

    # 5. Assert processing results
    assert result['total_files'] == 3, "Should have processed 3 files"
    assert len(result['successful_files']) == 3, "All 3 files should be processed successfully"
    assert len(result['failed_files']) == 0, "No files should have failed"
    assert result['success_rate'] == 1.0, "Success rate should be 100%"
    assert result['total_time'] > 0, "Total time should be positive"

    # 6. Assert that debug images were created (since debug=True)
    for i in range(3):
        doc_output_dir = output_dir / f"doc{i+1}"
        assert doc_output_dir.exists(), f"Output directory for doc{i+1} should exist"
        
        # Check for debug images (should exist when debug=True)
        doc_outputs = os.listdir(doc_output_dir)
        assert any(f.startswith("page_") and f.endswith("_detected.png") for f in doc_outputs), f"Debug images for doc{i+1} should exist"

    # 7. Assert that final text outputs are in the final_output_dir
    final_outputs = os.listdir(final_output_dir)
    for i in range(3):
        assert f"doc{i+1}_extracted_text.txt" in final_outputs, f"Extracted text file for doc{i+1} is missing from final output directory"

    # 8. Assert timing information in successful files
    for successful_file in result['successful_files']:
        assert 'processing_time' in successful_file, "Successful file should have processing_time"
        assert successful_file['processing_time'] > 0, "Processing time should be positive"
        assert 'output_dir' in successful_file, "Successful file should have output_dir" 