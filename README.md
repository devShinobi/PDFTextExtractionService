# PDF Paragraph Extractor

This project provides a robust pipeline for extracting paragraphs and text from PDF documents. It's designed to intelligently analyze document layouts, including multi-column formats, to ensure accurate text extraction in the correct reading order.

## Features

- **PDF to Image Conversion**: Converts each page of a PDF into a high-resolution image for analysis.
- **Intelligent Layout Analysis**:
    - Distinguishes between headings and body text based on font size.
    - Automatically detects multi-column layouts.
    - Stitches individual words into paragraphs based on proximity.
- **Configurable Margin Exclusion**: Allows for the exclusion of headers and footers to focus on the main content.
- **Accurate Text Extraction**: Uses Tesseract OCR to extract text from detected paragraphs.
- **Multiple Output Formats**:
    - **Annotated Images**: Saves images of each page with bounding boxes drawn around detected paragraphs.
    - **Detailed Text File**: A `.txt` file with the extracted text, including metadata like page and paragraph number.
    - **Clean Text File**: A clean `.txt` file containing only the extracted text for easy reading or further processing.
- **Debug Mode**: Optional debug mode to save bounding box images for visual verification of text detection.
- **Parallel Processing**: Batch processing with configurable parallel workers for improved performance.
- **Error Handling**: Robust error handling that continues processing other files if one fails.
- **Timing Statistics**: Detailed timing information and performance metrics for batch processing.

## Requirements

- Python 3.8+
- Tesseract OCR Engine

### Tesseract Installation

This tool requires the Tesseract OCR engine to be installed on your system.

- **Windows**: Download and install from the [UB Mannheim repository](https://github.com/UB-Mannheim/tesseract/wiki). Remember to add Tesseract to your system's PATH.
- **macOS**: `brew install tesseract`
- **Linux (Debian/Ubuntu)**: `sudo apt-get install tesseract-ocr`


## How to Use

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd pdf-paragraph-extractor
    ```

2.  **Install the Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Update Tesseract Path (if necessary):**
    If you installed Tesseract in a non-standard location on Windows, you may need to update the path at the top of `utils/image_processor.py`:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r'C:\path\to\your\tesseract.exe'
    ```

4.  **Run the pipeline:**
    The script is run from the command line, pointing to your PDF file.

    **Basic Usage:**
    ```bash
    python main.py "path/to/your/document.pdf"
    ```

    **Customizing Margins:**
    You can customize the top and bottom margins to exclude headers and footers.

    ```bash
    python main.py "my_doc.pdf" --top_margin 0.1 --bottom_margin 0.12
    ```

    **Separating Output Types:**
    You can specify separate directories for intermediate processing outputs (images) and final text outputs:

    ```bash
    python main.py "my_doc.pdf" --output_dir "./intermediate" --final_output_dir "./final_text"
    ```

    **Debug Mode:**
    Enable debug mode to save bounding box images showing detected text regions:

    ```bash
    python main.py "my_doc.pdf" --debug
    ```

    **Parallel Batch Processing:**
    For batch processing, you can control the number of parallel workers:

    ```bash
    python main.py "./input_folder" --max_workers 5
    ```

    - `--output_dir`: Specify a directory for the output files (defaults to `output/`).
    - `--final_output_dir`: Specify a separate directory for final text output files (optional).
    - `--debug`: Enable debug mode to save bounding box images (optional, defaults to False).
    - `--max_workers`: Maximum number of parallel workers for batch processing (default: 10).
    - `--top_margin`: Percentage of the page to exclude from the top (e.g., `0.1` for 10%).
    - `--bottom_margin`: Percentage of the page to exclude from the bottom.
    - `--ignore_vertical`: A flag to ignore vertically oriented text.

## Output

After running, you will find output files in the specified output directory (default is `output/`).

**With debug mode enabled (`--debug`):**
- `page_{#}_detected.png`: An image of each page with green bounding boxes around the detected paragraphs.
- `{pdf_name}_extracted_text.txt`: A detailed text file with metadata.

**Without debug mode (default):**
- `{pdf_name}_extracted_text.txt`: A detailed text file with metadata only (no debug images).

### Output Directory Structure

**Default behavior (all outputs in same directory, debug disabled):**
```
output/
└── document_extracted_text.txt
```

**With debug mode enabled:**
```
output/
├── page_1_detected.png
├── page_2_detected.png
└── document_extracted_text.txt
```

**With separate final output directory:**
```
intermediate/
├── page_1_detected.png
└── page_2_detected.png

final_text/
└── document_extracted_text.txt
```

**Batch processing with separate directories:**
```
batch_intermediate/
├── document1/
│   ├── page_1_detected.png
│   └── page_2_detected.png
└── document2/
    ├── page_1_detected.png
    └── page_2_detected.png

batch_final/
├── document1_extracted_text.txt
└── document2_extracted_text.txt
```

## Project Structure

```
.
├── app.py              # Main application
├── routes/            # API routes
├── utils/             # Utility functions
│   └── text_processor.py  # Text processing logic
├── tests/             # Test files
└── requirements.txt   # Dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 