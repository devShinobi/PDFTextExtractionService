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

## Performance & GPU Acceleration

The text extraction process is CPU-intensive. To improve performance, this tool offers two acceleration methods:

### Multi-Core CPU Parallelization

For batch processing multiple files, the application uses a `ProcessPoolExecutor` to distribute the workload across multiple CPU cores. This significantly speeds up the processing of large numbers of documents by running them in parallel.
- **How it works:** Each PDF is processed in a separate process, allowing the operating system to schedule these tasks across all available CPU cores.
- **Controlling it:** Use the `--max_workers` flag to set the number of parallel processes. For optimal performance, it's often best to set this to the number of CPU cores on your machine.
- **Choosing the Mode:** Use the `--parallel_mode` flag to switch between `thread` (default) and `process` (multi-core) execution. While `process` is typically better for CPU-bound tasks, `thread` may be faster on systems where process creation is slow or where the task has significant I/O, such as reading files or loading models. It is recommended to benchmark both modes on your specific workload.
    ```bash
    # Process a folder using 10 threads (default mode)
    python main.py "./input_folder" --max_workers 10
    
    # Process a folder using 4 CPU cores (for comparison)
    python main.py "./input_folder" --max_workers 4 --parallel_mode process
    ```

### GPU Acceleration for Context Analysis

The semantic context analysis (`--analyze_context`) can be further accelerated using an NVIDIA GPU. This offloads the heavy machine learning computations from the CPU.

**How to Enable GPU Acceleration**

1.  **Hardware/Driver Requirement:** You need an NVIDIA GPU with the appropriate CUDA drivers installed on your system.
2.  **Install GPU-Enabled PyTorch:** The standard `torch` package in `requirements.txt` is for CPU only. To enable GPU support, you must install the PyTorch version that matches your system's CUDA version.
    -   First, uninstall the existing CPU version: `pip uninstall torch torchvision torchaudio`
    -   Visit the [Official PyTorch Website](https://pytorch.org/get-started/locally/) to find the correct installation command for your specific setup (OS, package manager, CUDA version). For example, a common command for CUDA 11.8 is:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
3.  **Run with Context Analysis:** Execute the script with the `--analyze_context` flag.
    ```bash
    python main.py "path/to/your/document.pdf" --analyze_context
    ```

The application will automatically detect the GPU and use it for the natural language processing tasks, which can result in a **10x or greater speedup** for the context analysis step. The rest of the pipeline will continue to run in parallel on the CPU.

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
    - `--max_workers`: The maximum number of parallel threads or processes to use for batch processing.
    - `--parallel_mode`: The parallel execution mode. `thread` (default) is often faster for mixed I/O and CPU workloads, while `process` is better for purely CPU-bound tasks.
    - `--min_region_confidence`: The minimum average confidence score (0-100) for a detected text region to be included. Lower values may include text from images. (Default: 50).
    - `--cleanup`/`--no-cleanup`: Enable or disable automatic text cleanup (de-hyphenation, etc.). Enabled by default. (Default: True).
    - `--analyze_context`: Enable semantic analysis to find and tag outlier text regions. Disabled by default.
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