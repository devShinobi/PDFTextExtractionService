# PDF Text Extraction Service

A robust PDF text extraction service that uses OCR and layout analysis to extract text from PDF documents while preserving the document's structure.

## Features

- PDF to text conversion with layout preservation
- Multi-column text detection and processing
- OCR with Tesseract
- Layout analysis for complex documents
- Debug visualization of text blocks
- REST API interface

## Requirements

- Python 3.8+
- Tesseract OCR
- OpenCV
- Flask
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

## Usage

1. Start the server:
```bash
python app.py
```

2. Send a POST request to `/api/extract-text` with a PDF file:
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/api/extract-text
```

## Development

- Run tests: `pytest`
- Debug mode: Set `debug=True` in the API call to get visualization of text blocks

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