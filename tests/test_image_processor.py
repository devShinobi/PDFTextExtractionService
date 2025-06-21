import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_processor import ImageProcessor

def test_pdf_processing():
    """Test the complete PDF processing pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python test_image_processor.py <pdf_path> [exclude_bottom_percent] [exclude_top_percent]")
        print("Example: python test_image_processor.py document.pdf 0.1 0.07")
        return
    
    pdf_path = sys.argv[1]
    
    # Get exclude bottom percentage (default 0.07 = 7%)
    exclude_bottom_percent = 0.07
    if len(sys.argv) >= 3:
        try:
            exclude_bottom_percent = float(sys.argv[2])
            if exclude_bottom_percent < 0 or exclude_bottom_percent > 1:
                print("Error: exclude_bottom_percent must be between 0.0 and 1.0")
                return
        except ValueError:
            print("Error: exclude_bottom_percent must be a number")
            return
    
    # Get exclude top percentage (default 0.07 = 7%)
    exclude_top_percent = 0.07
    if len(sys.argv) >= 4:
        try:
            exclude_top_percent = float(sys.argv[3])
            if exclude_top_percent < 0 or exclude_top_percent > 1:
                print("Error: exclude_top_percent must be between 0.0 and 1.0")
                return
        except ValueError:
            print("Error: exclude_top_percent must be a number")
            return
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process the PDF
    print(f"Processing PDF: {pdf_path}")
    print(f"Excluding top {exclude_top_percent*100}% of each page")
    print(f"Excluding bottom {exclude_bottom_percent*100}% of each page")
    paragraphs = processor.process_pdf(pdf_path, output_dir="test_output", 
                                     exclude_bottom_percent=exclude_bottom_percent,
                                     exclude_top_percent=exclude_top_percent)
    
    # Print results
    print(f"\nExtracted {len(paragraphs)} paragraphs:")
    for i, para in enumerate(paragraphs):
        print(f"\nParagraph {i+1} (Page {para['page']}):")
        print(f"  Position: ({para['x']}, {para['y']}) - {para['width']}x{para['height']}")
        print(f"  Text length: {para['text_length']} characters")
        print(f"  Text: {para['text'][:100]}{'...' if len(para['text']) > 100 else ''}")

def test_pdf_processing_with_final_output_dir():
    """Test the PDF processing pipeline with separate final output directory."""
    if len(sys.argv) < 2:
        print("Usage: python test_image_processor.py <pdf_path> [exclude_bottom_percent] [exclude_top_percent]")
        print("Example: python test_image_processor.py document.pdf 0.1 0.07")
        return
    
    pdf_path = sys.argv[1]
    
    # Get exclude bottom percentage (default 0.07 = 7%)
    exclude_bottom_percent = 0.07
    if len(sys.argv) >= 3:
        try:
            exclude_bottom_percent = float(sys.argv[2])
            if exclude_bottom_percent < 0 or exclude_bottom_percent > 1:
                print("Error: exclude_bottom_percent must be between 0.0 and 1.0")
                return
        except ValueError:
            print("Error: exclude_bottom_percent must be a number")
            return
    
    # Get exclude top percentage (default 0.07 = 7%)
    exclude_top_percent = 0.07
    if len(sys.argv) >= 4:
        try:
            exclude_top_percent = float(sys.argv[3])
            if exclude_top_percent < 0 or exclude_top_percent > 1:
                print("Error: exclude_top_percent must be between 0.0 and 1.0")
                return
        except ValueError:
            print("Error: exclude_top_percent must be a number")
            return
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process the PDF with separate final output directory
    print(f"Processing PDF: {pdf_path}")
    print(f"Excluding top {exclude_top_percent*100}% of each page")
    print(f"Excluding bottom {exclude_bottom_percent*100}% of each page")
    print("Using separate final output directory for text files")
    
    paragraphs = processor.process_pdf(
        pdf_path, 
        output_dir="test_output", 
        final_output_dir="test_final_output",
        exclude_bottom_percent=exclude_bottom_percent,
        exclude_top_percent=exclude_top_percent
    )
    
    # Print results
    print(f"\nExtracted {len(paragraphs)} paragraphs:")
    for i, para in enumerate(paragraphs):
        print(f"\nParagraph {i+1} (Page {para['page']}):")
        print(f"  Position: ({para['x']}, {para['y']}) - {para['width']}x{para['height']}")
        print(f"  Text length: {para['text_length']} characters")
        print(f"  Text: {para['text'][:100]}{'...' if len(para['text']) > 100 else ''}")
    
    print(f"\nIntermediate outputs saved in: test_output")
    print(f"Final text outputs saved in: test_final_output")

if __name__ == "__main__":
    test_pdf_processing() 