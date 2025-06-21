#!/usr/bin/env python3
"""
Example script demonstrating the new final_output_dir parameter functionality.
This shows how to separate intermediate processing outputs from final text outputs.
"""

import os
from utils.image_processor import ImageProcessor
from utils.batch_processor import BatchProcessor

def example_single_file_processing():
    """Example of processing a single PDF file with separate output directories."""
    print("=== Single File Processing Example ===")
    
    # Example PDF path (you'll need to provide your own PDF file)
    pdf_path = "example.pdf"  # Change this to your PDF file path
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable to point to an existing PDF file.")
        return
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process with separate directories
    print(f"Processing: {pdf_path}")
    print("Intermediate outputs (images) will be saved to: ./intermediate_output")
    print("Final text outputs will be saved to: ./final_text_output")
    
    try:
        paragraphs = processor.process_pdf(
            pdf_path=pdf_path,
            output_dir="./intermediate_output",           # For intermediate processing outputs
            final_output_dir="./final_text_output",      # For final text outputs
            exclude_bottom_percent=0.07,
            exclude_top_percent=0.07
        )
        
        print(f"\nSuccessfully extracted {len(paragraphs)} text blocks.")
        print("Check the following directories:")
        print("  - ./intermediate_output/ (contains processed images)")
        print("  - ./final_text_output/ (contains extracted text files)")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def example_batch_processing():
    """Example of batch processing multiple PDF files with separate output directories."""
    print("\n=== Batch Processing Example ===")
    
    # Example input folder (you'll need to provide your own folder with PDF files)
    input_folder = "./input_pdfs"  # Change this to your input folder path
    
    if not os.path.exists(input_folder):
        print(f"Input folder not found: {input_folder}")
        print("Please update the input_folder variable to point to an existing folder with PDF files.")
        return
    
    # Initialize batch processor
    batch_processor = BatchProcessor()
    
    # Process with separate directories
    print(f"Processing all PDFs in: {input_folder}")
    print("Intermediate outputs will be saved to: ./batch_intermediate_output")
    print("Final text outputs will be saved to: ./batch_final_output")
    
    try:
        batch_processor.process_folder(
            input_folder=input_folder,
            output_folder="./batch_intermediate_output",    # For intermediate processing outputs
            final_output_dir="./batch_final_output",       # For final text outputs
            exclude_bottom_percent=0.07,
            exclude_top_percent=0.07
        )
        
        print("\nBatch processing complete!")
        print("Check the following directories:")
        print("  - ./batch_intermediate_output/ (contains subdirectories with processed images)")
        print("  - ./batch_final_output/ (contains all extracted text files)")
        
    except Exception as e:
        print(f"Error during batch processing: {e}")

def example_without_final_output_dir():
    """Example showing the original behavior (all outputs in same directory)."""
    print("\n=== Original Behavior Example ===")
    
    # Example PDF path
    pdf_path = "example.pdf"  # Change this to your PDF file path
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Process without final_output_dir (original behavior)
    print(f"Processing: {pdf_path}")
    print("All outputs (images and text) will be saved to: ./combined_output")
    
    try:
        paragraphs = processor.process_pdf(
            pdf_path=pdf_path,
            output_dir="./combined_output",  # Both images and text go here
            # final_output_dir not specified - uses output_dir for everything
            exclude_bottom_percent=0.07,
            exclude_top_percent=0.07
        )
        
        print(f"\nSuccessfully extracted {len(paragraphs)} text blocks.")
        print("All outputs saved in: ./combined_output/")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    print("PDF Processing Examples with final_output_dir Parameter")
    print("=" * 60)
    
    # Run examples (comment out the ones you don't want to run)
    example_single_file_processing()
    example_batch_processing()
    example_without_final_output_dir()
    
    print("\n" + "=" * 60)
    print("Examples complete!")
    print("\nTo use these examples:")
    print("1. Update the file paths in the functions above")
    print("2. Run: python example_usage.py")
    print("\nOr use the command line interface:")
    print("  Single file: python main.py your_file.pdf --output_dir ./intermediate --final_output_dir ./final")
    print("  Batch folder: python main.py ./input_folder --output_dir ./intermediate --final_output_dir ./final") 