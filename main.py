import argparse
from utils.image_processor import ImageProcessor
from utils.batch_processor import BatchProcessor
import os
import time

def main():
    """Main function to run the PDF processing pipeline."""
    parser = argparse.ArgumentParser(description="Process a single PDF file or a folder of PDF files.")
    parser.add_argument("input_path", type=str, help="Path to the PDF file or a folder containing PDF files.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files.")
    parser.add_argument("--final_output_dir", type=str, help="Directory to save final text output files (separate from intermediate outputs).")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save bounding box images.")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of parallel workers for batch processing (default: 10).")
    parser.add_argument("--top_margin", type=float, default=0.07, help="Percentage of top margin to exclude (0.0 to 1.0).")
    parser.add_argument("--bottom_margin", type=float, default=0.07, help="Percentage of bottom margin to exclude (0.0 to 1.0).")
    parser.add_argument("--ignore_vertical", action="store_true", help="If set, ignore text that appears to be vertically oriented.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input path not found at '{args.input_path}'")
        return

    # Prepare keyword arguments for the processors
    processor_kwargs = {
        'exclude_top_percent': args.top_margin,
        'exclude_bottom_percent': args.bottom_margin,
        'ignore_vertical_text': args.ignore_vertical
    }
        
    if os.path.isfile(args.input_path):
        # Process a single PDF file
        print("Processing a single PDF file.")
        if args.debug:
            print("Debug mode enabled - bounding box images will be saved.")
        else:
            print("Debug mode disabled - only text extraction will be performed.")
        
        processor = ImageProcessor()
        start_time = time.time()
        
        result = processor.process_pdf(
            pdf_path=args.input_path,
            output_dir=args.output_dir,
            final_output_dir=args.final_output_dir,
            debug=args.debug,
            **processor_kwargs
        )
        
        end_time = time.time()
        print(f"\nSingle file processing completed in {end_time - start_time:.2f} seconds.")
        
    elif os.path.isdir(args.input_path):
        # Process a folder of PDF files
        print("Processing a folder of PDF files.")
        if args.debug:
            print("Debug mode enabled - bounding box images will be saved.")
        else:
            print("Debug mode disabled - only text extraction will be performed.")
        
        batch_processor = BatchProcessor()
        
        result = batch_processor.process_folder(
            input_folder=args.input_path,
            output_folder=args.output_dir,
            final_output_dir=args.final_output_dir,
            max_workers=args.max_workers,
            debug=args.debug,
            **processor_kwargs
        )
        
        # Display summary
        if result:
            print(f"\nBatch processing summary:")
            print(f"  Total files: {result['total_files']}")
            print(f"  Successful: {len(result['successful_files'])}")
            print(f"  Failed: {len(result['failed_files'])}")
            print(f"  Success rate: {result['success_rate']*100:.1f}%")
    else:
        print(f"Error: Input path '{args.input_path}' is not a valid file or directory.")
        return
    
    if args.final_output_dir:
        print(f"\nProcessing complete. Intermediate outputs saved in '{args.output_dir}' directory.")
        print(f"Final text outputs saved in '{args.final_output_dir}' directory.")
    else:
        print(f"\nProcessing complete. All outputs saved in '{args.output_dir}' directory.")

if __name__ == "__main__":
    main() 