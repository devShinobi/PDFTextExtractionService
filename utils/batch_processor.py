import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .image_processor import ImageProcessor

class BatchProcessor:
    """
    Processes all PDF files in a given directory.
    """
    def __init__(self):
        self.image_processor = ImageProcessor()

    def process_folder(self, input_folder: str, output_folder: str, final_output_dir: str = None, max_workers: int = 10, debug: bool = False, **kwargs):
        """
        Scans a folder for PDF files and processes each one in parallel.

        Args:
            input_folder (str): The path to the folder containing PDF files.
            output_folder (str): The root directory where output sub-directories will be created.
            final_output_dir (str, optional): The directory where final text outputs will be saved.
                                            If None, text outputs will be saved in the same directory as intermediate outputs.
            max_workers (int): Maximum number of parallel workers (default: 10).
            debug (bool): If True, saves debug images with bounding boxes (default: False).
            **kwargs: Additional arguments to pass to the process_pdf method.
        """
        print(f"Starting batch processing for folder: {input_folder}")
        print(f"Using {max_workers} parallel workers")
        print(f"Debug mode: {'Enabled' if debug else 'Disabled'}")
        
        if final_output_dir:
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"Final text outputs will be saved to: {final_output_dir}")
        
        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in '{input_folder}'.")
            return

        print(f"Found {len(pdf_files)} PDF files to process.")
        
        # Initialize timing statistics
        start_time = time.time()
        successful_files = []
        failed_files = []
        
        def process_single_pdf(pdf_file):
            """Process a single PDF file and return results."""
            pdf_path = os.path.join(input_folder, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            pdf_output_dir = os.path.join(output_folder, pdf_name)
            
            file_start_time = time.time()
            
            try:
                os.makedirs(pdf_output_dir, exist_ok=True)
                
                print(f"\n{'='*60}")
                print(f"Processing: {pdf_path}")
                print(f"Intermediate outputs will be saved to: {pdf_output_dir}")
                if final_output_dir:
                    print(f"Final text output will be saved to: {final_output_dir}")
                print(f"{'='*60}\n")
                
                self.image_processor.process_pdf(
                    pdf_path=pdf_path,
                    output_dir=pdf_output_dir,
                    final_output_dir=final_output_dir,
                    debug=debug,
                    **kwargs
                )
                
                file_end_time = time.time()
                processing_time = file_end_time - file_start_time
                
                print(f"\nSuccessfully processed {pdf_file} in {processing_time:.2f} seconds.")
                
                return {
                    'file': pdf_file,
                    'status': 'success',
                    'processing_time': processing_time,
                    'output_dir': pdf_output_dir
                }
                
            except Exception as e:
                file_end_time = time.time()
                processing_time = file_end_time - file_start_time
                
                print(f"--- FAILED to process {pdf_file} after {processing_time:.2f} seconds. Error: {e} ---")
                
                return {
                    'file': pdf_file,
                    'status': 'failed',
                    'processing_time': processing_time,
                    'error': str(e)
                }
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_single_pdf, pdf_file): pdf_file for pdf_file in pdf_files}
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                result = future.result()
                if result['status'] == 'success':
                    successful_files.append(result)
                else:
                    failed_files.append(result)
        
        # Calculate and display timing statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        # Success statistics
        if successful_files:
            success_times = [f['processing_time'] for f in successful_files]
            avg_success_time = sum(success_times) / len(success_times)
            min_success_time = min(success_times)
            max_success_time = max(success_times)
            
            print(f"\n✅ SUCCESSFUL PROCESSING ({len(successful_files)}/{len(pdf_files)} files):")
            print(f"   Average processing time: {avg_success_time:.2f} seconds")
            print(f"   Fastest file: {min_success_time:.2f} seconds")
            print(f"   Slowest file: {max_success_time:.2f} seconds")
            print(f"   Total successful processing time: {sum(success_times):.2f} seconds")
        
        # Failure statistics
        if failed_files:
            print(f"\n❌ FAILED PROCESSING ({len(failed_files)}/{len(pdf_files)} files):")
            for failed_file in failed_files:
                print(f"   - {failed_file['file']}: {failed_file['error']} ({failed_file['processing_time']:.2f}s)")
        
        # Overall statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total batch time: {total_time:.2f} seconds")
        print(f"   Success rate: {len(successful_files)}/{len(pdf_files)} ({len(successful_files)/len(pdf_files)*100:.1f}%)")
        
        if successful_files:
            print(f"   Parallel efficiency: {sum([f['processing_time'] for f in successful_files])/total_time:.2f}x speedup")
        
        print(f"{'='*80}")
        
        return {
            'total_files': len(pdf_files),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_time': total_time,
            'success_rate': len(successful_files) / len(pdf_files) if pdf_files else 0
        } 