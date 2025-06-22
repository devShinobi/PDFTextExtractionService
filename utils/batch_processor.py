import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from .image_processor import ImageProcessor

def _process_pdf_worker(pdf_path: str, output_dir: str, final_output_dir: str, debug: bool, kwargs: dict):
    """
    A picklable, top-level function designed to be run in a separate process.
    This function handles the processing of a single PDF file.
    """
    file_start_time = time.time()
    pdf_file_name = os.path.basename(pdf_path)

    try:
        # Each worker process initializes its own ImageProcessor.
        # This is crucial for process safety and avoids issues with pickling complex objects.
        image_processor = ImageProcessor()
        
        print(f"Starting processing for: {pdf_file_name}")
        
        image_processor.process_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            final_output_dir=final_output_dir,
            debug=debug,
            **kwargs
        )
        
        file_end_time = time.time()
        return {
            'file': pdf_file_name,
            'status': 'success',
            'processing_time': file_end_time - file_start_time,
            'output_dir': output_dir
        }
    except Exception as e:
        file_end_time = time.time()
        # It's helpful to know which file failed and why.
        print(f"--- ERROR processing {pdf_file_name}. Error: {e} ---")
        return {
            'file': pdf_file_name,
            'status': 'failed',
            'processing_time': file_end_time - file_start_time,
            'error': str(e)
        }

class BatchProcessor:
    """
    Processes all PDF files in a given directory.
    """
    def __init__(self):
        """
        Initializes the BatchProcessor.
        Note: The ImageProcessor instance here is used for non-parallel API calls,
        but each worker in parallel processing will create its own instance.
        """
        self.image_processor = ImageProcessor()

    def process_folder(self, input_folder: str, output_folder: str, final_output_dir: str = None, max_workers: int = 10, debug: bool = False, mode: str = "thread", **kwargs):
        """
        Scans a folder for PDF files and processes each one in parallel.

        Args:
            input_folder (str): The path to the folder containing PDF files.
            output_folder (str): The root directory where output sub-directories will be created.
            final_output_dir (str, optional): The directory where final text outputs will be saved.
            max_workers (int): Maximum number of parallel workers/processes (default: 10).
            debug (bool): If True, saves debug images with bounding boxes (default: False).
            mode (str): The parallel execution mode. 'thread' (default) is often faster for I/O-heavy
                        tasks, while 'process' is better for CPU-bound tasks.
            **kwargs: Additional arguments to pass to the process_pdf method.
        """
        executor_class = ProcessPoolExecutor if mode == "process" else ThreadPoolExecutor
        print(f"Starting batch processing for folder: {input_folder}")
        print(f"Using up to {max_workers} parallel {mode}s")

        if final_output_dir:
            os.makedirs(final_output_dir, exist_ok=True)
            print(f"Final text outputs will be saved to: {final_output_dir}")

        pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

        if not pdf_files:
            print(f"No PDF files found in '{input_folder}'.")
            return

        print(f"Found {len(pdf_files)} PDF files to process.")

        start_time = time.time()
        successful_files, failed_files = [], []

        with executor_class(max_workers=max_workers) as executor:
            future_to_file = {}
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_folder, pdf_file)
                pdf_name = os.path.splitext(pdf_file)[0]
                pdf_output_dir = os.path.join(output_folder, pdf_name)
                os.makedirs(pdf_output_dir, exist_ok=True)

                if mode == "process":
                    future = executor.submit(_process_pdf_worker, pdf_path, pdf_output_dir, final_output_dir, debug, kwargs)
                else: # thread mode
                    future = executor.submit(self._thread_worker, pdf_path, pdf_output_dir, final_output_dir, debug, kwargs)
                future_to_file[future] = pdf_file

            print(f"Submitted {len(pdf_files)} files for processing...")

            for future in as_completed(future_to_file):
                result = future.result()
                if result['status'] == 'success':
                    successful_files.append(result)
                    print(f"✅ Successfully processed {result['file']} in {result['processing_time']:.2f}s.")
                else:
                    failed_files.append(result)
                    print(f"❌ FAILED to process {result['file']}. Error: {result['error']}")

        end_time = time.time()
        total_time = end_time - start_time

        self._print_summary(total_files=len(pdf_files), successful_files=successful_files, failed_files=failed_files, total_time=total_time)

        return {
            'total_files': len(pdf_files),
            'successful_files': successful_files,
            'failed_files': failed_files,
            'total_time': total_time,
            'success_rate': len(successful_files) / len(pdf_files) if pdf_files else 0
        }

    def _thread_worker(self, pdf_path, output_dir, final_output_dir, debug, kwargs):
        """Worker function for the ThreadPoolExecutor."""
        file_start_time = time.time()
        pdf_file_name = os.path.basename(pdf_path)
        try:
            self.image_processor.process_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                final_output_dir=final_output_dir,
                debug=debug,
                **kwargs
            )
            file_end_time = time.time()
            return {
                'file': pdf_file_name,
                'status': 'success',
                'processing_time': file_end_time - file_start_time,
                'output_dir': output_dir
            }
        except Exception as e:
            file_end_time = time.time()
            print(f"--- ERROR processing {pdf_file_name}. Error: {e} ---")
            return {
                'file': pdf_file_name,
                'status': 'failed',
                'processing_time': file_end_time - file_start_time,
                'error': str(e)
            }

    def _print_summary(self, total_files, successful_files, failed_files, total_time):
        """Prints a detailed summary of the batch processing results."""
        print(f"\n{'='*80}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        
        if successful_files:
            success_times = [f['processing_time'] for f in successful_files]
            avg_time = sum(success_times) / len(success_times)
            print(f"\n✅ SUCCESSFUL ({len(successful_files)}/{total_files}):")
            print(f"   Avg. Time per File: {avg_time:.2f}s | Fastest: {min(success_times):.2f}s | Slowest: {max(success_times):.2f}s")
        
        if failed_files:
            print(f"\n❌ FAILED ({len(failed_files)}/{total_files}):")
            for f in failed_files:
                print(f"   - {f['file']}: {f['error']}")
        
        print(f"\n📊 OVERALL STATS:")
        print(f"   Total Batch Time: {total_time:.2f} seconds")
        if successful_files and total_time > 0:
            total_cpu_time = sum(f['processing_time'] for f in successful_files)
            speedup = total_cpu_time / total_time
            print(f"   Total CPU Time (successful files): {total_cpu_time:.2f} seconds")
            print(f"   Parallel Speedup: {speedup:.2f}x")
        
        print(f"{'='*80}") 