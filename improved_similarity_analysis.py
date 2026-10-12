#!/usr/bin/env python3
"""
Improved Similarity Analysis for Verilog Designs

This script compares LLM-generated Verilog designs with their reference implementations
using SimEval. It first checks summary files to identify successful samples,
then calculates similarity scores and outputs results to CSV files.

Usage:
  python improved_similarity_analysis.py [--build-dir PATH] [--methods METHOD1,METHOD2,...] [--output-dir PATH]
"""

import os
import sys
import glob
import csv
import subprocess
import re
import logging
import time
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.FileHandler("similarity_analysis.log"),
                             logging.StreamHandler()])

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Directory containing SimEval scripts
SIMEVAL_DIR = os.path.join(BASE_DIR, "simeval")
# Directory containing reference designs (GATE_BREAKER_REF_DIR overrides the default)
REF_DIR = os.environ.get("GATE_BREAKER_REF_DIR", os.path.join(BASE_DIR, "dataset_netlist-to-rtl"))

def find_build_directories():
    """Find all build directories in the project."""
    build_dirs = []
    # Look for directories matching pattern build_* or loc_*
    for item in os.listdir(BASE_DIR):
        if (item.startswith("build_") or item.startswith("loc_")) and os.path.isdir(os.path.join(BASE_DIR, item)):
            build_dirs.append(os.path.join(BASE_DIR, item))
    
    if not build_dirs:
        logging.warning("No build directories found. Looking for directories starting with 'build_' or 'loc_'")
    
    return build_dirs

def extract_model_info(build_dir):
    """Extract model and configuration information from build directory name."""
    dir_name = os.path.basename(build_dir)
    parts = dir_name.split('_')
    
    # Handle both build_* and loc_* patterns
    if dir_name.startswith("build_"):
        model_info = {
            'task': parts[1] if len(parts) > 1 else "unknown",
            'model': parts[2] if len(parts) > 2 else "unknown",
            'shots': parts[3].replace('shots', '') if len(parts) > 3 else "unknown",
            'samples': parts[4].replace('n', '') if len(parts) > 4 else "unknown"
        }
    else:  # loc_* pattern
        model_info = {
            'task': parts[0] if len(parts) > 0 else "unknown",
            'model': '_'.join(parts[1:]) if len(parts) > 1 else "unknown",
            'shots': "unknown",
            'samples': "unknown"
        }
    
    return model_info

def find_problem_designs(build_dir):
    """Find all problem designs in a build directory."""
    problem_dirs = []
    for item in os.listdir(build_dir):
        item_path = os.path.join(build_dir, item)
        if item.startswith("Prob") and os.path.isdir(item_path):
            problem_dirs.append(item_path)
    
    return problem_dirs

def get_successful_samples(problem_dir):
    """Get all successful sample files for a particular problem by checking test logs.
    Uses strict matching to ensure only truly correct designs are selected."""
    successful_samples = []
    sample_count = 0
    
    for log_file in glob.glob(os.path.join(problem_dir, "*-sv-iv-test.log")):
        sample_file = log_file.replace("-sv-iv-test.log", ".sv")
        sample_count += 1
        
        # Check if the sample file exists
        if not os.path.exists(sample_file):
            continue
            
        # Check if the test was successful
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
                # Check for definitive successful test indicators
                success = False
                
                # Specific patterns that indicate a successful test
                if "Mismatches: 0 in" in content or "Mismatches: 0" in content:
                    # Check there are no errors
                    if "error" not in content.lower() and "exception" not in content.lower():
                        success = True
                
                # Additional check for VCD file generation 
                # (indicates simulation ran to completion)
                vcd_check = "dumpfile wave.vcd opened for output" in content
                
                # If we've passed all checks, consider this a successful sample
                if success and vcd_check:
                    sample_name = os.path.basename(sample_file)
                    logging.debug(f"Found successful sample: {sample_name}")
                    successful_samples.append(sample_file)
        except Exception as e:
            logging.warning(f"Error reading log file {log_file}: {e}")
    
    if successful_samples:
        logging.info(f"Found {len(successful_samples)}/{sample_count} successful samples in {os.path.basename(problem_dir)}")
    else:
        logging.warning(f"No successful samples found in {os.path.basename(problem_dir)} out of {sample_count} total samples")
    
    return successful_samples

def get_ref_file(problem_name):
    """Get the reference file for a problem, preprocessing if needed to ensure parser compatibility."""
    ref_file = os.path.join(REF_DIR, f"{problem_name}_ref.sv")
    if not os.path.exists(ref_file):
        logging.warning(f"Reference file not found: {ref_file}")
        return None
    
    # Create a temporary file with preprocessed content
    try:
        with open(ref_file, 'r') as f:
            content = f.read()
        
        # Preprocess problematic syntax
        modified = False
        
        # Replace 'x with a compatible syntax
        if "'x" in content:
            content = content.replace("'x", "2'bx")  # Replace with explicit width
            modified = True
        
        # Replace other problematic SystemVerilog syntax as needed
        
        if modified:
            # Create temp file with modified content
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"temp_{os.path.basename(ref_file)}")
            with open(temp_file, 'w') as f:
                f.write(content)
            logging.info(f"Created preprocessed reference file for {problem_name} due to syntax compatibility issues")
            return temp_file
        
        return ref_file
    except Exception as e:
        logging.error(f"Error preprocessing reference file {ref_file}: {e}")
        return ref_file

def run_similarity_analysis(sample_file, ref_file, method, timeout=60):
    """Run a similarity analysis on a pair of files using a specific method.
    
    Args:
        sample_file: Path to the sample file
        ref_file: Path to the reference file
        method: Similarity method to use (ast, cfg, netlist)
        timeout: Subprocess timeout in seconds
        
    Returns:
        Float: Similarity score if successful, None otherwise
    """
    sample_name = os.path.basename(sample_file)
    ref_name = os.path.basename(ref_file)
    
    # Validate input files exist
    if not os.path.exists(sample_file):
        logging.error(f"Sample file does not exist: {sample_file}")
        return None
    
    if not os.path.exists(ref_file):
        logging.error(f"Reference file does not exist: {ref_file}")
        return None
    
    # Create a unique identifier for this analysis run
    analysis_id = f"{method}_{sample_name}"
    
    try:
        cmd = [
            "python", 
            os.path.join(SIMEVAL_DIR, "main.py"),
            sample_file,
            ref_file,
            "--method", method,
            "--log_level", "INFO"  # Use INFO level for more detailed output
        ]
        
        # Log the command being executed for troubleshooting
        logging.debug(f"[{analysis_id}] Running command: {' '.join(cmd)}")
        
        # Run SimEval with timeout
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        # Log success even if we can't extract a score
        logging.debug(f"[{analysis_id}] SimEval process completed in {elapsed_time:.2f}s")
        
        # Extract the similarity score from the output
        stdout = result.stdout.strip()
        
        # Primary pattern: "Similarity Score (METHOD): VALUE"
        match = re.search(r"Similarity Score \((\w+)\): (\d+\.\d+)", stdout)
        if match:
            score = float(match.group(2))
            logging.info(f"Successfully extracted {method.upper()} score for {sample_name}: {score}")
            return score
        
        # Fallback pattern: "Similarity: VALUE"
        match = re.search(r"[Ss]imilarity[:\s]+(\d+\.\d+)", stdout)
        if match:
            score = float(match.group(1))
            logging.info(f"Extracted {method.upper()} score using fallback pattern for {sample_name}: {score}")
            return score
        
        # If we got here, no similarity score was found
        # Check for specific error patterns in the output
        error_patterns = {
            "Error parsing": "Parsing error",
            "Exception": "Exception occurred",
            "cannot open": "File access error",
            "Illegal character": "Syntax error",
            "Error: One or both Verilog files could not be parsed": "Parsing failure"
        }
        
        # Extract and log any error messages
        error_found = False
        error_message = ""
        
        for pattern, error_type in error_patterns.items():
            if pattern in stdout:
                # Find the actual error message surrounding the pattern
                pattern_index = stdout.find(pattern)
                start_index = max(0, pattern_index - 50)
                end_index = min(len(stdout), pattern_index + 200)
                
                # Extract context around the error
                error_context = stdout[start_index:end_index]
                
                # Add to error message
                if error_message:
                    error_message += "\n"
                error_message += f"{error_type}: {error_context.strip()}"
                error_found = True
        
        if error_found:
            logging.warning(f"[{analysis_id}] SimEval error: {error_message}")
        else:
            # If stdout is empty or very short, check stderr
            if not stdout or len(stdout) < 20:
                logging.warning(f"[{analysis_id}] No meaningful output from SimEval. Empty or minimal response.")
            else:
                # Log a shortened version of the output
                max_lines = 5
                lines = stdout.strip().split('\n')
                shortened = '\n'.join(lines[:max_lines])
                if len(lines) > max_lines:
                    shortened += f"\n... ({len(lines) - max_lines} more lines)"
                    
                logging.warning(f"[{analysis_id}] Could not extract similarity score. SimEval output:\n{shortened}")
                
            # Log full output at debug level
            logging.debug(f"[{analysis_id}] Full SimEval stdout:\n{stdout}")
        
        return None
        
    except subprocess.TimeoutExpired as e:
        logging.warning(f"[{analysis_id}] Timeout after {timeout}s: {e}")
        return None
    except subprocess.CalledProcessError as e:
        # Log detailed error information
        logging.warning(f"[{analysis_id}] SimEval process error (code {e.returncode})")
        
        # Extract meaningful error message from stdout and stderr
        error_message = ""
        
        if e.stdout:
            stdout = e.stdout.strip()
            if stdout:
                # Try to find specific error patterns
                for pattern in ["Error", "Exception", "failed", "cannot", "Illegal"]:
                    pattern_lower = pattern.lower()
                    stdout_lower = stdout.lower()
                    if pattern_lower in stdout_lower:
                        # Find line containing error
                        lines = stdout.split('\n')
                        for line in lines:
                            if pattern_lower in line.lower():
                                error_message += f"STDOUT: {line.strip()}\n"
                                break
                
                # If no specific error found, include beginning of stdout
                if not error_message and len(stdout) > 0:
                    lines = stdout.split('\n')
                    error_message += f"STDOUT: {lines[0]}"
                    if len(lines) > 1:
                        error_message += f" ... ({len(lines)} lines total)\n"
                    else:
                        error_message += "\n"
        
        if e.stderr:
            stderr = e.stderr.strip()
            if stderr:
                lines = stderr.split('\n')
                error_message += f"STDERR: {lines[0]}"
                if len(lines) > 1:
                    error_message += f" ... ({len(lines)} lines total)"
        
        if error_message:
            logging.warning(f"[{analysis_id}] Error details: {error_message}")
        else:
            logging.warning(f"[{analysis_id}] No detailed error information available")
            
        # Log full output at debug level
        if e.stdout:
            logging.debug(f"[{analysis_id}] Full STDOUT:\n{e.stdout}")
        if e.stderr:
            logging.debug(f"[{analysis_id}] Full STDERR:\n{e.stderr}")
            
        return None
    except Exception as e:
        logging.warning(f"[{analysis_id}] Unexpected error: {type(e).__name__}: {e}")
        return None

def analyze_single_sample(sample_file, ref_file, methods, timeout=60):
    """Analyze a single sample file against its reference using specified methods.
    
    Args:
        sample_file: Path to the sample file
        ref_file: Path to the reference file
        methods: List of similarity methods to use
        timeout: Timeout in seconds for each analysis run
        
    Returns:
        Dict mapping methods to similarity scores
    """
    results = {}
    sample_name = os.path.basename(sample_file)
    
    for method in methods:
        logging.debug(f"Running {method} similarity analysis for {sample_name}")
        score = run_similarity_analysis(sample_file, ref_file, method, timeout=timeout)
        results[method] = score
    
    # Log summary of results
    successful_methods = [m for m, s in results.items() if s is not None]
    failed_methods = [m for m, s in results.items() if s is None]
    
    if successful_methods:
        logging.info(f"Successful similarity calculations for {sample_name}: {', '.join(successful_methods)}")
        if failed_methods:
            logging.warning(f"Failed similarity methods for {sample_name}: {', '.join(failed_methods)}")
    else:
        logging.warning(f"All similarity methods failed for {sample_name}: {', '.join(methods)}")
    
    # Log scores for successful methods
    for method, score in results.items():
        if score is not None:
            logging.info(f"{sample_name} {method.upper()} similarity: {score:.6f}")
    
    return results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run similarity analysis on Verilog designs.")
    parser.add_argument("--build-dir", 
                        help="Path to a specific build directory to analyze. If not provided, will discover all build directories.")
    parser.add_argument("--methods", default="ast,cfg,netlist",
                        help="Comma-separated list of similarity methods to run (default: ast,cfg,netlist)")
    parser.add_argument("--output-dir", 
                        help="Path to output directory for results. Defaults to 'similarity_results' in project root.")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Number of parallel processes to use (default: 4)")
    parser.add_argument("--timeout", type=int, default=100,
                        help="Timeout in seconds for each similarity analysis (default: 60)")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Skip to next sample if similarity calculation fails")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Determine methods to run
    methods = [m.strip() for m in args.methods.split(",")]
    valid_methods = ["ast", "cfg", "netlist"]
    methods = [m for m in methods if m in valid_methods]
    if not methods:
        logging.error(f"No valid methods specified. Please choose from: {', '.join(valid_methods)}")
        sys.exit(1)
    logging.info(f"Running the following similarity methods: {', '.join(methods)}")
    
    # Determine build directories to process
    if args.build_dir:
        if os.path.isdir(args.build_dir):
            build_dirs = [args.build_dir]
            logging.info(f"Processing specified build directory: {args.build_dir}")
        else:
            logging.error(f"Specified build directory does not exist: {args.build_dir}")
            sys.exit(1)
    else:
        # Find all build directories
        build_dirs = find_build_directories()
        if not build_dirs:
            sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(BASE_DIR, "similarity_results")
    
    # Create output directory for results if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each build directory
    for build_dir in build_dirs:
        model_info = extract_model_info(build_dir)
        logging.info(f"Processing build directory: {os.path.basename(build_dir)}")
        logging.info(f"Model: {model_info['model']}, Shots: {model_info['shots']}, Samples: {model_info['samples']}")
        
        # Create output file for this model
        output_file = os.path.join(output_dir, f"similarity_{os.path.basename(build_dir)}.csv")
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Create header row with dynamic method columns
            header = ['Problem', 'Sample', 'Model', 'Shots']
            for method in methods:
                header.append(f"{method.upper()}_Similarity")
            
            writer.writerow(header)
            
            # Find all problem directories
            problem_dirs = find_problem_designs(build_dir)
            logging.info(f"Found {len(problem_dirs)} problem directories")
            
            # Process each problem
            all_results = []
            for problem_dir in tqdm(problem_dirs, desc="Processing problems"):
                problem_name = os.path.basename(problem_dir)
                ref_file = get_ref_file(problem_name)
                
                if not ref_file:
                    continue
                
                # Process only successful samples for this problem
                sample_files = get_successful_samples(problem_dir)
                
                if not sample_files:
                    logging.warning(f"No successful samples found for {problem_name}, skipping")
                    continue
                
                logging.info(f"Found {len(sample_files)} sample(s) for {problem_name}")
                
                # Use process pool for parallel execution
                with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                    future_to_sample = {
                        executor.submit(analyze_single_sample, sample_file, ref_file, methods, args.timeout): sample_file
                        for sample_file in sample_files
                    }
                    
                    for future in as_completed(future_to_sample):
                        sample_file = future_to_sample[future]
                        sample_name = os.path.basename(sample_file)
                        try:
                            results = future.result()
                            
                            # Check if we got any valid results
                            if all(results.get(method) is None for method in methods):
                                logging.warning(f"All similarity methods failed for {sample_name}")
                                continue
                            
                            # Write to CSV - dynamic columns based on methods used
                            row = [
                                problem_name,
                                sample_name,
                                model_info['model'],
                                model_info['shots']
                            ]
                            
                            # Add results for each method in order
                            for method in methods:
                                row.append(results.get(method))
                                
                            writer.writerow(row)
                            
                            # Collect all results for summary statistics
                            sample_result = {
                                'problem': problem_name,
                                'sample': sample_name
                            }
                            for method in methods:
                                sample_result[method] = results.get(method)
                            
                            all_results.append(sample_result)
                            
                        except Exception as e:
                            logging.error(f"Error processing {sample_file}: {e}")
            
            # Calculate summary statistics
            if all_results:
                # Write summary to CSV
                writer.writerow([])
                writer.writerow(['SUMMARY'])
                writer.writerow(['Metric', 'Average', 'Min', 'Max', 'Count', 'Success Rate'])
                
                # Calculate stats for each method
                for method in methods:
                    # Count total attempts and successful calculations
                    total_attempts = len(all_results)
                    scores = [r[method] for r in all_results if r.get(method) is not None]
                    success_count = len(scores)
                    success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
                    
                    if scores:
                        writer.writerow([
                            method.upper(), 
                            sum(scores) / len(scores),
                            min(scores), 
                            max(scores), 
                            success_count,
                            f"{success_rate:.1f}%"
                        ])
                    else:
                        writer.writerow([
                            method.upper(),
                            "N/A",
                            "N/A",
                            "N/A",
                            0,
                            "0.0%"
                        ])
        
        logging.info(f"Results saved to {output_file}")

def summarize_results(output_files):
    """Summarize results from multiple output files."""
    logging.info("Summarizing results from all output files...")
    
    # Collect statistics for problems and methods
    problem_stats = defaultdict(lambda: {"total": 0, "success": 0})
    method_stats = defaultdict(lambda: {"total": 0, "success": 0})
    
    for output_file in output_files:
        try:
            with open(output_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                # Find method columns
                method_columns = {}
                for i, header in enumerate(headers):
                    if "_Similarity" in header:
                        method_name = header.replace("_Similarity", "").lower()
                        method_columns[method_name] = i
                
                # Process data rows
                for row in reader:
                    if not row or len(row) < len(headers) or row[0] == "SUMMARY":
                        continue
                    
                    problem = row[0]
                    
                    # Skip summary rows
                    if problem.startswith("AVERAGE") or problem == "OVERALL AVERAGE" or not problem:
                        continue
                    
                    # Process each method
                    for method, col_idx in method_columns.items():
                        if col_idx < len(row):
                            problem_stats[problem]["total"] += 1
                            method_stats[method]["total"] += 1
                            
                            # Check if we have a valid similarity score
                            value = row[col_idx]
                            if value and value != "None" and value != "N/A":
                                problem_stats[problem]["success"] += 1
                                method_stats[method]["success"] += 1
        except Exception as e:
            logging.error(f"Error processing {output_file}: {e}")
    
    # Print summary
    logging.info("\nSummary Statistics:")
    
    logging.info("\nMethod Success Rates:")
    for method, stats in sorted(method_stats.items()):
        if stats["total"] == 0:
            rate = 0
        else:
            rate = stats["success"] / stats["total"] * 100
        logging.info(f"{method.upper()}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    logging.info("\nProblem Success Rates:")
    # Sort problems by success rate (descending)
    sorted_problems = sorted(
        problem_stats.items(),
        key=lambda x: (x[1]["success"] / x[1]["total"] if x[1]["total"] > 0 else 0),
        reverse=True
    )
    
    for problem, stats in sorted_problems:
        if stats["total"] == 0:
            rate = 0
        else:
            rate = stats["success"] / stats["total"] * 100
        logging.info(f"{problem}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    # List problems with no successful calculations
    zero_success = [p for p, s in problem_stats.items() 
                   if s["success"] == 0 and s["total"] > 0]
    if zero_success:
        logging.warning("\nProblems with NO successful similarity calculations:")
        for problem in sorted(zero_success):
            logging.warning(f"- {problem} (0/{problem_stats[problem]['total']})")

if __name__ == "__main__":
    # Configure file handler for DEBUG level while keeping console at INFO
    file_handler = logging.FileHandler("similarity_analysis_detailed.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Track start time
    start_time = time.time()
    
    # Run main function
    output_files = []
    try:
        args = parse_args()
        # Run main processing
        main()
        
        # Find output files
        if args.output_dir:
            output_files = glob.glob(os.path.join(args.output_dir, "similarity_*.csv"))
    except Exception as e:
        logging.error(f"Error during execution: {e}")
    
    # Print overall summary
    if output_files:
        summarize_results(output_files)
    
    # Print execution time
    elapsed_time = time.time() - start_time
    logging.info(f"\nTotal execution time: {elapsed_time:.2f} seconds")