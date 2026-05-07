"""Compute TIDMAD Benchmark 1 (Denoising Score) from pre-chunked HDF5 files.

The data is already in one-second chunks with pre-calculated peak frequencies.
Parallel processing via concurrent.futures; with 8 cores takes ~5-10 minutes.
"""
import numpy as np
import h5py as h5
import logging
import argparse
import os
import gc
from tqdm import tqdm
import concurrent.futures
import math
import csv
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)

def GetChunkPSD(file_path, file_name, channel, chunk_idx):
    """
    Calculate PSD for a single pre-chunked time series.
    
    Args:
        file_path: Path to the HDF5 file
        file_name: Name of the HDF5 file
        channel: 1 or 2 (corresponds to time_series_ch1 or time_series_ch2)
        chunk_idx: Index of the chunk to process
    
    Returns:
        freq_array: Frequency values for the PSD
        psd_chunk: Power spectral density values
        sample_rate: Sampling rate in Hz
    """
    file_full_path = os.path.join(file_path, file_name)
    N = 10000000  # Chunk length
    
    with h5.File(file_full_path, 'r') as h5f:
        if channel == 1:
            data = h5f['time_series_ch1'][chunk_idx, :]
        elif channel == 2:
            data = h5f['time_series_ch2'][chunk_idx, :]
        else:
            raise ValueError("Channel must be 1 or 2")
        
        sample_rate = h5f.attrs['sample_rate_hz']
        
        TS = np.array(data, dtype=np.float32)
        dt = 1.0 / sample_rate
        
        # Calculate PSD using FFT
        psd_chunk = dt / N * (abs(np.fft.rfft(TS)) ** 2)[1:]
        freq_array = np.linspace(0, 5e6, len(psd_chunk))
    
    del data, TS
    gc.collect()
    
    return freq_array, psd_chunk, sample_rate

def getSNR(freq, pwr, target_freq):
    """
    Calculate SNR at a target frequency.
    
    Args:
        freq: Frequency array
        pwr: Power array
        target_freq: Target frequency (already calculated, not searched)
    
    Returns:
        snr: Signal-to-noise ratio
    """
    # Try exact match first (original behavior)
    try:
        center_id = int(np.where(freq == target_freq)[0][0])
    except IndexError:
        # Fall back to closest match if exact match not found
        center_id = int(np.argmin(np.abs(freq - target_freq)))
    
    sig_range = 1
    noise_range = 50
    signal = np.sum(pwr[center_id-sig_range:center_id+sig_range+1])
    noise = np.sum(pwr[center_id-noise_range:center_id+noise_range+1]) - signal
    
    if noise <= 0:
        return 0.0
    
    return signal / noise

def process_chunk(chunk_idx, file_path, file_name, peak_freqs):
    """
    Process a single chunk for SNR calculation.
    
    Args:
        chunk_idx: Index of the chunk
        file_path: Path to files
        file_name: Name of HDF5 file
        peak_freqs: Array of pre-calculated peak frequencies
    
    Returns:
        chunk_idx, snr_ch2, snr_ch1
    """
    target_freq = peak_freqs[chunk_idx, 0]
    
    # Process channel 2 (SG)
    freq_ch2, psd_ch2, _ = GetChunkPSD(file_path, file_name, channel=2, chunk_idx=chunk_idx)
    snr_ch2 = getSNR(freq_ch2, psd_ch2, target_freq)
    
    # Process channel 1 (SQUID)
    freq_ch1, psd_ch1, _ = GetChunkPSD(file_path, file_name, channel=1, chunk_idx=chunk_idx)
    snr_ch1 = getSNR(freq_ch1, psd_ch1, target_freq)
    
    return chunk_idx, snr_ch2, snr_ch1

def calculateBenchmark(file_path, file_names, args):
    """
    Calculate denoising benchmark score.
    
    Args:
        file_path: Path to HDF5 files
        file_names: List of HDF5 file names to process
        args: Argument namespace with parallel, num_workers, and coarse
    
    Returns:
        score: Log-transformed denoising score
        all_snr_ch2: Array of channel 2 SNRs
        all_snr_ch1: Array of channel 1 SNRs
    """
    snr_ch2_all = []
    snr_ch1_all = []
    
    for file_name in file_names:
        logging.info(f"Processing file: {file_name}")
        file_full_path = os.path.join(file_path, file_name)
        
        # Get number of chunks and peak frequencies
        with h5.File(file_full_path, 'r') as h5f:
            n_chunks = h5f.attrs['n_chunks']
            peak_freqs = h5f['signal_frequency'][:]
        
        # For coarse scan, only process every 10th chunk
        chunk_indices = list(range(0, n_chunks, 10)) if args.coarse else list(range(n_chunks))
        n_chunks_to_process = len(chunk_indices)
        
        snr_ch2 = np.zeros(n_chunks_to_process)
        snr_ch1 = np.zeros(n_chunks_to_process)
        
        if args.parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                tasks = [
                    executor.submit(process_chunk, chunk_idx, file_path, file_name, peak_freqs)
                    for chunk_idx in chunk_indices
                ]
                
                for future_idx, future in enumerate(tqdm(concurrent.futures.as_completed(tasks), total=n_chunks_to_process, desc=file_name)):
                    idx, result_snr_ch2, result_snr_ch1 = future.result()
                    snr_ch2[future_idx] = result_snr_ch2
                    snr_ch1[future_idx] = result_snr_ch1
        else:
            for process_idx, chunk_idx in enumerate(tqdm(chunk_indices, desc=file_name)):
                idx, result_snr_ch2, result_snr_ch1 = process_chunk(chunk_idx, file_path, file_name, peak_freqs)
                snr_ch2[process_idx] = result_snr_ch2
                snr_ch1[process_idx] = result_snr_ch1
        
        snr_ch2_all.extend(snr_ch2)
        snr_ch1_all.extend(snr_ch1)
    
    snr_ch2_all = np.array(snr_ch2_all)
    snr_ch1_all = np.array(snr_ch1_all)
    
    # Normalize channel 2
    snr_ch2_all = snr_ch2_all / np.amax(snr_ch2_all)
    
    # Calculate score
    score = np.round(np.sum(np.multiply(snr_ch2_all, snr_ch1_all)) / snr_ch1_all.size, decimals=2) + 1e-10
    log_score = math.log(score, 5.27)
    
    return log_score, snr_ch2_all, snr_ch1_all

def save_results(score, snr_ch2, snr_ch1, output_path, model_name):
    """
    Save benchmark results to CSV file.
    
    Args:
        score: The benchmark score
        snr_ch2: Array of channel 2 SNRs
        snr_ch1: Array of channel 1 SNRs
        output_path: Path to save the CSV
        model_name: Name of the denoising model
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with metadata
        writer.writerow(['TIDMAD Benchmark 1: Denoising Score'])
        writer.writerow(['Timestamp', timestamp])
        writer.writerow(['Model', model_name])
        writer.writerow(['Score', score])
        writer.writerow([])
        
        # Write per-chunk data
        writer.writerow(['Chunk Index', 'SNR Channel 2', 'SNR Channel 1'])
        for i, (ch2, ch1) in enumerate(zip(snr_ch2, snr_ch1)):
            writer.writerow([i, ch2, ch1])
    
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Benchmark 1: Denoising Score from pre-chunked HDF5 files"
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default=os.getcwd(),
        help='Directory containing HDF5 files (default: current working directory)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=os.getcwd(),
        help='Directory to save results CSV (default: current working directory)'
    )
    parser.add_argument(
        '--files', '-f',
        type=str,
        nargs='+',
        help='Specific HDF5 file(s) to process. If not specified, will auto-detect all .h5 files'
    )
    parser.add_argument(
        '--coarse', '-c',
        action='store_true',
        help='Run coarse scan: calculate SNR for every 10th chunk (faster approximation)'
    )
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Enable parallel processing with multiprocessing'
    )
    parser.add_argument(
        '--num_workers', '-w',
        type=int,
        default=8,
        help='Number of worker processes for parallel processing (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Get list of files to process
    if args.files:
        file_list = args.files
    else:
        file_list = [f for f in os.listdir(args.data_dir) if f.endswith('.h5')]
        file_list.sort()
    
    if not file_list:
        logging.error("No HDF5 files found in the specified directory")
        exit(1)
    
    logging.info(f"Processing files: {file_list}")
    
    score, snr_ch2, snr_ch1 = calculateBenchmark(args.data_dir, file_list, args)
    
    scan_type = "Coarse" if args.coarse else "Fine"
    print(f"\n{'='*50}")
    print(f"{scan_type} Denoising Score: {score}")
    print(f"{'='*50}\n")
    
    # Save results
    output_file = os.path.join(args.output_dir, 'benchmark_results.csv')
    model_name = os.path.commonprefix(file_list) if file_list else 'unknown'
    save_results(score, snr_ch2, snr_ch1, output_file, model_name)
