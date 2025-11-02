#!/usr/bin/env python3
"""
End-to-end BPSK simulation with RRC pulse shaping, matched filtering,
and timing offset analysis.
"""
import argparse
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from .rrc_utils import rrc_taps
from .plots import plot_eye, plot_ber_curve, save_spectrum

def awgn(x, ebn0_db, sps=1):
    """
    Add AWGN to signal x for given Eb/N0 (dB).
    
    Parameters:
        x (np.ndarray): Input signal (after pulse shaping)
        ebn0_db (float): Eb/N0 in dB
        sps (int): Samples per symbol
    
    Returns:
        np.ndarray: Noisy signal
    
    Note: For BPSK, Eb = Es (energy per symbol) = signal power × symbol period.
    For an oversampled signal, Es = P_avg × sps where P_avg is average power per sample.
    """
    ebn0_linear = 10 ** (ebn0_db / 10.0)
    # Average signal power per sample
    signal_power = np.mean(x ** 2)
    # Energy per bit = signal power × samples per symbol (for BPSK, 1 bit/symbol)
    eb = signal_power * sps
    # Noise power per sample: N0/2 = Eb / (2 * Eb/N0)
    noise_power_per_sample = eb / (2.0 * ebn0_linear * sps)
    sigma = np.sqrt(noise_power_per_sample)
    return x + np.random.normal(0, sigma, size=x.shape)

def run_once(n_bits, sps, beta, span, ebn0_db, offset=0):
    """
    Run single BPSK trial with optional timing offset.
    
    Parameters:
        n_bits (int): Number of bits to simulate (output)
        sps (int): Samples per symbol
        beta (float): RRC roll-off factor
        span (int): RRC filter span in symbols
        ebn0_db (float): Eb/N0 in dB
        offset (int): Timing offset in samples
    
    Returns:
        dict: Simulation results including BER, waveforms, etc.
    """
    # Generate extra bits to compensate for filter delay
    # We need span extra symbols on each side for the RRC filter
    n_bits_with_margin = n_bits + 2 * span
    bits_extended = np.random.randint(0, 2, n_bits_with_margin) * 2 - 1  # ±1 BPSK symbols
    taps = rrc_taps(beta=beta, sps=sps, span=span)

    # Pulse shaping
    tx = np.repeat(bits_extended, sps)
    tx_shaped = lfilter(taps, 1.0, tx)
    
    # Channel (AWGN)
    rx = awgn(tx_shaped, ebn0_db, sps=sps)

    # Matched filtering
    rx_f = lfilter(taps, 1.0, rx)

    # Sampling (with optional offset)
    # Group delay from cascaded TX and RX filters
    # Each RRC filter has group delay of (length-1)/2
    # Total delay for matched filter pair = length - 1
    group_delay_samples = len(taps) - 1
    
    # The first symbol's optimal sampling point is at group_delay_samples
    # With timing offset, we sample at group_delay_samples + offset
    sampling_start = group_delay_samples + offset
    
    if sampling_start < 0:
        raise ValueError(f"Negative offset {offset} is too large")
    
    # Sample every sps samples starting from sampling_start
    sampled_indices = np.arange(sampling_start, len(rx_f), sps)
    sampled_all = rx_f[sampled_indices]
    
    # We generated n_bits_with_margin symbols, but want to evaluate only the middle n_bits
    # to avoid edge effects
    start_symbol_idx = span
    end_symbol_idx = start_symbol_idx + n_bits
    
    if end_symbol_idx > len(sampled_all):
        raise ValueError(f"Not enough samples after filtering: got {len(sampled_all)}, "
                        f"need at least {end_symbol_idx}")
    
    sampled = sampled_all[start_symbol_idx:end_symbol_idx]
    bits = bits_extended[start_symbol_idx:end_symbol_idx]
    
    # Decision and BER
    decisions = np.sign(sampled)
    # Handle zero crossings
    decisions[decisions == 0] = 1
    ber = np.mean(decisions != bits)

    return {
        "bits": bits,
        "tx_shaped": tx_shaped,
        "rx_f": rx_f,
        "sampled": sampled,
        "decisions": decisions,
        "ber": ber,
        "taps": taps,
    }

def run_sweep(snr_list, n_bits, sps, beta, span, outdir, offset=0):
    """
    Run BER sweep over SNR values.
    
    Parameters:
        snr_list (list): List of SNR values in dB
        n_bits (int): Number of bits per trial
        sps (int): Samples per symbol
        beta (float): RRC roll-off factor
        span (int): RRC filter span
        outdir (str): Output directory for results
        offset (int): Timing offset in samples
    
    Returns:
        dict: Results from the last simulation run
    """
    os.makedirs(outdir, exist_ok=True)
    if not snr_list:
        raise ValueError("snr_list must contain at least one SNR value")
    
    results = []
    res = None
    
    for snr in snr_list:
        res = run_once(n_bits, sps, beta, span, snr, offset=offset)
        results.append((snr, res["ber"]))
    
    # Save CSV
    csv_path = os.path.join(outdir, "ber.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db", "ber"])
        writer.writerows(results)
    
    if res is None:
        raise RuntimeError("No simulation was run, this should not happen")
    
    return res  # Return last trial for plotting

def main():
    args = parse_args()
    snrs = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    print("🚀 Running Week 10 Matched Filter Simulation")
    print(f"Parameters: sps={args.sps}, beta={args.beta}, span={args.span}, n_bits={args.n_bits}")
    print(f"SNR range: {args.snr_min} to {args.snr_max} dB (step={args.snr_step})")

    # Clean (ideal timing) - sample at the center of the eye
    # Due to np.repeat creating blocks, optimal sampling is around sps//2 - 1
    ideal_offset = args.sps // 2 - 1
    print(f"→ Baseline (ideal timing at offset={ideal_offset} samples)...")
    res_clean = run_sweep(snrs, args.n_bits, args.sps, args.beta, args.span,
                          os.path.join(args.outdir, "clean"), offset=ideal_offset)
    plot_eye(res_clean["rx_f"], args.sps, "plots/eye_clean.png")

    # With timing offset - sample near the symbol transition (edge of eye)
    # This introduces ISI
    offset_samples = 0  # At the symbol boundary - worst case
    print(f"→ With timing offset ({offset_samples} samples - at symbol boundary)...")
    res_offset = run_sweep(snrs, args.n_bits, args.sps, args.beta, args.span,
                           os.path.join(args.outdir, "offset"), offset=offset_samples)
    plot_eye(res_offset["rx_f"], args.sps, "plots/eye_offset.png")

    # Load BER data
    def load_ber(path):
        with open(path) as f:
            lines = f.read().strip().split("\n")[1:]
            return [float(line.split(",")[1]) for line in lines]

    ber_clean = load_ber(os.path.join(args.outdir, "clean", "ber.csv"))
    ber_offset = load_ber(os.path.join(args.outdir, "offset", "ber.csv"))

    # Plot combined BER
    plot_ber_curve(snrs, ber_clean, ber_offset, offset_samples, "plots/ber_curve.png")

    # Spectrum (once is enough)
    save_spectrum(res_clean["taps"], args.sps, "plots/spectrum.png")

    print("Simulation complete. Outputs saved to plots/ and benchmarks/")

def parse_args():
    parser = argparse.ArgumentParser(description="BPSK RRC Matched Filter Simulation")
    parser.add_argument("--n_bits", type=int, default=20000)
    parser.add_argument("--sps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--span", type=int, default=6)
    parser.add_argument("--snr_min", type=int, default=0)
    parser.add_argument("--snr_max", type=int, default=10)
    parser.add_argument("--snr_step", type=int, default=2)
    parser.add_argument("--outdir", type=str, default="benchmarks")
    return parser.parse_args()

if __name__ == "__main__":
    main()