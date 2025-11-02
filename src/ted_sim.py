#!/usr/bin/env python3
"""
BPSK simulation with Gardner Timing Error Detector (TED) for blind timing recovery.
Demonstrates closed-loop symbol timing synchronization.
"""
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from .rrc_utils import rrc_taps

def awgn(x, ebn0_db, sps=1):
    """
    Add AWGN to signal x for given Eb/N0 (dB).
    
    Parameters:
        x (np.ndarray): Input signal (after pulse shaping)
        ebn0_db (float): Eb/N0 in dB
        sps (int): Samples per symbol
    
    Returns:
        np.ndarray: Noisy signal
    """
    ebn0_linear = 10 ** (ebn0_db / 10.0)
    # Average signal power per sample
    signal_power = np.mean(x ** 2)
    # Energy per bit = signal power × samples per symbol
    eb = signal_power * sps
    # Noise power per sample
    noise_power_per_sample = eb / (2.0 * ebn0_linear * sps)
    sigma = np.sqrt(noise_power_per_sample)
    return x + np.random.normal(0, sigma, size=x.shape)

def gardner_error(y_prev, y_mid, y_next):
    """
    Compute Gardner timing error for a single symbol.
    
    Gardner TED (data-aided version): 
        e[n] = (y[nT] - y[(n-1)T]) * y[nT - T/2]
    
    Or decision-directed:
        e[n] = (d[nT] - d[(n-1)T]) * y[nT - T/2]
        where d[] are hard decisions
    
    Parameters:
        y_prev: sample at previous symbol time (n-1)T
        y_mid: sample at midpoint between (n-1)T and nT
        y_next: sample at current symbol time nT
    
    Returns:
        Timing error estimate
    
    Positive error means we're sampling too late (need to advance)
    Negative error means we're sampling too early (need to delay)
    """
    # Data-aided Gardner (using actual sample values)
    return (y_next - y_prev) * y_mid

def run_ted_simulation(n_bits=5000, sps=8, beta=0.35, span=6, ebn0_db=10, 
                       mu_initial=0.0, k_p=0.01, save_plots=True):
    """
    Run BPSK simulation with Gardner TED.
    
    Parameters:
        n_bits: number of bits to simulate
        sps: samples per symbol
        beta: RRC roll-off factor
        span: RRC filter span in symbols
        ebn0_db: Eb/N0 in dB
        mu_initial: initial fractional timing offset (in samples, relative to optimal)
        k_p: proportional gain for timing loop (typically 0.001 to 0.1)
        save_plots: whether to save diagnostic plots
        
    Returns:
        dict with BER, timing logs, decisions, etc.
    """
    # Generate extra bits to avoid edge effects
    n_bits_with_margin = n_bits + 2 * span
    bits_extended = np.random.randint(0, 2, n_bits_with_margin) * 2 - 1
    taps = rrc_taps(beta=beta, sps=sps, span=span)

    # Transmitter
    tx = np.repeat(bits_extended, sps)
    tx_shaped = lfilter(taps, 1.0, tx)

    # Channel
    rx = awgn(tx_shaped, ebn0_db, sps=sps)

    # Matched filter
    rx_f = lfilter(taps, 1.0, rx)

        # Timing recovery loop using Gardner TED
    # Start at the group delay (optimal timing) + initial offset
    group_delay = len(taps) - 1
    # Optimal timing for our signal generation is around sps//2 - 1
    optimal_offset = sps // 2 - 1
    
    # Initialize mu at optimal + initial_offset
    mu = float(group_delay + optimal_offset + mu_initial)
    mu_log = []
    error_log = []
    decisions_out = []
    symbols_out = []
    
    # For Gardner TED, we need samples at T/2 spacing
    half_symbol = sps // 2
    
    # State for tracking: we need the previous symbol sample and the midpoint
    prev_symbol_sample = None
    prev_mid_sample = None
    
    n_symbols_out = 0
    max_iterations = n_bits_with_margin * 2  # Safety limit
    
    for iteration in range(max_iterations):
        if n_symbols_out >= n_bits_with_margin:
            break
            
        # Current sampling index for symbol decision point
        idx_symbol = int(np.round(mu))
        
        if idx_symbol >= len(rx_f):
            break
        
        # Get symbol sample
        symbol_sample = rx_f[idx_symbol]
        
        # Make decision
        decision = np.sign(symbol_sample)
        if decision == 0:
            decision = 1
        decisions_out.append(decision)
        symbols_out.append(symbol_sample)
        n_symbols_out += 1
        
        # Compute Gardner error if we have enough history
        # We need: previous symbol, midpoint between prev and current, current symbol
        if prev_symbol_sample is not None and prev_mid_sample is not None:
            # Gardner: error = (symbol_sample - prev_symbol_sample) * prev_mid_sample
            # prev_mid_sample is the sample halfway between prev_symbol and current symbol
            error = gardner_error(prev_symbol_sample, prev_mid_sample, symbol_sample)
            error_log.append(error)
            
            # Update timing with loop filter
            # Negative feedback: positive error → reduce step size
            mu += sps - k_p * error
        else:
            # Not enough history yet
            mu += sps
            error_log.append(0.0)
        
        mu_log.append(mu)
        
        # Now sample the midpoint for NEXT iteration (between current and next symbol)
        idx_mid = int(np.round(mu - half_symbol))
        if idx_mid >= 0 and idx_mid < len(rx_f):
            prev_mid_sample = rx_f[idx_mid]
        else:
            prev_mid_sample = None
        
        prev_symbol_sample = symbol_sample
    
    # Extract the middle n_bits for BER calculation
    decisions_out = np.array(decisions_out)
    # Skip the first 'span' symbols and take n_bits
    if len(decisions_out) > span + n_bits:
        decisions_for_ber = decisions_out[span:span + n_bits]
        bits_for_comparison = bits_extended[span:span + n_bits]
    else:
        # Take what we have after skipping span
        decisions_for_ber = decisions_out[span:]
        bits_for_comparison = bits_extended[span:span + len(decisions_for_ber)]
    
    ber = np.mean(decisions_for_ber != bits_for_comparison)

    if save_plots:
        os.makedirs("plots", exist_ok=True)

    if save_plots:
        os.makedirs("plots", exist_ok=True)
    if save_plots:
        os.makedirs("plots", exist_ok=True)
        
        # Plot recovered eye diagram
        plt.figure(figsize=(7, 4))
        symbols_arr = np.array(symbols_out)
        # Plot overlapping 2-symbol traces
        for j in range(0, min(len(symbols_arr) - 2 * sps, 500), 2):
            trace_len = 2 * sps
            if j + trace_len < len(symbols_arr):
                plt.plot(symbols_arr[j:j + trace_len], alpha=0.3, color='C0', linewidth=0.8)
        plt.title("Eye Diagram (After Gardner TED Recovery)")
        plt.xlabel("Sample Index (2 symbol periods)")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig("plots/eye_ted.png", dpi=150)
        plt.close()

        # Plot timing error convergence
        plt.figure(figsize=(10, 4))
        plt.plot(error_log, label="Gardner Timing Error", alpha=0.7)
        plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
        plt.xlabel("Symbol Index")
        plt.ylabel("Timing Error")
        plt.title(f"Gardner TED Error Convergence (k_p={k_p}, initial offset={mu_initial})")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/ted_error.png", dpi=150)
        plt.close()
        
        # Plot mu trajectory
        plt.figure(figsize=(10, 4))
        plt.plot(mu_log, label="Timing Offset (mu)")
        plt.xlabel("Sample Index")
        plt.ylabel("mu (samples)")
        plt.title("Timing Recovery Trajectory")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/ted_mu.png", dpi=150)
        plt.close()
        # Save BER
        os.makedirs("benchmarks/ted", exist_ok=True)
        with open("benchmarks/ted/ber.csv", "w") as f:
            f.write("snr_db,ber,mu_initial,k_p\n")
            f.write(f"{ebn0_db},{ber},{mu_initial},{k_p}\n")

    return {
        "ber": ber,
        "mu_log": mu_log,
        "error_log": error_log,
        "decisions": decisions_for_ber,
        "bits": bits_for_comparison
    }

if __name__ == "__main__":
    print("Running Gardner TED simulation...")
    print("Testing timing recovery with various initial offsets...\n")
    
    # Test multiple scenarios
    test_cases = [
        (0.0, 0.002, "No offset, small k_p"),
        (2.0, 0.005, "Moderate offset +2"),
        (-2.0, 0.005, "Moderate offset -2"),
        (4.0, 0.01, "Large offset +4"),
    ]
    
    for mu_init, kp, desc in test_cases:
        res = run_ted_simulation(n_bits=3000, ebn0_db=15, mu_initial=mu_init, 
                                k_p=kp, save_plots=False)
        
        if len(res['error_log']) > 200:
            avg_error_early = np.mean(np.abs(res['error_log'][50:150]))
            avg_error_late = np.mean(np.abs(res['error_log'][-100:]))
            converged = "✅" if avg_error_late < avg_error_early * 1.2 else "⚠️"
        else:
            avg_error_early = avg_error_late = 0
            converged = "?"
        
        print(f"{desc:25s}: BER={res['ber']:.4f}, " +
              f"|err|_early={avg_error_early:.3f}, |err|_late={avg_error_late:.3f} {converged}")
    
    # Generate plots with best parameters
    print("\nGenerating plots with mu_initial=2.0, k_p=0.005...")
    res_final = run_ted_simulation(n_bits=5000, ebn0_db=12, mu_initial=2.0, 
                                   k_p=0.005, save_plots=True)
    print(f"Final BER: {res_final['ber']:.4f}")
    print("\nPlots saved to plots/eye_ted.png, plots/ted_error.png, and plots/ted_mu.png")