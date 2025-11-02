import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

def plot_eye(rx_filtered, sps, fname):
    """
    Plot eye diagram from matched-filter output.
    
    Parameters:
        rx_filtered (np.ndarray): Received filtered signal
        sps (int): Samples per symbol
        fname (str): Output filename for the plot
    """
    L = len(rx_filtered)
    mid = L // 2
    window = sps * 40
    start = max(0, mid - window // 2)
    end = min(L, start + window)
    segment = rx_filtered[start:end]
    
    plt.figure(figsize=(6, 3))
    traces_plotted = 0
    for i in range(0, len(segment) - 2 * sps, sps):
        if i + 2 * sps <= len(segment):
            plt.plot(segment[i:i + 2 * sps], color="C0", alpha=0.4)
            traces_plotted += 1
    
    if traces_plotted == 0:
        plt.text(0.5, 0.5, "Insufficient data for eye diagram", 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.title("Eye Diagram (Matched-Filter Output)")
    plt.xlabel("Samples (per symbol interval)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def plot_ber_curve(snr_list, ber_clean, ber_offset, offset_samples, fname):
    """
    Plot BER curves for ideal and offset timing.
    
    Parameters:
        snr_list (list): List of SNR values in dB
        ber_clean (list): BER values for ideal timing
        ber_offset (list): BER values for timing offset
        offset_samples (int): Timing offset in samples
        fname (str): Output filename for the plot
    """
    plt.figure(figsize=(6, 4))
    plt.semilogy(snr_list, ber_clean, "o-", label="Ideal timing", linewidth=2)
    plt.semilogy(snr_list, ber_offset, "s--", label=f"Offset = {offset_samples} samples", linewidth=2)
    plt.xlabel("Eb/N0 (dB)", fontsize=11)
    plt.ylabel("BER", fontsize=11)
    plt.title("BER Performance Comparison")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=10)
    
    # Set reasonable y-axis limits if BER values are available
    if len(ber_clean) > 0 and len(ber_offset) > 0:
        min_ber = min(min(ber_clean), min(ber_offset))
        max_ber = max(max(ber_clean), max(ber_offset))
        if min_ber > 0 and max_ber > 0:
            plt.ylim([max(min_ber * 0.5, 1e-6), min(max_ber * 2, 1.0)])
    
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

def save_spectrum(taps, sps, fname):
    """
    Plot RRC filter frequency response.
    
    Parameters:
        taps (np.ndarray): RRC filter taps
        sps (int): Samples per symbol
        fname (str): Output filename for the plot
    """
    nfft = 2048
    H = fft.fftshift(fft.fft(taps, nfft))
    # Frequency axis normalized by sampling rate (Fs = sps * symbol_rate)
    # For normalized frequency: -0.5 to 0.5 (where 1.0 = sampling rate)
    freqs = np.linspace(-0.5, 0.5, nfft)
    
    plt.figure(figsize=(7, 4))
    plt.plot(freqs, 20 * np.log10(np.abs(H) + 1e-12), linewidth=1.5)
    plt.title("RRC Filter Frequency Response")
    plt.xlabel("Normalized Frequency (f/Fs)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    plt.xlim([-0.5, 0.5])
    
    # Add a secondary x-axis showing frequency in terms of symbol rate
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax2.set_xlim(-0.5 * sps, 0.5 * sps)
    ax2.set_xlabel("Frequency (f/Rs, where Rs = symbol rate)")
    
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()