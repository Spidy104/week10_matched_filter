import numpy as np

def rrc_taps(beta=0.35, sps=8, span=6):
    """
    Generate Root Raised Cosine (RRC) filter taps.
    
    Parameters:
        beta (float): Roll-off factor (0 < beta <= 1)
        sps (int): Samples per symbol
        span (int): Filter span in symbols (total length = span * sps + 1)
    
    Returns:
        np.ndarray: Unity-gain normalized RRC filter taps (sum = 1)
    """
    if not (0 < beta <= 1):
        raise ValueError("beta must be in (0, 1]")
    if sps <= 0 or span <= 0:
        raise ValueError("sps and span must be positive integers")

    N = span * sps
    t = np.arange(-N // 2, N // 2 + 1) / float(sps)
    taps = np.zeros_like(t)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            # Center point
            taps[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(abs(ti) - 1.0 / (4 * beta)) < 1e-12:
            # Singular points
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den

    # Normalize for unity gain (DC gain = 1)
    tap_sum = np.sum(taps)
    if abs(tap_sum) < 1e-12:
        raise RuntimeError("Generated RRC filter has zero sum")
    taps /= tap_sum
    return taps