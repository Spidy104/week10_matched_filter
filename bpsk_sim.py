#!/usr/bin/env python3
"""Top-level shim to run the BPSK simulation from the project root.

This simply delegates to `src.bpsk_sim.main()` so you can run:

    python bpsk_sim.py --help
    python bpsk_sim.py --n_bits 1000 --snr_min 2 --snr_max 2
"""
from src import bpsk


def main():
    # Delegate to the package's main function
    bpsk.main()


if __name__ == "__main__":
    main()
