#!/usr/bin/env python3
"""Top-level shim to run the TED simulation from the project root.

This delegates to `src.ted_sim.run_ted_simulation` with a small
CLI to control a few common parameters.
"""
import argparse
from src import ted


def parse_args():
    p = argparse.ArgumentParser(description="Run Gardner TED simulation (shim)")
    p.add_argument("--n_bits", type=int, default=3000)
    p.add_argument("--ebn0_db", type=float, default=15.0)
    p.add_argument("--mu_initial", type=float, default=2.0)
    p.add_argument("--k_p", type=float, default=0.005)
    p.add_argument("--save_plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    res = ted.run_ted_simulation(n_bits=args.n_bits,
                                 ebn0_db=args.ebn0_db,
                                 mu_initial=args.mu_initial,
                                 k_p=args.k_p,
                                 save_plots=args.save_plots)
    print(f"Final BER: {res['ber']:.4f}")


if __name__ == "__main__":
    main()
