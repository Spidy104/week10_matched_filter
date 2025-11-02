"""Package initializer for the simulation package.

This module re-exports the main simulation modules so top-level shim
scripts (in the project root) can import and invoke them without
changing their internal relative imports.

Usage from project root:
	python bpsk_sim.py    # calls src.bpsk_sim.main()
	python ted_sim.py     # calls src.ted_sim.run_ted_simulation(...)
"""

from . import bpsk_sim as bpsk
from . import ted_sim as ted

__all__ = ["bpsk", "ted"]
