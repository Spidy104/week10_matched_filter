import numpy as np
from .rrc_utils import rrc_taps
import pytest


def test_rrc_unity_gain():
    """Ensure RRC filter has unity gain (sum of taps = 1)."""
    taps = rrc_taps(beta=0.35, sps=8, span=6)
    tap_sum = np.sum(taps)
    assert abs(tap_sum - 1.0) < 1e-10, f"Sum of taps = {tap_sum:.12f}, expected 1.0"

def test_rrc_symmetry():
    """RRC filter should be symmetric."""
    taps = rrc_taps(beta=0.35, sps=8, span=6)
    assert np.allclose(taps, taps[::-1], atol=1e-10), "Filter taps should be symmetric"

def test_rrc_extreme_beta():
    """Test edge cases for beta."""
    # Very small beta
    taps_small = rrc_taps(beta=0.01, sps=4, span=4)
    assert np.sum(taps_small) == pytest.approx(1.0, abs=1e-10)
    
    # Maximum beta
    taps_max = rrc_taps(beta=1.0, sps=4, span=4)
    assert np.sum(taps_max) == pytest.approx(1.0, abs=1e-10)

def test_rrc_invalid_beta():
    """Test that invalid beta values raise errors."""
    with pytest.raises(ValueError):
        rrc_taps(beta=0.0, sps=4, span=4)  # beta must be > 0
    
    with pytest.raises(ValueError):
        rrc_taps(beta=1.5, sps=4, span=4)  # beta must be <= 1
    
    with pytest.raises(ValueError):
        rrc_taps(beta=-0.1, sps=4, span=4)  # negative beta

def test_rrc_invalid_sps():
    """Test that invalid samples per symbol raise errors."""
    with pytest.raises(ValueError):
        rrc_taps(beta=0.35, sps=0, span=4)
    
    with pytest.raises(ValueError):
        rrc_taps(beta=0.35, sps=-1, span=4)

def test_rrc_invalid_span():
    """Test that invalid span values raise errors."""
    with pytest.raises(ValueError):
        rrc_taps(beta=0.35, sps=4, span=0)
    
    with pytest.raises(ValueError):
        rrc_taps(beta=0.35, sps=4, span=-1)

def test_rrc_filter_length():
    """Verify filter length is correct: span * sps + 1."""
    for span in [2, 4, 6, 8]:
        for sps in [2, 4, 8]:
            taps = rrc_taps(beta=0.35, sps=sps, span=span)
            expected_length = span * sps + 1
            assert len(taps) == expected_length, f"Expected {expected_length}, got {len(taps)}"

def test_rrc_center_tap_maximum():
    """For typical parameters, center tap should be the maximum."""
    taps = rrc_taps(beta=0.35, sps=8, span=6)
    center_idx = len(taps) // 2
    assert taps[center_idx] == np.max(taps), "Center tap should be maximum for typical RRC"

def test_rrc_different_parameters():
    """Test RRC with various realistic parameter combinations."""
    params = [
        (0.22, 4, 6),   # Common in satellite communications
        (0.35, 8, 8),   # Common in digital communications
        (0.5, 4, 4),    # Moderate roll-off
        (0.25, 16, 10), # High oversampling
    ]
    
    for beta, sps, span in params:
        taps = rrc_taps(beta=beta, sps=sps, span=span)
        assert np.sum(taps) == pytest.approx(1.0, abs=1e-10)
        assert np.allclose(taps, taps[::-1], atol=1e-10)
        assert len(taps) == span * sps + 1

def test_rrc_no_nans_or_infs():
    """Ensure filter doesn't produce NaN or Inf values."""
    taps = rrc_taps(beta=0.35, sps=8, span=6)
    assert not np.any(np.isnan(taps)), "Filter contains NaN values"
    assert not np.any(np.isinf(taps)), "Filter contains Inf values"

def test_rrc_singular_points():
    """Test that singular points (t = ±1/(4β)) are handled correctly."""
    # Choose parameters that will hit singular points
    beta = 0.25  # 1/(4*0.25) = 1.0
    sps = 8
    span = 8
    taps = rrc_taps(beta=beta, sps=sps, span=span)
    
    assert not np.any(np.isnan(taps)), "Singular points not handled correctly"
    assert np.sum(taps) == pytest.approx(1.0, abs=1e-10)

if __name__ == "__main__":
    test_rrc_unity_gain()
    test_rrc_symmetry()
    test_rrc_extreme_beta()
    test_rrc_invalid_beta()
    test_rrc_invalid_sps()
    test_rrc_invalid_span()
    test_rrc_filter_length()
    test_rrc_center_tap_maximum()
    test_rrc_different_parameters()
    test_rrc_no_nans_or_infs()
    test_rrc_singular_points()
    print("✅ All RRC tests passed.")