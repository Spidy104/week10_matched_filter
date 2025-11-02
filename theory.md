# Theory notes — matched filtering, RRC pulses, AWGN scaling, and Gardner TED

This document provides an expanded set of notes and derivations for the
simulations in this repository (BPSK with RRC pulse-shaping, matched
filtering, AWGN channel, and Gardner timing recovery).

## System overview

The pipeline implemented in the code is a standard digital communications
chain for BPSK:

- Symbols (±1) are generated and oversampled (e.g. with `np.repeat`) with
    samples-per-symbol `sps`.
- A root-raised-cosine (RRC) pulse-shaping FIR filter is applied at the
    transmitter.
- The channel is AWGN (additive white Gaussian noise).
- The receiver applies a matched RRC filter (same taps) and then samples
    the filtered waveform at symbol instants.
- Blind timing recovery (optional) uses the Gardner Timing Error Detector
    (TED) to adjust the sampling instant.

The notes below cover the RRC impulse, filter normalization and delay,
the AWGN scaling for Eb/N0, sampling and offsets, and Gardner TED
derivation and practical tips.

## Root-Raised-Cosine (RRC) pulse

The RRC impulse response (time domain) is commonly written as:

$$
t_{\text{RRC}}(t)=\frac{\sin\left(\pi t/T (1-\beta)\right)+4\beta (t/T)\cos\left(\pi t/T (1+\beta)\right)}{\pi t/T \left(1-(4\beta t/T)^2\right)}
$$

where $T$ is the symbol period and $\beta$ is the roll-off factor
(0 ≤ $\beta$ ≤ 1). In a sampled implementation we evaluate the continuous
formula at discrete sample instants separated by $T_s = T / \text{sps}$.

Practical notes for implementation:

- Filter length (number of taps) is typically `L = span * sps + 1`, where
    `span` is the filter half-length in symbols and `sps` is samples per
    symbol.
- Group delay per FIR = `(L-1)/2` samples. For a matched TX+RX pair
    (two identical RRC filters) the total matched filter group delay in
    samples is `L - 1` (i.e., twice the single-filter group delay).
- In this project the taps are normalized for unity DC gain (sum(taps)=1).
    This preserves the low-frequency gain and makes the amplitude scaling
    predictable for Eb/N0 calculations.

Example: for `span=6` and `sps=8`, the filter length is `L=6*8+1=49`.
Single-filter group delay = 24 samples; matched-pair group delay = 48
samples.

## AWGN and Eb/N0 scaling (derivation)

We want to add AWGN so that the simulated received waveform achieves a
target Eb/N0 (where Eb is energy per bit and N0 is the one-sided noise
power spectral density). For real baseband samples, the noise variance
per real sample (one-sided) is $N_0/2$.

Definitions and relationships:

- Let $P_{\text{avg}}$ be the average signal power per sample after
    filtering: $P_{\text{avg}} = E[ x[n]^2 ] = \text{mean}(x**2)$.
- Samples-per-symbol = `sps`.
- Energy per symbol: $E_s = P_{\text{avg}} \times T_{\text{symbol}}$.
    With normalized sampling where the sample period is 1, $E_s =
    P_{\text{avg}} \times \text{sps}$.
- For BPSK with 1 bit/symbol: $E_b = E_s$.
- Target Eb/N0 (linear): $\gamma_b = 10^{\text{Eb/N0}_{\text{dB}}/10}$.

Noise power per sample derivation:

1. $N_0 = E_b / \gamma_b$.
2. Noise variance per (real) sample = $\sigma^2 = N_0/2 = \dfrac{E_b}{2\gamma_b}$.
3. Substitute $E_b = P_{\text{avg}} \cdot \text{sps}$:

$$
\sigma^2 = \dfrac{P_{\text{avg}} \cdot \text{sps}}{2 \gamma_b}.
$$

Therefore the AWGN standard deviation to add to each real sample is
$\sigma = \sqrt{\dfrac{P_{\text{avg}} \cdot \text{sps}}{2 \gamma_b}}$.

In code this is implemented by measuring `signal_power = np.mean(x**2)`,
computing `ebn0_linear = 10**(ebn0_db/10.0)` and then using the formula
above to compute `sigma` before drawing `N(0, sigma^2)` samples.

Notes:

- The correctness of this scaling depends on consistent normalization of
    the pulse shape (unity-sum taps in this repo) and on `sps` being the
    actual number of samples per symbol.
- If you normalize filter energy instead of DC gain you must account for
    that scaling when computing `P_{\text{avg}}`.

## Sampling and timing offset

Because the transmit waveform is oversampled, selecting where to sample
relative to the filter group delay matters:

- Total matched filter delay (in samples) = `len(taps) - 1`.
- Choose sampling start index as `sampling_start = total_group_delay + offset`.
- Then sample every `sps` samples starting at `sampling_start`.

When using `np.repeat(symbols, sps)` the waveform is piecewise-constant
per symbol period. The optimal sampling location is typically near the
midpoint of the flat region (often `sps//2 - 1` in these implementations
because of zero-based rounding and filter delay). If the sample index
falls near an edge (transition) you will see increased ISI and a BER
floor even at high SNR.

## Gardner Timing Error Detector (TED)

The Gardner TED is widely used for blind timing recovery in systems that
have samples at symbol rate and an accessible mid-sample (i.e., samples
at $T/2$). The detector uses three successive samples:

- $y[(n-1)T]$: previous-symbol sample,
- $y[(n-1)T + T/2]$: midpoint sample between previous and current,
- $y[nT]$: current-symbol sample.

The classical error estimate (data-aided or decision-directed) is:

$$
e[n] = (y[nT] - y[(n-1)T]) \cdot y\left[nT - \dfrac{T}{2}\right].
$$

Interpretation:

- $e[n] > 0$ means the sample is (by the sign convention used here)
    typically *late* and the loop should advance earlier (reduce the
    sampling index). The exact sign mapping depends on the update rule
    used in the implementation — match conventions between detector and
    loop update.
- $e[n] < 0$ means the sample is too early.

Loop update (proportional-only, as used in the code):

$$
\mu_{k+1} = \mu_k + sps - k_p \cdot e[n]
$$

where $\mu$ is the running sample index (in samples) used to pick the
next symbol sample, `sps` is the nominal step between symbols, and
`k_p` is the proportional loop gain. The code implements the midpoint
sample for the current error as the midpoint sampled in the previous
iteration (so the three samples are temporally ordered: previous
symbol, midpoint between previous and current, current symbol).

Practical tuning tips for Gardner TED

- Start with a small `k_p` (e.g. 0.001–0.01). Too large a gain will
    oscillate; too small will converge slowly.
- Check the sign convention by injecting a known offset and verifying
    that `mu` moves in the correct direction.
- Plot `mu` (timing trajectory) and the error `e[n]` to visually confirm
    convergence.

## Practical example (numerical)

Using typical repo defaults:

- `sps = 8`, `beta = 0.35`, `span = 6` → `L = 6*8 + 1 = 49` taps.
- Single-filter group delay = 24 samples → matched-pair delay = 48.

If you generate `n_bits_with_margin = n_bits + 2*span` symbols and
sample starting at `sampling_start = 48 + offset`, then the k-th sample
index used for symbol k is `sampling_start + k*sps` (rounded as needed).

## Common pitfalls

- Forgetting to compensate group delay leads to misaligned sampling and
    artificially high BER.
- Using the wrong energy normalization when computing sigma breaks the
    Eb/N0 relationship.
- Sampling the midpoint incorrectly (e.g., taking the midpoint after
    the current symbol instead of before it) breaks Gardner TED ordering
    and prevents convergence.

## Recommended parameters (empirical)

- `sps`: 8 — good for visualization and reasonable accuracy.
- `beta`: 0.25–0.35 — balances bandwidth vs ISI.
- `span`: 6 — typical for RRC with low ripple.
- `k_p`: 0.001–0.01 — tune based on convergence speed vs stability.

## References

- John G. Proakis, Masoud Salehi — "Digital Communications" (Gardner TED
    and synchronization chapters)
- Gardner, F. M. — classical timing recovery literature and tutorials

---

If you want I can convert this into a Jupyter notebook with interactive
plots (eye diagrams, mu trajectories) and numerical checks of the sigma
formula. Tell me which section you'd like expanded or plotted.
