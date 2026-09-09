"""Real-axis PAM demodulation for already symbol-timed samples."""

import numpy as np

from .common import DemodulationResult


def demodulate_pam(samples: np.ndarray, order: int = 4) -> DemodulationResult:
    """Slice a PAM constellation against its own estimated axis and scale.

    The generator builds PAM4 as `levels = [-3, -1, 1, 3]` normalised to unit average
    power, mapped from bits as natural binary, MSB-first (`bits_to_symbol_indices`), so
    index 0..3 runs from the most negative level to the most positive. This inverts
    exactly that mapping - it is not a new convention.

    Two properties are estimated from the samples alone, never supplied:

    * **Axis.** PAM is a real one-dimensional constellation, so its samples lie on a line
      through the origin. That line has 180-degree symmetry, which is what makes
      `angle(mean(x^2)) / 2` a valid estimate of its rotation: squaring folds the two
      opposite lobes onto one. Without this a rotated capture would collapse onto the
      wrong projection.
    * **Scale.** Upstream stages deliver unit-RMS symbols while this grid has average
      power `mean(levels^2)`, so the samples are rescaled to the grid rather than the
      grid to the samples. This is the same defect that made QAM slice everything onto
      the innermost pair before BANK.md Entry 007.
    """

    if order != 4:
        raise ValueError("PAM order must be 4")
    values = np.asarray(samples, dtype=np.complex64).astype(np.complex128)
    levels = np.arange(-(order - 1), order, 2, dtype=np.float64)
    if values.size == 0:
        return DemodulationResult(
            np.array([], dtype=np.complex64), np.array([], dtype=np.uint8),
            "PAM4", {"order": order})

    # Estimate the constellation axis and rotate it back onto the real axis.
    squared_mean = np.mean(values**2)
    if np.abs(squared_mean) > 0:
        values = values * np.exp(-0.5j * np.angle(squared_mean))
    projected = np.real(values)

    # Rescale to the grid's average power; the absolute scale carries no information.
    grid_power = float(np.mean(levels**2))
    power = float(np.mean(projected**2))
    if power > 0:
        projected = projected * np.sqrt(grid_power / power)

    indices = np.argmin(np.abs(projected[:, None] - levels), axis=1).astype(np.uint8)
    bits_per_symbol = int(np.log2(order))
    bits = ((indices[:, None] >> np.arange(bits_per_symbol - 1, -1, -1)) & 1)
    symbols = levels[indices].astype(np.complex64)
    return DemodulationResult(symbols, bits.astype(np.uint8).ravel(), "PAM4",
                              {"order": order})
