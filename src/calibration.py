"""Calibration module for gel electrophoresis ladder-based size estimation.

Provides log-linear regression calibration that maps vertical band positions
(pixel y-coordinates) to DNA fragment sizes in base pairs (bp). Supports
multiple standard DNA ladder types.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Standard DNA Ladder Definitions
# ---------------------------------------------------------------------------
# Sizes are listed in descending order (largest fragments first), matching
# the expected migration order on a gel (top → bottom = large → small).

LADDER_TYPES: Dict[str, List[int]] = {
    "1kb_plus": [
        10000, 8000, 6000, 5000, 4000, 3000, 2500,
        2000, 1500, 1200, 1000, 900, 800, 700, 600, 500,
    ],
    "100bp": [
        1517, 1200, 1000, 900, 800, 700, 600,
        500, 400, 300, 200, 100,
    ],
    "neb_1kb": [
        10002, 8001, 6001, 5001, 4001, 3001,
        2000, 1500, 1000, 517, 500,
    ],
}


def get_available_ladders() -> List[str]:
    """Return the list of supported ladder type keys.

    Returns:
        List of ladder type identifiers that can be passed to
        :func:`calibrate_ladder`.
    """
    return list(LADDER_TYPES.keys())


def calibrate_ladder(
    ladder_positions: Union[List[float], np.ndarray],
    ladder_type: str = "1kb_plus",
    known_sizes: Optional[List[int]] = None,
) -> Tuple[Callable[[float], int], Tuple[float, float], float, List[Tuple[float, int]]]:
    """Calibrate lane migration using ladder band y-coordinates.

    Fits a log-linear model: ``ln(size_bp) = slope * y + intercept``.

    When the number of detected bands does not match the number of known
    ladder sizes, the algorithm searches for the contiguous subset of
    known sizes that maximises the R² fit (with a negative-slope
    constraint, since larger fragments migrate less).

    Args:
        ladder_positions: Detected band centroid y-coordinates (pixels).
        ladder_type: Key into :data:`LADDER_TYPES` (e.g. ``"1kb_plus"``).
            Ignored when *known_sizes* is provided explicitly.
        known_sizes: Optional explicit list of known fragment sizes (bp),
            ordered largest → smallest. Overrides *ladder_type*.

    Returns:
        A 4-tuple:

        * **position_to_bp** — function mapping a y-coordinate → estimated bp.
        * **(slope, intercept)** — regression coefficients.
        * **r_squared** — R² goodness-of-fit metric.
        * **matched_pairs** — list of (position, size) pairs used in the fit.

    Raises:
        KeyError: If *ladder_type* is not found in :data:`LADDER_TYPES`.
    """
    # Resolve known sizes
    if known_sizes is None:
        if ladder_type not in LADDER_TYPES:
            raise KeyError(
                f"Unknown ladder type '{ladder_type}'. "
                f"Available: {get_available_ladders()}"
            )
        known_sizes = LADDER_TYPES[ladder_type]

    m = len(ladder_positions)
    n = len(known_sizes)

    if m == 0:
        # No bands detected — return a dummy calibration
        dummy_func: Callable[[float], int] = lambda y: 0
        return dummy_func, (0.0, 0.0), 0.0, []

    # Sort detected coordinates ascending (migration ↑ down the gel)
    positions = np.sort(np.asarray(ladder_positions, dtype=float))

    # Sort known sizes descending (largest at top of gel)
    sizes = np.sort(known_sizes)[::-1]

    matched_sizes: np.ndarray
    matched_positions: np.ndarray

    if m == n:
        matched_sizes = sizes
        matched_positions = positions
    elif m < n:
        # Fewer detected bands than known sizes.
        # Search for the contiguous subset of known sizes that yields
        # the highest R² with a physically meaningful (negative) slope.
        best_r2 = -1.0
        best_subset = None

        for i in range(n - m + 1):
            subset = sizes[i : i + m]
            slope_i, intercept_i, r_val_i, _, _ = linregress(
                positions, np.log(subset)
            )
            r2_i = r_val_i ** 2 if not np.isnan(r_val_i) else 0.0

            # Prefer negative slope (larger y → smaller size)
            if slope_i < 0 and r2_i > best_r2:
                best_r2 = r2_i
                best_subset = subset

        if best_subset is not None:
            matched_sizes = best_subset
            matched_positions = positions
        else:
            # Fallback — take the first m sizes
            matched_sizes = sizes[:m]
            matched_positions = positions
    else:
        # More detected bands than known sizes — take the top n positions
        matched_positions = positions[:n]
        matched_sizes = sizes

    # --- Final regression on the matched pairs ---
    log_sizes = np.log(matched_sizes)
    slope, intercept, r_val, _, _ = linregress(matched_positions, log_sizes)
    r_squared = r_val ** 2 if not np.isnan(r_val) else 0.0

    # Mapping function: y-coordinate → estimated base-pair size
    def position_to_bp(y: float) -> int:
        """Map a pixel y-coordinate to an estimated fragment size (bp)."""
        log_size = slope * y + intercept
        estimated_size = np.exp(log_size)
        # Cap minimum size to 50 bp to avoid nonsensical extrapolation
        return int(max(50, estimated_size))

    matched_pairs = list(zip(matched_positions.tolist(), matched_sizes.tolist()))

    return position_to_bp, (slope, intercept), r_squared, matched_pairs
