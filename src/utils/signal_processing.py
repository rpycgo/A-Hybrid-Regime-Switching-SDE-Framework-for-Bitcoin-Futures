import pandas as pd
import numpy as np


def apply_sticky_breakout_filter(binary_signals: np.ndarray, minimum_duration: int = 5) -> np.ndarray:
    """
    Validate signals only if the state persists for a minimum duration.
    Uses vectorized rolling sum for high performance.

    Args:
        binary_signals (np.ndarray): Array of 0 and 1 indicating raw entry signals.
        minimum_duration (int): Required consecutive periods for validation.

    Returns:
        np.ndarray: Filtered signals (1 if valid, 0 otherwise).
    """
    # Create a rolling window sum. If the sum equals minimum_duration,
    # it means the signal has been '1' for the entire window.
    signal_series = pd.Series(binary_signals)
    sticky_mask = signal_series.rolling(window=minimum_duration).sum() == minimum_duration

    # Convert boolean mask to integer (1/0)
    return sticky_mask.astype(int).values
