import numpy as np


def cohen_kappa(x, y):
    """
    Calculates the Cohen's kappa given two arrays.
    """

    # Confusion Matrix
    a = np.sum((x == 1) & (y == 1))
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))
    d = np.sum((x == 0) & (y == 0))

    # Total (reusable)
    total = a + b + c + d
    if total == 0:
        return 0

    # Observed proportionate agreement
    p_o = (a + d) / total

    # Overall random agreement probability (p_yes + p_no)
    p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / total**2

    # Cohen's kappa
    return (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0
