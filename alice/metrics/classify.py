import numpy as np


# Classification Metrics
def accuracy(y_true, y_pred):
    """
    Calculates classification accuracy given true and predicted target
    values.
    """

    return np.sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred):
    """
    Calculates classification precision given true and predicted target
    values.
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tp_and_fp = np.sum(y_pred == 1)

    return tp / tp_and_fp if tp_and_fp != 0 else np.nan


def recall(y_true, y_pred):
    """
    Calculates classification recall given true and predicted target
    values.
    """

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tp_and_fn = np.sum(y_true == 1)

    return tp / tp_and_fn if tp_and_fn != 0 else np.nan


def f1(y_true, y_pred):
    """
    Calculates the F1 score given true and predicted target values.
    """

    pr = precision(y_true, y_pred)
    re = recall(y_true, y_pred)

    return 2 * (pr * re) / (pr + re) if pr + re != 0 else np.nan
