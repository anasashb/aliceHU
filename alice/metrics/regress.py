import numpy as np


# Regression Error Metrics
def mse(y_true, y_pred):
    """
    Calculates mean squared error given true and predicted target
    values.
    """

    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Calculates root mean squared error given true and predicted target
    values.
    """

    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Calculates mean absolute error given true and predicted target
    values.
    """

    return np.mean(np.abs(y_true - y_pred))
