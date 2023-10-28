import numpy as np

def pearson(x, y):
    '''
    Calculates the Pearson correlation coefficient given two arrays.
    '''

    # Compute means
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    # Compute numerator
    num = np.sum((x - x_bar) * (y - y_bar))
    
    # Compute denominator
    denom = np.sqrt(np.sum((x - x_bar)**2) * np.sum((y - y_bar)**2))

    # Compute correlation coefficient
    r = num / denom

    return r
