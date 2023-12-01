import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_binomial(trues, x, y):
    '''
    Wrapper function for Statsmodels implementation of McNemar test of homogeneity with binomial distribution.
    Under H_0, binary classification labels are assumed to be homogeneous. 
    Args:
        trues (array-like): True binary labels.
        x (array-like): Binary classification results from one model.
        y (array-like): Binary classification results from another model.
    Returns:
        pvalue (float): The p-value.        
        statistic (float): The test statistic.
    '''
    # Generate contingency table (refer to mcnemar_table)
    table = mcnemar_table(trues, x, y)
    # Conduct test: exact=True -> function uses binomial dist.
    result = mcnemar(table, exact=True)

    return result.pvalue, result.statistic

def mcnemar_chisquare(trues, x, y):
    '''
    Wrapper function for Statsmodels implementation of McNemar test of homogeneity with chi-square distribution.
    Under H_0, binary classification labels are assumed to be homogeneous. 
    Args:
        trues (array-like): True binary labels.
        x (array-like): Binary classification results from one model.
        y (array-like): Binary classification results from another model.
    Returns:
        pvalue (float): The p-value.        
        statistic (float): The test statistic.
    '''
    # Generate contingency table (refer to mcnemar_table)
    table = mcnemar_table(trues, x, y)
    # Conduct test: exact=True -> function uses binomial dist.
    result = mcnemar(table, exact=False)

    return result.pvalue, result.statistic


def mcnemar_table(trues, x, y):
    '''
    Generates a 2x2 contingency table for McNemar's test.
    Args:
        trues (array-like): True binary labels.
        x (array-like): Binary classification results from one model.
        y (array-like): Binary classification results from another model.
    Returns:
        table (np.array): (2,2) table where such that:
            - table[0,0]: Count of instances where both models classify correclty.
            - table[0,1]: Count of instances where first model (x) classifies correctly, but second (y) does not.
            - table[1,0]: Count of instances where first model (x) classifies incorrectly, but secon (y) is correct.
            - table[1,1]: Count of instances where both models classify incorrectly.
    '''
    # Error handling if the arrays dont contain equal amount of instances
    if len(trues) == len(x) == len(y):
        pass
    else: 
        raise ValueError('All inputs are not of the same length. Ensure each input has equal number of observations.')
    
    # We begin with initializing a contingency table that contains zeros only
    table = np.zeros((2,2))
    
    # Iterate through true classes and predictions and increment relevant table entry
    for true, x_pred, y_pred in zip(trues, x, y):
        # Case when both models correct
        if x_pred == true and y_pred == true:
            # Increment first row first column by 1
            table[0,0] += 1
        # Case when model x correct and model y incorrect
        elif x_pred == true and y_pred != true:
            # Increment first row second column by 1
            table[0,1] += 1
        # Case when model x incorrect and y correct
        elif x_pred != true and y_pred == true:
            # Increment second row first column by 1
            table[1,0] += 1
        # Case when both models incorrect
        elif x_pred != true and y_pred != true:
            table[1,1] += 1
    # Return final table
    return table
