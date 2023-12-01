from scipy import stats

def t_test(trues, x, y):
    '''
    Wrapper function for scipy.stats implementation of t-test of independence. 
    Incorporating the Levene test, function automatically picks whether to use t-test or Welch's t-test.

    Args:
        trues (array-like): True values - an unused variable only there to ensure smooth implementation of testing in cases where other other tests require true values (refer to mcnemar tests classifier tests).
        x (array-like): Regression results from one model.
        y (array-like): Regression results from another model.
    Returns:
        pvalue (float): The p-value.
        statistic (float): The test statistic.
    '''
    # Conduct Levene test
    levene_results = stats.levene(x, y)

    # Pick regular t-test if p-value obtained from Levene test > 0.05
    if levene_results.pvalue > 0.05:
        result = stats.ttest_ind(x, y, equal_var=True)
    # Pick Welch t-test if p-value obtained from Levene test <= 0.05
    else:
        result = stats.ttest_ind(x, y, equal_var=False)
    
    # Return p-value and test statistic
    return result[1], result[0]
