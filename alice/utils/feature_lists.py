from itertools import chain

def dummy_grouper(feature_list, dummy_list):
    '''
    Function is meant to handle one-hot-encoded categorical variables.

    Args:
        feature_list: A list of features obtained using list(df.columns).
        dummy_list: A list of lists containing groups of one-hot-encoded variables. Each entry should be an exact name of the column as a string.
    Use Case:
        The function can be used when user has one or more one-hot-encoded categorical variables.
        Let X an Z denote three-class categorical variables in a dataframe.
        Assuming the user encodes them using pd.get_dummies() and sets one class per each feature as base:
        The dummy list should look like [['x_dummy_1', 'x_dummy_2'], ['z_dummy_1', 'z_dummy_2']].
        The function will then pick these groups up and treat ['x_dummy_1', 'x_dummy_2'] and ['z_dummy_1', 'z_dummy_2'] as individual features.
        Therefore, it gets ensured that the _deselect_feature method treats each one-hot-encoded variable as one feature to drop during iteration.
    '''
    # Empty container for feature list with grouped dummy columnes
    missing_features = []
    feature_list_with_grouped_dummies = []
    # Flatten the dummy_list of lists into a single list
    dummy_list_flat = [item for sublist in dummy_list for item in sublist]
    # Iterate per feature in the feature list
    for feature in feature_list:
        # Append all features that are not one-hot-encoded versions of categorical variables to the new list
        if feature not in dummy_list_flat:
            feature_list_with_grouped_dummies.append(feature)
    # Iterate over groups in the dummy list of lists and check that all features in a group is in the original feature list
    for group in dummy_list:
        if all(feature in feature_list for feature in group):
            # Append group if condition satisfied
            feature_list_with_grouped_dummies.append(group)
        else:
            missing_features = [feature for feature in group if feature not in feature_list]
            raise ValueError(f'The following features from the group {group} are not found in the original feature list: {missing_features}')    
    
    return feature_list_with_grouped_dummies


def feature_fixer(feature_list, features_to_fix):
    '''
    Fixes a set of features so that they do not get removed in _deselect_feature methods.
    Args:
        feature_list: List of features, either obtained as list(df.columns) or after conducting dummy grouping using dummy_grouper.
        features_to_fix: List of features to fix, given as as strings containing column names of the dataframe. 
    Example:
        features_to_fix can be a list such as ['feature_1', 'feature_1', 'feature_3']
        In case of dummy grouping it could also be a list such as ['feature_1', 'feature_2', ['feature_3_dummy_1', 'feature_3_dummy_2', 'feature_3_dummy_3']]    
    '''
    # Empty container for a feature list that will not include those that we do not want to get deselected
    feature_list_without_fixed_features = []
    # Simple loop 
    for item in feature_list:
        if item not in features_to_fix:
            feature_list_without_fixed_features.append(item)
    return feature_list_without_fixed_features

def feature_list_flatten(feature_list):
    '''
    Flattens a feature list that includes grouped feature names.
    Returns:
        ['X1', 'X2', 'X3', ['X4_dummy_1', 'X4_dummy_2', 'X4_dummy_3']] >>> ['X1', 'X2', 'X3', 'X4_dummy_1', 'X4_dummy_2', 'X4_dummy_3']
    '''
    return list(chain.from_iterable([item if isinstance(item, list) else [item] for item in feature_list]))
