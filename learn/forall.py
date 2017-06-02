
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


error_messages = {
    "No clear target in training data": 
        ("The training data must have " 
         "exactly one more column than " 
         "the test data."),
    "Training data has too many columns":
        ("The training data has more "
         "than one column different than "
         "the testing data: %s")
}

def X_y_split(X_train, X_test):
    """
    Determines which variables are the target
    and which are the features. Returns just
    The X and y data in the training dataset
    as a tuple.
    
    Example usage:
    X, y = learn.X_y_split(X_train, X_test)
    
    Parameters
    ----------
    X_train: pandas dataframe
        The data that has the target in it.
    
    X_test: pandas dataframe
        The data that does not have the target in it.
    """
    n_train_cols = X_train.shape[1]
    n_test_cols = X_test.shape[1]
    
    if n_train_cols != n_test_cols + 1:
        msg = error_messages["No clear target in training data"]
        raise ValueError(msg)
        
    test_columns = set(X_test.columns)
    train_columns = set(X_test.columns)
    target_columns = train_columns - test_columns
    if len(target_columns) > 1:
        key = "Training data has too many columns"
        msg_ = error_messages[key]
        msg = msg_ % str(target_columns)
        raise ValueError(msg)

    extra_columns_in_test = test_columns - train_columns