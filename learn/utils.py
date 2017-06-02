
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None


def make_regression_data(source="boston", 
                         missing_data=None, 
                         categorical=None, 
                         outliers=None):
    """
    Utility function to assist in loading different 
    sample datasets. Returns training data (that 
    contains the target) and testing data (that
    does not contain the target).
    
    Parameters
    ----------
    source: string, optional (default="boston")
        The specific dataset to load. Options:
        - "boston": Boston housing dataset
        
    missing_data: bool or NoneType (default=None)
        To be implemented
        Determines if there is missing data
        
    categorical: bool or NoneType (default=None)
        To be implemented
        Determines if there is categorical data
        
    outliers: bool or NoneType (default=None)
        To be implemented
        Determines if there are outliers in the dataset
    """
    boston_data = load_boston()
    X = pd.DataFrame(data=boston_data.data, 
                     columns=boston_data.feature_names)
    y = pd.Series(data=boston_data.target)
    X_train, X_test, y_train, _ = train_test_split(X, 
                                                   y, 
                                                   test_size=.5,
                                                   random_state=42)
    X_train["target"] = y_train
    return X_train, X_test
    