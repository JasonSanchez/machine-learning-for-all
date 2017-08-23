
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

error_messages = {
    "No clear target in training data": 
        ("The training data must have " 
         "exactly one more column than " 
         "the test data."),
    "Training data has too many columns":
        ("The training data has more "
         "than one column different than "
         "the testing data: %s"),
    "Column names inconsistent":
        ("The training columns and the "
         "test columns must have "
         "identical names excepts for "
         "the target variables. "
         "Different columns: %s")
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
    X_train = X_train.copy()
    n_train_cols = X_train.shape[1]
    n_test_cols = X_test.shape[1]
    
    if n_train_cols != n_test_cols + 1:
        msg = error_messages["No clear target in training data"]
        raise ValueError(msg)
        
    test_columns = set(X_test.columns)
    train_columns = set(X_train.columns)
    target_columns = train_columns - test_columns
    if len(target_columns) > 1:
        key = "Training data has too many columns"
        msg_ = error_messages[key]
        msg = msg_ % str(target_columns)
        raise ValueError(msg)

    extra_columns_in_test = test_columns - train_columns
    if extra_columns_in_test:
        key = "Column names inconsistent"
        msg_ = error_messages[key]
        msg = msg_ % str(extra_columns_in_test)
        raise ValueError(msg)     

    y_name = target_columns.pop()
    y = X_train.pop(y_name)
    return X_train, y


def X_to_train_test(X, target_name, test_size=.05):
    X = X.copy()
    y = X.pop(target_name)
    X_train, X_test, y_train, _ = train_test_split(X, 
                                                   y, 
                                                   test_size=test_size,
                                                   random_state=42)
    X_train[target_name] = y_train
    return X_train, X_test


def make_data(source):
    """
    Utility function to assist in loading different 
    sample datasets. Returns training data (that 
    contains the target) and testing data (that
    does not contain the target).
    
    Parameters
    ----------
    source: string, optional (default="boston")
        The specific dataset to load. Options:
        - Regression: "boston", "diabetes"
        - Classification: "cancer", "digits", "iris", "titanic"
    """
    if source == "boston":
        data = datasets.load_boston()
    elif source == "diabetes":
        data = datasets.load_diabetes()
        data["feature_names"] = ["f{}".format(v) 
                                 for v in range(10)]
    elif source == "cancer":
        data = datasets.load_breast_cancer()
    elif source == "digits":
        data = datasets.load_digits()
        data["feature_names"] = ["f{}".format(v) 
                                 for v in range(64)]        
    elif source == "iris":
        data = datasets.load_iris()
    elif source == "titanic":
        train_data_path = "../tests/test_data/titanic/train.csv"
        test_data_path = "../tests/test_data/titanic/test.csv"

        X_train = pd.read_csv(train_data_path)
        X_test = pd.read_csv(test_data_path)
        return X_train, X_test
    elif source == "abalone":
        train_data_path = "../tests/test_data/abalone_age/abalone.data"
        col_names = ["Sex", "Length", "Diameter", "Height", 
                     "Whole_weight", "Shucked_weight", 
                     "Viscera_weight", "Shell_weight", "Rings"]
        X = pd.read_csv(train_data_path, header=None, names=col_names)
        X["Rings"] = (X.Rings >= 9).astype(int)
        return X_to_train_test(X, "Rings")
    elif source == "bank_marketing":
        train_data_path = "../tests/test_data/bank_marketing/bank-full.csv"
        X = pd.read_csv(train_data_path, sep=";")
        return X_to_train_test(X, "y")
    elif source == "car_evaluation":
        train_data_path = "../tests/test_data/car_evaluation/car.data"
        col_names = ["buying", "maint", "doors", 
                     "persons", "lug_boot", "safety", "car_evaluation"]
        X = pd.read_csv(train_data_path, header=None, names=col_names)
        return X_to_train_test(X, "car_evaluation")
    elif source == "income":
        train_data_path = "../tests/test_data/census_income/adult.data"
        col_names = ["age", "workclass", "fnlwgt", 
                     "education", "education-num", 
                     "marital-status", "occupation", 
                     "relationship", "race", "sex",
                     "capital-gain", "capital-loss", 
                     "hours-per-week", "native-country",
                     "income"]
        train = pd.read_csv(train_data_path, skiprows=[0], 
                            header=None, names=col_names)
        test_data_path = "../tests/test_data/census_income/adult.test"
        test = pd.read_csv(test_data_path, skiprows=[0], 
                           header=None, names=col_names)
        X = pd.concat([train,test])
        return X_to_train_test(X, "income")
    elif source == "chess":
        train_data_path = "../tests/test_data/chess/kr-vs-kp.data"
        X = pd.read_csv(train_data_path, header=None)
        return X_to_train_test(X, 36)
    elif source == "mushrooms":
        train_data_path = "../tests/test_data/mushroom/agaricus-lepiota.data"
        X = pd.read_csv(train_data_path, header=None)
        return X_to_train_test(X, 0)
    elif source == "tictactoe":
        train_data_path = "../tests/test_data/tictactoe/tic-tac-toe.data"
        X = pd.read_csv(train_data_path, header=None)
        return X_to_train_test(X, 9)
    elif source == "wine-origin":
        train_data_path = "../tests/test_data/wine_origin/wine.data"
        X = pd.read_csv(train_data_path, header=None)
        return X_to_train_test(X, 0)
    elif source == "wine-quality":
        train_data_path = "../tests/test_data/wine_quality/winequality-white.csv"
        X = pd.read_csv(train_data_path, sep=";")
        X["quality"] = (X.quality > 5).astype(int)
        return X_to_train_test(X, "quality")
    else:
        raise ValueError("Not a valid dataset.")
    X = pd.DataFrame(data=data.data, 
                     columns=data.feature_names)
    y = pd.Series(data=data.target)
    X_train, X_test, y_train, _ = train_test_split(X, 
                                                   y, 
                                                   test_size=.05,
                                                   random_state=42)
    X_train["target"] = y_train
    return X_train, X_test


def is_categorical(x, 
                   max_classes="auto", 
                   strings_are_categorical=True):
    """
    Check if a target variable is a classification
    problem or a regression problem. Returns True if
    classification and False if regression. On failure,
    raises a ValueError.
    
    Parameters
    ----------
    x: array-like
        This should be the target variable. Ideally, 
        you should convert it to be numeric before 
        using this function.
        
    max_classes: int or float, optional (default="auto")
        Determines the max number of unique values
        there can be for it being a categorical variable
        
        If "auto" - sets it equal to 10% of the dataset or
            100, whichever is smaller
        If float - interprets as percent of dataset size
        If int - interprets as number of classes
        
    strings_are_categorical: bool, optional (default=True)
        If a variable is a string and cannot be coerced
        to a number, returns True regardless of the number
        of unique values. 
    """
    x = pd.Series(x)
    n = len(x)
    n_unique = len(x.unique())
    if max_classes == "auto":
        auto_n_classes = .05
        n_max_classes = int(n*auto_n_classes)
        max_classes = min(n_max_classes, 100)
    if isinstance(max_classes, float):
        n_max_classes = int(n*max_classes)
        max_classes = min(n_max_classes, int(n/2))
    # If x is numeric
    if x.dtype.kind in 'bifc':
        # If there are more than max_classes
        # classify as a regression problem
        if n_unique > max_classes:
            return False
        # If there are floating point numbers
        # classify as a regression problem
        decimals = (x - x.astype(int)).mean()
        if decimals > .01:
            return False
    if n_unique <= max_classes:
        return True
    try:
        x.astype(float)
        return False
    except ValueError:
        if strings_are_categorical:
            return True
        msg = ("Malformed data. "
               "Variable is non-numeric "
               "and there are more "
               "unique values than allowed "
               "by max_classes")
        raise ValueError(msg)
        
        
def categorical_columns(X):
    """Returns a list of all categorical columns"""
    cats = X.apply(is_categorical, axis=0)
    categoricals = cats[cats].index.tolist()
    return categoricals