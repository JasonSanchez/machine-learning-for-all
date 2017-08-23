
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_auc_score
from sklearn import metrics
from learn import utils

def categorical_unique_counts(X):
    """
    Returns series of categorical columns
    and count of unique values in each 
    column
    """
    cats = utils.categorical_columns(X)
    return X[cats].apply(pd.Series.nunique, axis=0)

def small_categorical(X, large_class_threshold=10):
    counts = categorical_unique_counts(X)
    mask = counts < large_class_threshold
    return counts[mask].index.tolist()

def large_categorical(X, large_class_threshold=10):
    counts = categorical_unique_counts(X)
    mask = counts>=large_class_threshold
    return counts[mask].index.tolist()

def word_to_num(word, max_char=5):
    """
    Assigns a number to a word that
    is the approximate sort order of
    the word
    
    Words with the same first max_char
    will have the same value.
    """
    word_val = 0
    for n, char in enumerate(str(word)):
        if n > max_char:
            break
        num = ord(char)/130
        den = 10**n
        total = num/den
        word_val += total
    return word_val

def word_size(word):
    """
    Returns the length of the word
    """
    return len(str(word))

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """
    Adds a new "NULL" category for missing values
    """
    def __init__(self, fill_value="NULL"):
        self.fill_value = fill_value
    
    def fit(self, X, y=None):
        self.cat_cols = utils.categorical_columns(X)
        return self
    
    def transform(self, X, y=None):
        fill_values = {c:self.fill_value for c in self.cat_cols}
        return X.fillna(fill_values, axis=0)

class NumericImputer(BaseEstimator, TransformerMixin):
    """
    TODO: Add option for indicator variable if NaN
    """
    def __init__(self, method="mean"):
        self.method = method
        
    def fit(self, X, y=None):
        if self.method == "mean":
            self.fill_values = X.mean()
        if self.method == "max":
            self.fill_values = X.max() + 1
        return self
    
    def transform(self, X, y=None):
        cols = (X.dtypes[X.dtypes =="object"]).index
        if len(cols):
            print(X[cols].describe())
        X[~pd.np.isfinite(X)] = 0 #TODO: Fix
        return X.fillna(self.fill_values)

    
class KeepNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.numeric_columns = X.dtypes[X.dtypes != "object"].index.tolist()
        return self
        
    def transform(self, X, y=None):
        return X[self.numeric_columns]
    
    
class Categoricals(BaseEstimator, TransformerMixin):
    def __init__(self, large_class_threshold=10):
        """
        Anything large_class_threshold and larger
        will be treated as a categorical features
        with a large number of categories.
        """
        self.large_class_threshold = large_class_threshold
        
    def fit(self, X, y=None):
        lct = self.large_class_threshold
        self.small = small_categorical(X, lct)
        self.large = large_categorical(X, lct)
        self.all = self.small + self.large
        # Save category unique value counts for feature
        # engineering
        self.value_counts = defaultdict(int)
        for col in self.large:
            self.value_counts[col] = X[col].value_counts()
        return self
    
    def transform(self, X, y=None):
        # Add sort value based features
        for col in self.all:
            X[col] = X[col].asobject
            new_col = X[col].apply(word_to_num)
            col_name = str(col)+"__sort"
            X[col_name] = new_col
        
        for col in self.large:
            # Add count based features
            counts = self.value_counts[col]
            X = X.join(counts, 
                       on=col, 
                       rsuffix="__counts")
            # Add word length features
            new_col = X[col].apply(word_size)
            col_name = str(col)+"__length"
            X[col_name] = new_col
        return X

    
class Standardize(BaseException, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self
    
    def transform(self, X, y=None):
        return (X - self.mean)/self.std
    

class DropBadColumns(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Drops columns with:
        * NaN standard deviation
        * Zero standard deviation
        """
        pass
    
    def fit(self, X, y=None):
        std = X.std(axis=0)
        null_std = std.isnull()
        zero_std = std == 0
        bad_std_cols = std[null_std | zero_std].index.values.tolist()
        self.to_drop = bad_std_cols
        return self
    
    def transform(self, X):
        return X.drop(self.to_drop, axis=1)
    
    
def regression_metrics(y, y_hat):
    exp_var = metrics.explained_variance_score(y, y_hat)
    mae = metrics.mean_absolute_error(y, y_hat)
    mse = metrics.mean_squared_error(y, y_hat)
    medae = metrics.median_absolute_error(y, y_hat)
    r2 = metrics.r2_score(y, y_hat)
    results = {
        "Explained variance score": exp_var,
        "Mean absolute error": mae,
        "Mean squared error": mse,
        "Root mean squared error": mse**.5,
        "Median absolute error": medae,
        "R^2 score": r2
    }
    return results


class RegressionPredict(BaseEstimator):
    def __init__(self, time_to_compute=100):
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        self.lr = RidgeCV()
        self.lr.fit(X, y)
        lr_pred = cross_val_predict(self.lr, X, y, cv=10, n_jobs=-1).reshape(-1, 1)
        
        self.rf = RandomForestRegressor(n_estimators=self.time_to_compute, 
                                        random_state=42, 
                                        oob_score=True, 
                                        n_jobs=-1)
        self.rf.fit(X, y)
        rf_pred = self.rf.oob_prediction_.reshape(-1, 1)

        layer_1 = np.hstack([
            lr_pred, 
            rf_pred
        ])

        self.lr_1 = LinearRegression()
        self.generalized_predictions = cross_val_predict(self.lr_1, 
                                                         layer_1, 
                                                         y, 
                                                         cv=10, 
                                                         n_jobs=-1, 
                                                         method="predict")
        self.lr_1.fit(layer_1, y)
        return self
    
    def predict(self, X):
        lr_pred = self.lr.predict(X).reshape(-1, 1)
        rf_pred = self.rf.predict(X).reshape(-1, 1)
        layer_1 = np.hstack([
            lr_pred, 
            rf_pred
        ])
        final_predictions = self.lr_1.predict(layer_1)
        return self.rf.predict(X)
        return final_predictions
    

class Regression(BaseEstimator, RegressorMixin):
    def __init__(self, time_to_compute=100):
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        self.model = RegressionPredict(time_to_compute=self.time_to_compute)
        self.model.fit(X, y)
        self.oob_predictions = self.model.generalized_predictions
        
        self.all_metrics = regression_metrics(y, self.oob_predictions)
        self.score_type = "R^2*100"
        self.score = int(self.all_metrics["R^2 score"]*100)
        self.display_score = "%d/100" % self.score
        self.understandable_metric_name = "Average prediction error"
        self.understandable_metric_value = self.all_metrics["Mean absolute error"]
        self.understandable_metric_description = "On average, the predictions will be off by %.2f." % self.understandable_metric_value
        return self
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    

def classification_metrics(y, y_hat):
    results = {}
    y_prob = y_hat[:, 1]
    y_pred = (y_prob > .5).astype(int)
    y_bin = label_binarize(y, 
                           sorted(pd.Series(y).unique()))
    binary = y_bin.shape[1] == 1
    if binary:
        # Fix the binary case returning a column vector
        y_bin = np.hstack((-(y_bin - 1), y_bin))
    ave_precision = metrics.average_precision_score(y_bin, y_hat)
    auc = metrics.roc_auc_score(y_bin, y_hat)
    log_loss = metrics.log_loss(y_bin, y_hat)
    data = {
        "Accuracy": (y_hat.argmax(axis=1) == y).mean(),
        "Average precision score": ave_precision,
        "AUC": auc,
        "Log loss (cross-entropy loss)": log_loss
            }
    if binary:
        brier = metrics.brier_score_loss(y, y_prob)
        f1 = metrics.f1_score(y, y_pred)
        cks = metrics.cohen_kappa_score(y, y_pred)
        hamming = metrics.hamming_loss(y, y_pred)
        hinge = metrics.hinge_loss(y, y_pred)
        jaccard = metrics.jaccard_similarity_score(y, y_pred)
        matt = metrics.matthews_corrcoef(y, y_pred)
        precision = metrics.precision_score(y, y_pred)
        recall = metrics.recall_score(y, y_pred)
        binary_data = {
            "Brier score loss": brier,
            "F1 score": f1,
            "Cohen's kappa": cks,
            "Average Hamming loss": hamming,
            "Hinge loss": hinge,
            "Jaccard similarity coefficient": jaccard,
            "Matthews correlation coefficient": matt,
            "Precision": precision,
            "Recall": recall
            }
        data.update(binary_data)
    return data


class ClassificationPredict(BaseEstimator):
    def __init__(self, time_to_compute=100):
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        smallest_class_size = pd.Series(y).value_counts().min()
        cv = min(smallest_class_size, 10)
        if cv == 1:
            raise ValueError("One of the classes you are trying to predict has only one observation!")
        
        self.lr = LogisticRegression(C=1)
        self.lr.fit(X, y)
        lr_pred = cross_val_predict(self.lr, X, y, cv=cv, n_jobs=-1, method="predict_proba")
        
        self.rf = RandomForestClassifier(n_estimators=self.time_to_compute, 
                                         random_state=42, oob_score=True, n_jobs=-1)
        self.rf.fit(X, y)
        rf_pred = self.rf.oob_decision_function_

        layer_1 = np.hstack([
            lr_pred, 
            rf_pred
        ])

        self.lr_1 = LogisticRegression(C=1)
        self.generalized_predictions = cross_val_predict(self.lr_1, 
                                                         layer_1, 
                                                         y, 
                                                         cv=cv, 
                                                         n_jobs=-1, 
                                                         method="predict_proba")
        self.lr_1.fit(layer_1, y)
        return self
    
    def predict_proba(self, X):
        lr_pred = self.lr.predict_proba(X)
        rf_pred = self.rf.predict_proba(X)
        layer_1 = np.hstack([
            lr_pred, 
            rf_pred
        ])
        final_predictions = self.lr_1.predict_proba(layer_1)
        return final_predictions

    def predict(self, X):
        predictions = self.predict_proba(X)
        return predictions.argmax(1)
    

class Classification(BaseEstimator, ClassifierMixin):
    def __init__(self, time_to_compute=100):
        """
        """
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        y = pd.Series(y)
        self.n_classes = len(y.unique())
        self.label_encoder = None
        self.label_encoder = LabelEncoder().fit(y)
        y = self.label_encoder.transform(y)

        self.model = ClassificationPredict(time_to_compute=self.time_to_compute)
        self.model.fit(X, y)
        self.oob_predictions = self.model.generalized_predictions
        
        self.all_metrics = classification_metrics(y, 
                                                  self.oob_predictions)
        self.score_type = "(AUC - .5)*200"
        self.score = int((self.all_metrics["AUC"] - .5)*200)
        self.display_score = "%d/100" % self.score
        self.understandable_metric_name = "Accuracy"
        self.understandable_metric_value = self.all_metrics["Accuracy"]*100
        self.understandable_metric_description = "The predictions are expected to be correct %.2f%% of the time" % self.understandable_metric_value
        return self
        
    def predict(self, X):
        predictions = self.model.predict(X)
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        return predictions

    
class All():
    def __init__(self, time_to_compute=None, force_model=None):
        """
        time_to_compute: higher numbers mean longer compute time 
        and more accurate results
        
        force_model: None, "regression", or "classification"
        Forces the model used
        """
        self.time_to_compute = time_to_compute or 100
        self.force_model = force_model
        
        
    def fit(self, X, y):
        if isinstance(y, pd.Series):
            self.target_name = y.name
        else:
            self.target_name = "what you are trying to predict"
        X.columns = [str(col) for col in X.columns]
        # Determine type of problem
        if self.force_model:
            self.classification = self.force_model == "classification"
        else:
            self.classification = utils.is_categorical(y, max_classes=.1)
        if self.classification:
            model = Classification(time_to_compute=self.time_to_compute)
        else:
            model = Regression(time_to_compute=self.time_to_compute)
        # Create pipeline
        steps = [("categorical_imputation", CategoricalImputer()),
                 ("make_categoricals_numeric", Categoricals()),
                 ("keep_only_numeric", KeepNumeric()),
                 ("numeric_imputation", NumericImputer("max")),
                 ("drop_bad_columns", DropBadColumns()),
                 ("scale", Standardize()),
                 ("model", model)]
        pipe = Pipeline(steps)
        pipe.fit(X, y)
        self.model = pipe
        self.score = pipe.named_steps["model"].score
        self.score_type = pipe.named_steps["model"].score_type
        self.display_score = pipe.named_steps["model"].display_score
        self.all_metrics = pipe.named_steps["model"].all_metrics
        self.understandable_metric_name = pipe.named_steps["model"].understandable_metric_name
        self.understandable_metric_description = pipe.named_steps["model"].understandable_metric_description
        self.understandable_metric_value = pipe.named_steps["model"].understandable_metric_value
        return self
        
    def predict(self, X):
        X.columns = [str(col) for col in X.columns]
        predictions = self.model.predict(X)
        return predictions