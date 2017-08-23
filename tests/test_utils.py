import unittest
from learn import utils

class TestUtils(unittest.TestCase):
    def test_making_data_simple(self):
        for data in ["boston", "iris"]:
            X_train, X_test = utils.make_data(source=data)
            train_cols = X_train.columns
            test_cols = X_test.columns
            # Training data should have exactly one additional column
            self.assertEqual(len(train_cols), len(test_cols)+1)
            # Ensure only one column name is different
            n_diff_cols = len(set(X_train.columns) - set(X_test.columns))
            self.assertEqual(1, n_diff_cols)
        
    def test_is_classification_problem(self):
        # Shorten function name
        icp = utils.is_categorical
        # Regression because floats
        result = icp([1.1, 2.1])
        self.assertEqual(result, 0)
        # Regression because number of unique
        result = icp([1,2,3,4])
        self.assertEqual(result, 0)
        # Classification because words
        result = icp(["cat"]*20+["dog"]*20)
        self.assertEqual(result, 1)
        # Classification because number of uniques
        result = icp([0]*20+[1]*20)
        self.assertEqual(result, 1)
        # Real data tests - Regression
        for dataset in ["boston", "diabetes"]:
            data = utils.make_data(source=dataset)
            X, y = utils.X_y_split(*data)
            self.assertEqual(icp(y), 0)
        # Real data tests - Classification
        for dataset in ["cancer", "digits", "iris"]:
            data = utils.make_data(source=dataset)
            X, y = utils.X_y_split(*data)
            self.assertEqual(icp(y), 1)
            
# class TestXYSplit(unittest.TestCase):
#     pass

if __name__ == '__main__':
    unittest.main()