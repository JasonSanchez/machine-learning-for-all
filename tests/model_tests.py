import unittest
from learn import utils

class TestUtils(unittest.TestCase):
    def test_making_regression_data_simple(self):
        X_train, X_test = utils.make_regression_data()
        train_cols = X_train.columns
        test_cols = X_test.columns
        self.assertEquals(len(train_cols), len(test_cols)+1)

class TestXYSplit(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()