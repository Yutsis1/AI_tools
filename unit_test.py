import unittest
from train import x_train
from test import x_test, y_pred
from test import f1


class Test(unittest.TestCase):
    def test_len_pred_test_equal(self):
        self.assertEqual(len(x_test), len(y_pred))

    def test_features_len_equal(self):
        self.assertEqual(x_test.shape[1], x_train.shape[1])

    def test_metric_score(self):
        self.assertGreater(f1, 0.5)


if __name__ == '__main__':
    unittest.main()
