import unittest
import numpy as np
from aferl.dataset import Dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.X = [[1, 0.5, 'red', '2012-12-12'],
        [1, 0.5, 'blue', '2012-12-12'],
        [1, '?', 'red', '2013-11-23'],
        [1, 0.5, '?', '2012-01-11'],
        ['?', 0.5, 'red', '2000-04-15']]
        self.y = [1, 2, 1, 1, 2]
        self.data_info = ['numeric', 'numeric', 'nominal', 'date']

    def test_init(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        expected_X = np.array([[1.0, 0.5, 1.0, 1355266800.0],
        [1.0, 0.5, 0.0, 1355266800.0],
        [1.0, 0, 1.0, 1385161200.0],
        [1.0, 0.5, 0, 1326236400.0],
        [0, 0.5, 1.0, 955749600.0]])
        expected_data_info = ['numeric', 'numeric', 'nominal', 'timestamp']

        np.testing.assert_equal(d.X, expected_X)
        self.assertEqual(d.data_info, expected_data_info)
        self.assertEqual(d.X.dtype, 'float64')

    def test_init_2(self):  
        X = [[1, 0.5, 'red', '2012-12-12', -1],
        [1, 0.5, 'blue', '2012-12-12', 1],
        [1, '?', 'red', '2013-11-23', -1],
        [1, 0.5, '?', '2012-01-11', 1],
        ['?', 0.5, 'red', '2000-04-15', '?']]  
        data_info = ['numeric', 'numeric', 'nominal', 'date', 'nominal']  
        d = Dataset(X, self.y, data_info, '?', '%Y-%m-%d', True)

        expected_X = np.array([[1.0, 0.5, 1.0, 1355266800.0, 0],
        [1.0, 0.5, 0.0, 1355266800.0, 1],
        [1.0, 0, 1.0, 1385161200.0, 0],
        [1.0, 0.5, 0, 1326236400.0, 1],
        [0, 0.5, 1.0, 955749600.0, 0]])
        expected_data_info = ['numeric', 'numeric', 'nominal', 'timestamp', 'nominal']
        expected_missing_values = (np.array([2, 3, 4, 4]), np.array([1, 2, 0, 4]))

        np.testing.assert_equal(d.X, expected_X)
        np.testing.assert_equal(d.missing_values, expected_missing_values)
        self.assertEqual(d.data_info, expected_data_info)
        self.assertEqual(d.X.dtype, 'float64')

    def test_contains_missing_value(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)
        self.assertEqual(d.contains_missing_values(), True)

        d.mark_as_imputed()
        self.assertEqual(d.contains_missing_values(), False)

    def test_filter_data(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        actual = d.filter_data(['numeric', 'timestamp'])
        expected = np.array([[1.0, 0.5, 1355266800.0],
        [1.0, 0.5, 1355266800.0],
        [1.0, 0, 1385161200.0],
        [1.0, 0.5, 1326236400.0],
        [0, 0.5, 955749600.0]])

        expected_X = np.array([[1.0, 0.5, 1.0, 1355266800.0],
        [1.0, 0.5, 0.0, 1355266800.0],
        [1.0, 0, 1.0, 1385161200.0],
        [1.0, 0.5, 0, 1326236400.0],
        [0, 0.5, 1.0, 955749600.0]])
        expected_data_info = ['numeric', 'numeric', 'nominal', 'timestamp']

        np.testing.assert_equal(actual, expected)
        np.testing.assert_equal(d.X, expected_X)
        np.testing.assert_equal(d.data_info, expected_data_info)
        

    def test_remove_type(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        d.remove_type('nominal')
        expected_X = np.array([[1.0, 0.5, 1355266800.0],
        [1.0, 0.5, 1355266800.0],
        [1.0, 0, 1385161200.0],
        [1.0, 0.5, 1326236400.0],
        [0, 0.5, 955749600.0]])
        expected_data_info = ['numeric', 'numeric', 'timestamp']

        np.testing.assert_equal(d.X, expected_X)
        np.testing.assert_equal(d.data_info, expected_data_info)

    def test_remove_column(self):
        X = [[1, 0.5, 'red', '2012-12-12'],
        [1, 0.5, 'blue', '2012-12-12'],
        [1, '?', 'red', '?'],
        [1, 0.5, '?', '2012-01-11'],
        ['?', 0.5, 'red', '2000-04-15']]
        d = Dataset(X, self.y, self.data_info, '?', '%Y-%m-%d', True)
        d.remove_columns([0,2])

        expected_X = [[0.5, 1355266800.0],
        [0.5, 1355266800.0],
        [0, 0],
        [0.5, 1326236400.0],
        [0.5, 955749600.0]]
        expected_data_info = ['numeric', 'timestamp']
        expected_missing_values = (np.array([2, 2]), np.array([0,1])) 

        np.testing.assert_equal(d.X, expected_X)
        np.testing.assert_equal(d.data_info, expected_data_info)
        np.testing.assert_equal(d.missing_values, expected_missing_values)


    # def test_remove_type_preserve_nan(self):        
    #     d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)
    #     d.X[2, 1] = 2
    #     d.remove_type('numeric', True)

    #     expected_X = np.array([[1.0, 2.0, 1355266800.0],
    #     [1.0, 1.0, 1355266800.0],
    #     [1.0, 2.0, 1385161200.0],
    #     [1.0, 0, 1326236400.0],
    #     [0, 2.0, 955749600.0]])
    #     expected_data_info = ['numeric', 'nominal', 'timestamp']

    #     np.testing.assert_equal(d.X, expected_X)
    #     np.testing.assert_equal(d.data_info, expected_data_info)

    def test_get_type_indexes(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        type_indexes = d.get_type_indexes('numeric')
        expected_indexes = [0, 1]

        np.testing.assert_equal(type_indexes, expected_indexes)

    def test_contains_type(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        self.assertEqual(d.contains_type('numeric'), True)
        self.assertEqual(d.contains_type('string'), False)

        d.remove_type('numeric')
        self.assertEqual(d.contains_type('numeric'), False)

    def test_copy(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        d_copy = d.copy()

        np.testing.assert_equal(d.X, d_copy.X)
        np.testing.assert_equal(d.data_info, d_copy.data_info)
        np.testing.assert_equal(d.missing_values, d_copy.missing_values)
        self.assertEqual(d == d_copy, False)

    def test_extend(self):        
        d = Dataset(self.X, self.y, self.data_info, '?', '%Y-%m-%d', True)

        d.extend(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), ['numeric', 'numeric', 'aaa'])

        expected_X = np.array([[1.0, 0.5, 1.0, 1355266800.0, 1, 2],
        [1.0, 0.5, 0.0, 1355266800.0, 3, 4],
        [1.0, 0, 1.0, 1385161200.0, 5, 6],
        [1.0, 0.5, 0, 1326236400.0, 7, 8],
        [0, 0.5, 1.0, 955749600.0, 9, 10]])
        expected_data_info = ['numeric', 'numeric', 'nominal', 'timestamp', 'numeric', 'numeric', 'aaa']

        np.testing.assert_equal(d.X, expected_X)
        self.assertEqual(d.data_info, expected_data_info)
        self.assertEqual(d.X.dtype, 'float64')
        self.assertEqual(d.contains_type('aaa'), True)


if __name__ == '__main__':
    unittest.main()