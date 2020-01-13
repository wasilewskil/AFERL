import unittest
import numpy as np
import aferl.transformations as t
from aferl.dataset import Dataset
from math import log, sqrt

class TestTransformations(unittest.TestCase):
    def test_square(self):
        X = [[1, 0.5, 123],
        [1, 0.5, 123],
        [1, 2, 123],
        [1, 0.5, 123],
        [3, 0.5, 123]]
        y = [1, 2, 1, 1, 2]
        data_info = ['numeric', 'numeric', 'nominal']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationSquare()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[1, 0.5, 123, 1.0, 0.25],
        [1.0, 0.5, 123.0, 1.0, 0.25],
        [1.0, 2.0, 123.0, 1.0, 4.],
        [1.0, 0.5, 123.0, 1.0, 0.25],
        [3.0, 0.5, 123.0, 9.0, 0.25]])
        np.testing.assert_equal(d.X, expected_X)
        np.testing.assert_equal(d.data_info, ['numeric']*2 + ['nominal']+ ['numeric']*2)

        d, _ = tc.transform(dataset)
        expected_X = np.array([[1, 0.5, 123, 1.0, 0.25],
        [1.0, 0.5, 123.0, 1.0, 0.25],
        [1.0, 2.0, 123.0, 1.0, 4.],
        [1.0, 0.5, 123.0, 1.0, 0.25],
        [3.0, 0.5, 123.0, 9.0, 0.25]])
        np.testing.assert_equal(d.X, expected_X)
        np.testing.assert_equal(d.data_info, ['numeric']*2 + ['nominal']+ ['numeric']*2)

        self.assertNotEqual(transformation, tc)

    def test_logarithm(self):
        X = [[-1, 0, 1], [-3, 1, 2]]
        y = [1, 2]
        data_info = ['numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationLogarithm()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, 1, log(3), log(1)], [-3, 1, 2, log(1), log(2)]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*5)

        d, _ = tc.transform(dataset)
        expected_X = np.array([[-1, 0, 1, log(3), log(1)], [-3, 1, 2, log(1), log(2)]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*5)

        self.assertNotEqual(transformation, tc)

    def test_sqrt(self):
        X = [[-1, 0, 1], [-3, 1, 2]]
        y = [1, 2]
        data_info = ['numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationSqrt()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, 1, sqrt(2), sqrt(1)], [-3, 1, 2, sqrt(0), sqrt(2)]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*5) 

        d, _ = tc.transform(dataset)
        expected_X = np.array([[-1, 0, 1, sqrt(2), sqrt(1)], [-3, 1, 2, sqrt(0), sqrt(2)]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*5)  

        self.assertNotEqual(transformation, tc)

    def test_minmax(self):
        X = [[-1, 0, 1, 5, 0, -3], [-3, 1, 2, 5, 0, 0], [1, 3, 2, 5, 0, 3]]
        y = [1, 2, 3]
        data_info = ['numeric', 'nominal', 'nominal', 'numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationMinMax()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, 1, 5, 0, -3, 0.5, 0], [-3, 1, 2, 5, 0, 0, 0, 0.5], [1, 3, 2, 5, 0, 3, 1, 1]])
        np.testing.assert_equal(d.X, expected_X)  
        np.testing.assert_equal(d.data_info, ['numeric', 'nominal', 'nominal', 'numeric', 'numeric', 'numeric'] + ['numeric'] * 2)

        X = [[-1, 0, 1, 5, 0, -1.5], [-3, 1, 2, 5, 0, 0], [1, 3, 2, 5, 0, 1.5]]
        y = [1, 2, 3]
        data_info = ['numeric', 'nominal', 'nominal', 'numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        d, _ = tc.transform(dataset)
        expected_X = np.array([[-1, 0, 1, 5, 0, -1.5, 0.5, 0.25], [-3, 1, 2, 5, 0, 0, 0, 0.5], [1, 3, 2, 5, 0, 1.5, 1, 0.75]])
        np.testing.assert_equal(d.X, expected_X)  
        np.testing.assert_equal(d.data_info, ['numeric', 'nominal', 'nominal', 'numeric', 'numeric', 'numeric'] + ['numeric'] * 2)

        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.scaler is None)
        self.assertTrue(tc.scaler is not None)

    def test_zscore(self):
        X = [[-1, 0, 1, 5, 0, -3], [-3, 1, 2, 5, 0, 0], [1, 3, 2, 5, 0, 3]]
        y = [1, 2, 3]
        data_info = ['numeric', 'nominal', 'nominal', 'numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationMinMax()
        d, tc = transformation.transform(dataset)
        self.assertFalse(d.contains_missing_values())
        np.testing.assert_equal(d.data_info, ['numeric', 'nominal', 'nominal'] + ['numeric'] * 5)

        d, _ = tc.transform(dataset)
        self.assertFalse(d.contains_missing_values())
        np.testing.assert_equal(d.data_info, ['numeric', 'nominal', 'nominal'] + ['numeric'] * 5)
        
        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.scaler is None)
        self.assertTrue(tc.scaler is not None)

    def test_div(self):
        X = [[-1, 0.5, 0], [-3, 1, 1], [1, 3, 1]]
        y = [1, 2, 3]
        data_info = ['numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationDiv()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0.5, 0, -2, -0.5, 0, 0], [-3, 1, 1, -3, -1/3, -1/3, 1], [1, 3, 1, 1/3, 3, 1, 1/3]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*7) 

        d, _ = transformation.transform(dataset)
        expected_X = np.array([[-1, 0.5, 0, -2, -0.5, 0, 0], [-3, 1, 1, -3, -1/3, -1/3, 1], [1, 3, 1, 1/3, 3, 1, 1/3]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*7) 
        self.assertNotEqual(transformation, tc)

    def test_sum(self):
        X = [[-1, 0], [-3, 1], [1, 3]]
        y = [1, 2, 3]
        data_info = ['numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationSum()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, -1], [-3, 1, -2], [1, 3, 4]])
        np.testing.assert_equal(d.X, expected_X)  
        np.testing.assert_equal(d.data_info, ['numeric']*3)

        d, _ = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, -1], [-3, 1, -2], [1, 3, 4]])
        np.testing.assert_equal(d.X, expected_X)  
        np.testing.assert_equal(d.data_info, ['numeric']*3) 
        self.assertNotEqual(transformation, tc)

    def test_diff(self):
        X = [[-1, 0], [-3, 1], [1, 3]]
        y = [1, 2, 3]
        data_info = ['numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationDiff()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, -1], [-3, 1, -4], [1, 3, -2]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*3) 

        d, _ = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, -1], [-3, 1, -4], [1, 3, -2]])
        np.testing.assert_equal(d.X, expected_X) 
        np.testing.assert_equal(d.data_info, ['numeric']*3) 
        self.assertNotEqual(transformation, tc)

    def test_mul(self):
        X = [[-1, 0], [-3, 1], [1, 3]]
        y = [1, 2, 3]
        data_info = ['numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationMul()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, 0], [-3, 1, -3], [1, 3, 3]])
        np.testing.assert_equal(d.X, expected_X)   
        np.testing.assert_equal(d.data_info, ['numeric']*3)

        d, _ = transformation.transform(dataset)
        expected_X = np.array([[-1, 0, 0], [-3, 1, -3], [1, 3, 3]])
        np.testing.assert_equal(d.X, expected_X)   
        np.testing.assert_equal(d.data_info, ['numeric']*3) 
        self.assertNotEqual(transformation, tc)

    def test_aggregation(self):
        X = [[0, -1], [-3, 1], [1, 3]]
        y = [1, 2, 3]
        data_info = ['numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationAggregation()
        d, tc = transformation.transform(dataset)
        expected_X = np.array([[0, -1, -1, 0, -0.5, 0.5], [-3, 1, -3, 1, -1, 2], [1, 3, 1, 3, 2, 1]])
        np.testing.assert_equal(d.X, expected_X)   
        np.testing.assert_equal(d.data_info, ['numeric']*6) 

        d, _ = transformation.transform(dataset)
        expected_X = np.array([[0, -1, -1, 0, -0.5, 0.5], [-3, 1, -3, 1, -1, 2], [1, 3, 1, 3, 2, 1]])
        np.testing.assert_equal(d.X, expected_X)   
        np.testing.assert_equal(d.data_info, ['numeric']*6) 

        self.assertNotEqual(transformation, tc)

    def test_ohe(self):
        X = [[1, 0.5, 'red'],
        [1, 0.5, 'blue'],
        [1, 1, 'red'],
        ['?', 0.5, 'red'],
        [2, 0.5, 'red']]
        y = [1, 2, 1, 1, 2]
        data_info = ['numeric', 'numeric', 'nominal']

        dataset = Dataset(X, y, data_info, '?', '%Y-%m-%d', True)
        transformation = t.TransformationOneHotEncoder()
        d, tc = transformation.transform(dataset)

        expected_X = [[1, 0.5, 1, 0],
        [1, 0.5, 0, 1],
        [1, 1, 1, 0],
        [0, 0.5, 1, 0],
        [2, 0.5, 1, 0]]

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal', 'boolean'])
        np.testing.assert_equal(d.X, expected_X) 

        X = [[1, 0.5, 'red'],
        [1, 0.5, 'red'],
        [1, 1, 'blue'],
        ['?', 0.5, 'red'],
        [2, 0.5, 'red']]
        y = [1, 2, 1, 1, 2]
        data_info = ['numeric', 'numeric', 'nominal']

        dataset = Dataset(X, y, data_info, '?', '%Y-%m-%d', True)
        d, _ = tc.transform(dataset)

        expected_X = [[1, 0.5, 1, 0],
        [1, 0.5, 1, 0],
        [1, 1, 0, 1],
        [0, 0.5, 1, 0],
        [2, 0.5, 1, 0]]

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal', 'boolean'])
        np.testing.assert_equal(d.X, expected_X) 

        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.encoder is None)
        self.assertTrue(tc.encoder is not None)

        

    def test_knn(self):
        X = [[1, 0.5, 'red', '2012-12-12'],
        [1, 0.5, 'blue', '2012-12-12'],
        [1, '?', 'red', '2013-11-23'],
        [1, 0.5, '?', '2012-01-11'],
        ['?', 0.5, 'red', '2000-04-15']]
        y = [1, 2, 1, 1, 2]
        data_info = ['numeric', 'numeric', 'nominal', 'date']

        dataset = Dataset(X, y, data_info, '?', '%Y-%m-%d', True)
        transformation = t.TransformationImputeKnn()
        d, tc = transformation.transform(dataset)

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal', 'timestamp'])
        self.assertFalse(d.contains_missing_values())

        d, _ = tc.transform(dataset)

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal', 'timestamp'])
        self.assertFalse(d.contains_missing_values())
        
        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.imputer is None)
        self.assertTrue(tc.imputer is not None)


    def test_iteimp(self):
        X = [[1, 0.5, 'red', '2012-12-12'],
        [1, 0.5, 'blue', '2012-12-12'],
        [1, '?', 'red', '2013-11-23'],
        [1, 0.5, '?', '2012-01-11'],
        ['?', 0.5, 'red', '2000-04-15']]
        y = [1, 2, 1, 1, 2]
        data_info = ['numeric', 'numeric', 'nominal', 'date']

        dataset = Dataset(X, y, data_info, '?', '%Y-%m-%d', True)
        transformation = t.TransformationImputeIterative()
        d, tc = transformation.transform(dataset)

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal', 'timestamp'])
        self.assertFalse(d.contains_missing_values())

        d, _ = tc.transform(dataset)

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal', 'timestamp'])
        self.assertFalse(d.contains_missing_values())
        
        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.imputer is None)
        self.assertTrue(tc.imputer is not None)

    def test_impsim(self):
        X = [[1, 1, 'red'],
        [2, 2, 'blue'],
        [3, '?', 'red'],
        [5, 5, '?'],
        [5, 5, 'blue'],
        ['?', 2, 'red']]
        y = [1, 2, 1, 1, 2]
        data_info = ['numeric', 'numeric', 'nominal']

        dataset = Dataset(X, y, data_info, '?', '%Y-%m-%d', True)
        transformation = t.TransformationImputeSimple()
        d, tc = transformation.transform(dataset)

        expected_X = [[1, 1, 1],
        [2, 2, 0],
        [3, 2, 1],
        [5, 5, 1],
        [5, 5, 0],
        [3, 2, 1]]

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal'])
        self.assertFalse(d.contains_missing_values())
        np.testing.assert_equal(d.X, expected_X) 

        d, _ = tc.transform(dataset)

        self.assertEqual(d.data_info, ['numeric', 'numeric', 'nominal'])
        self.assertFalse(d.contains_missing_values())
        
        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.imputer is None)
        self.assertTrue(tc.imputer is not None)

    def test_select_percentile(self):
        X = [[-1, 1, 1, 1, -1], [-3, 1, 2, 1, -3], [1, 2, 3, 1, 1]]
        y = [1, 2, 3]
        data_info = ['numeric', 'numeric', 'numeric', 'numeric', 'numeric']
        dataset = Dataset(X, y, data_info)

        transformation = t.TransformationSelectPercentile(25)
        d, tc = transformation.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric']) 

        self.assertNotEqual(transformation, tc)

        d, _ = tc.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric']) 

        transformation = t.TransformationSelectPercentile(50)
        d, tc = transformation.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric', 'numeric']) 

        self.assertNotEqual(transformation, tc)

        d, _ = tc.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric', 'numeric']) 

        transformation = t.TransformationSelectPercentile(75)
        d, tc = transformation.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric', 'numeric', 'numeric']) 

        self.assertNotEqual(transformation, tc)

        d, _ = tc.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric', 'numeric', 'numeric'])
        

    def test_binning_nominal(self):
        X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
        y = [1, 2, 3]
        data_info = ['numeric']
        dataset = Dataset(X, y, data_info)
        
        expected_X = [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5], [7, 6], [8, 7], [9, 8], [10, 9]]

        transformation = t.TransformationBinningNominal()
        d, tc = transformation.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric'] + ['nominal'])
        np.testing.assert_equal(d.X, expected_X)

        expected_X = [[10, 9], [10, 9], [3, 2], [3, 2], [4, 3], [4, 3], [5, 4], [5, 4], [6, 5], [6, 5]]

        X = [[10], [10], [3], [3], [4], [4], [5], [5], [6], [6]]
        y = [1, 2, 3]
        data_info = ['numeric']
        dataset = Dataset(X, y, data_info)

        d, _ = tc.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric'] + ['nominal'])
        np.testing.assert_equal(d.X, expected_X)

        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.binner is None)
        self.assertTrue(tc.binner is not None)
         

    def test_binning_mean(self):
        X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20]]
        y = [1, 2, 3]
        data_info = ['numeric']
        dataset = Dataset(X, y, data_info)

        expected_X = [[1, 1.95], [2, 1.95], [3, 3.85], [4, 3.85], [5, 5.75], [6, 5.75], [7, 7.65], [8, 7.65], [9, 9.55], [10, 9.55], [11, 11.45], [12, 11.45], [13, 13.35], [14, 13.35], [15, 15.25], [16, 15.25], [17, 17.15], [18, 17.15], [19, 19.05], [20, 19.05]]

        transformation = t.TransformationBinningMean()
        d, tc = transformation.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric'] + ['numeric']) 
        np.testing.assert_allclose(d.X, expected_X)

        X = [[5], [4], [3], [2], [1]]
        y = [1, 2, 3]
        data_info = ['numeric']
        dataset = Dataset(X, y, data_info)

        expected_X = [[5, 5.75], [4, 3.85], [3, 3.85], [2, 1.95], [1, 1.95]]

        d, _ = tc.transform(dataset)
        np.testing.assert_equal(d.data_info, ['numeric'] + ['numeric']) 
        np.testing.assert_allclose(d.X, expected_X)

        self.assertNotEqual(transformation, tc)
        self.assertTrue(transformation.binner is None)
        self.assertTrue(tc.binner is not None)

    def test_chain(self):
        X = [[1, 1, 1],
        [2, 2, 0],
        [3, 2, 1],
        [5, 5, 1],
        [5, 5, 0],
        [3, 2, 1]]
        y = [1, 2, 1, 1, 2, 3]
        data_info = ['numeric', 'numeric', 'numeric']
        d = Dataset(X, y, data_info)

        t1 = t.TransformationMul()
        t2 = t.TransformationSelectPercentile(50)
        t3 = t.TransformationMinMax()
        t4 = t.TransformationSelectPercentile(20)
        t5 = t.TransformationBinningNominal()

        d1, tc1 = t1.transform(d)
        d2, tc2 = t2.transform(d1)
        d3, tc3 = t3.transform(d2)
        d4, tc4 = t4.transform(d3)
        d5, tc5 = t5.transform(d4)

        d11, _ = tc1.transform(d)
        d22, _ = tc2.transform(d11)
        d33, _ = tc3.transform(d22)
        d44, _ = tc4.transform(d33)
        d55, _ = tc5.transform(d44)

        np.testing.assert_equal(d5.X, d55.X)
        np.testing.assert_equal(d5.data_info, d55.data_info)


if __name__ == '__main__':
    unittest.main()