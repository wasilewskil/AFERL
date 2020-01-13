import random
import operator
import numpy as np
import datetime as dt
import time
import warnings

from aferl.dataset import Dataset
from aferl.utils import can_convert_to_float, RunWithTimeout
from math import log, sqrt, sin, cos, sinh, cosh, tan, tanh
from statistics import median
from sklearn.feature_selection import SelectKBest, SelectFpr, SelectFdr, SelectFwe, SelectPercentile, VarianceThreshold, f_classif
from itertools import combinations
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
warnings.simplefilter("ignore")

class TransformationFactory():
    def __init__(self, allowed_transformations = None):
        self.transformations = []
        self.allowed_transformations = allowed_transformations
        self.create_transformations()

    def create_transformations(self):
        self.transformations = [
            # Feature selection
            TransformationSelectPercentile(75),
            TransformationSelectPercentile(50),
            TransformationSelectPercentile(20),

            # #Imputing
            TransformationImputeKnn(),
            TransformationImputeIterative(), 
            TransformationImputeSimple(),    

            # #Encoding  
            TransformationOneHotEncoder(),

            # # #Binning
            TransformationBinningNominal(),
            TransformationBinningMean(), 

            # # #Normalization 
            TransformationMinMax(),            
            TransformationZscore(),                  
            TransformationSigmoid(),  

            # #Math
            TransformationLogarithm(),
            TransformationSqrt(),      
            TransformationCos(),
            TransformationSin(),
            TransformationSquare(),
            TransformationTanh(),
            TransformationSum(),
            TransformationDiff(),
            TransformationMul(),
            TransformationDiv(),

            # #Other
            TransformationAggregation(),       
        ]
        self.transformations = [t for t in self.transformations if t.name in self.allowed_transformations]
        random.shuffle(self.transformations)


class TransformationBase():
    def __init__(self):           
        self.rewards_sum = 0
        self.no_usages = 0
        self.name = ''
        self.avg_reward = 0
        self.transformer = None
        self.is_copied_instance = False
        self.indexes_duplicates = None

    def get_avg_reward(self):
        return self.avg_reward
    
    def add_usage(self, reward):
        self.rewards_sum = self.rewards_sum + reward
        self.no_usages = self.no_usages + 1
        self.avg_reward = self.rewards_sum / self.no_usages

    def is_feature_selection(self):
        return self.name.endswith('_fs')

    def is_imputing(self):
        return self.name.endswith('_imp')

    def is_encoding(self):
        return self.name.endswith('_enc')

    def transform_simple(self, dataset, function, data_type, function_type = None, result_data_type = None):  
        function_type = 'single' if function_type == None else function_type
        result_data_type = 'numeric' if result_data_type == None else result_data_type

        if data_type == 'numeric':
            filtered = dataset.filter_data(['numeric'])
        elif data_type != 'all':
            filtered = dataset.filter_data([data_type])

        filtered = self.before_transform(filtered)    
        shape = [0, filtered.shape[1]]    
        result = np.empty(shape, dtype='float64')
        
        if filtered.shape[1] > 0:
            if function_type == 'matrix':   
                result = function(filtered)
            else:
                def process_row(row):
                    row_res = []
                    for val in row: 
                        row_res.append(function(val))
                    return row_res
                
                res_list = []
                for row in filtered:
                    res_list.append(process_row(row))
                result = np.vstack((result, res_list))
                
            np.nan_to_num(result)
            dataset.extend(result, [result_data_type]*result.shape[1])
        return dataset

    def transform(self, dataset):
        if self.is_feature_selection() == False:
            #job = RunWithTimeout(self._transform, dataset.copy())
            res = self._transform(dataset.copy())
            if res is None:
                print("Probalby killed or not")
                return None, None
            if self.is_copied_instance == True and self.indexes_duplicates is not None:
                res.remove_columns(self.indexes_duplicates)
            elif self.is_copied_instance == False:
                self.indexes_duplicates = self._remove_duplicates(res)
        else:
            res = self._transform(dataset.copy())
        return res, self.copy()

    def _transform(self, dataset):
        pass

    def _remove_duplicates(self, dataset):
        _, index = np.unique(dataset.X, axis=1, return_index=True)
        mask = np.ones(dataset.X.shape[1], np.bool)
        mask[index] = 0        
        indexes = np.where(mask == True)[0].tolist()
        dataset.remove_columns(indexes)
        return indexes

    def before_transform(self, X):
        return X
    
    def copy(self):
        if self.is_copied_instance:
            return None
        else:
            copy = self._copy()
            copy.indexes_duplicates = self.indexes_duplicates
            copy.is_copied_instance = True
            self.indexes_duplicates = None
            return copy

    def _copy(self):
        pass

class TransformationFeatureSelection(TransformationBase):
    def __init__(self, indexes_duplicates = None, indexes_select = None):
        super().__init__()
        self.indexes_duplicates = indexes_duplicates
        self.indexes_select = indexes_select

    def _transform(self, dataset):
        if self.is_copied_instance == True and (self.indexes_duplicates is not None or self.indexes_select is not None):
            self._remove(dataset)
            return dataset

        self.indexes_duplicates = self._remove_duplicates(dataset)

        selector = self._get_selector()
        selector.fit_transform(dataset.X, dataset.y)
        self.indexes_select = np.where(selector.get_support() == False)[0].tolist()
        if dataset.X.shape[1] > len(self.indexes_select):
            dataset.remove_columns(self.indexes_select)  
        else:
            self.indexes_select = None      
        return dataset

    def _remove(self, dataset):
        if self.indexes_duplicates is not None:
            dataset.remove_columns(self.indexes_duplicates)
        
        if self.indexes_select is not None:
            dataset.remove_columns(self.indexes_select)

    def _get_selector(self):
        pass

    def _copy(self):
        pass

class TransformationSelectPercentile(TransformationFeatureSelection):
    def __init__(self, percentile, indexes_duplicates = None, indexes_select = None):
        super().__init__(indexes_duplicates, indexes_select)
        self.name = 'fsp' + str(percentile) + '_fs'
        self.percentile = percentile

    def _copy(self):
        transformation = TransformationSelectPercentile(self.percentile, self.indexes_duplicates, self.indexes_select)
        self.indexes_select = None
        return transformation

    def _get_selector(self):
        return SelectPercentile(f_classif, self.percentile)

class TransformationLogarithm(TransformationBase):
    def __init__(self, shift = None):
        super().__init__()
        self.name = 'log'
        self.shift = shift

    def before_transform(self, X):
        if self.is_copied_instance == False or self.shift is None:
            self.shift = (X.min(axis=0) - 1).clip(max=0)

        result = X - self.shift
        result[result < 1] = 1
        return result 
   
    def _transform(self, dataset):
        return self.transform_simple(dataset, np.log, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationLogarithm(self.shift)
        self.shift = None
        return t

class TransformationSqrt(TransformationBase):
    def __init__(self, shift = None):
        super().__init__()
        self.name = 'sqrt'
        self.shift = shift

    def before_transform(self, X):
        if self.is_copied_instance == False or self.shift is None:
            self.shift = (X.min(axis=0) - 1).clip(max=0)

        result = X - self.shift
        result[result < 0] = 0    
        return X - X.min(axis=0).clip(max=0)

    def _transform(self, dataset):
        return self.transform_simple(dataset, np.sqrt, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationSqrt(self.shift)
        self.shift = None
        return t

class TransformationSquare(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'square'

    def square(self, matrix):
        return np.power(matrix, 2)

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.square, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationSquare()
        return t


class TransformationSin(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'sin'

    def _transform(self, dataset):
        return self.transform_simple(dataset, np.sin, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationSin()
        return t

class TransformationCos(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'cos'
    
    def _transform(self, dataset):
        return self.transform_simple(dataset, np.cos, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationCos()
        return t

class TransformationTanh(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'tanh'

    def _transform(self, dataset):
        return self.transform_simple(dataset, np.tanh, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationTanh()
        return t


class TransformationAggregation(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'aggregation'
    
    def aggregate(self, matrix):
        result = np.empty((matrix.shape[0], 0), matrix.dtype)
        result = np.append(result, np.min(matrix, axis = 1).reshape(matrix.shape[0], 1), axis = 1)
        result = np.append(result, np.max(matrix, axis = 1).reshape(matrix.shape[0], 1), axis = 1)
        result = np.append(result, np.mean(matrix, axis = 1).reshape(matrix.shape[0], 1), axis = 1)
        result = np.append(result, np.std(matrix, axis = 1).reshape(matrix.shape[0], 1), axis = 1)
        return result

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.aggregate, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationAggregation()
        return t

class TransformationSigmoid(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'sigmoid'

    def sigmoid(self, matrix):
        return (1 / (1 + np.exp(-matrix)))

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.sigmoid, 'numeric', 'matrix')

    def _copy(self):
        t = TransformationSigmoid()
        return t

class TransformationScaling(TransformationBase):
    def __init__(self, scaler = None):
        super().__init__()
        self.scaler = scaler

    def _transform(self, dataset):
        return self.transform_simple(dataset, self._scale, 'numeric', 'matrix')

    def _scale(self, X):
        if self.is_copied_instance == False or self.scaler is None:
            self.scaler = self._get_scaller()
            self.scaler.fit(X)
        
        return self.scaler.transform(X)

    def _get_scaller(self):
        pass

class TransformationMinMax(TransformationScaling):
    def __init__(self, scaler = None):
        super().__init__(scaler)
        self.name = 'minmax'

    def _copy(self):
        t = TransformationMinMax(self.scaler)
        self.scaler = None
        return t

    def _get_scaller(self):
        return MinMaxScaler()


class TransformationZscore(TransformationScaling):
    def __init__(self, scaler = None):
        super().__init__()
        self.name = 'zscore'

    def _copy(self):
        t = TransformationZscore(self.scaler)
        self.scaler = None
        return t

    def _get_scaller(self):
        return StandardScaler()

class TransformationSum(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'sum'

    def sum_row(self, matrix):
        result = np.empty((matrix.shape[0], 0), matrix.dtype)
        for c in combinations(range(0, matrix.shape[1]), 2):
            result = np.append(result, (matrix[:,c[0]]+matrix[:,c[1]]).reshape(matrix.shape[0], 1), axis = 1)
        return result

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.sum_row, 'numeric', 'matrix')

    def _copy(self):
        return TransformationSum()

class TransformationDiff(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'diff_row'
    
    def diff_row(self, matrix):
        result = np.empty((matrix.shape[0], 0), matrix.dtype)
        for c in combinations(range(0, matrix.shape[1]), 2):
            diff_res = (matrix[:,c[0]]-matrix[:,c[1]]).reshape(matrix.shape[0], 1)
            result = np.append(result, diff_res, axis = 1)
        return result

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.diff_row, 'numeric', 'matrix')

    def _copy(self):
        return TransformationDiff()

class TransformationDiv(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'div_row'
    
    def div_row(self, matrix):
        result = np.empty((matrix.shape[0], 0), matrix.dtype)
        for c in combinations(range(0, matrix.shape[1]), 2):
            if 0 not in matrix[:,c[1]]:
                div_res = (matrix[:,c[0]]/matrix[:,c[1]]).reshape(matrix.shape[0], 1)
                result = np.append(result, div_res, axis = 1)
                if 0 not in matrix[:,c[0]]:
                    result = np.append(result, 1/div_res, axis = 1)
                continue
            if 0 not in matrix[:,c[0]]:
                div_res = (matrix[:,c[1]]/matrix[:,c[0]]).reshape(matrix.shape[0], 1)
                result = np.append(result, div_res, axis = 1)
                continue
        return result

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.div_row, 'numeric', 'matrix')

    def _copy(self):
        return TransformationDiv()

class TransformationMul(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'mul_row'

    def mul_row(self, matrix):
        result = np.empty((matrix.shape[0], 0), matrix.dtype)
        for c in combinations(range(0, matrix.shape[1]), 2):
            result = np.append(result, (matrix[:,c[0]]*matrix[:,c[1]]).reshape(matrix.shape[0], 1), axis = 1)
        return result

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.mul_row, 'numeric', 'matrix')

    def _copy(self):
        return TransformationMul()

class TransformationOneHotEncoder(TransformationBase):
    def __init__(self, encoder = None):
        super().__init__()
        self.name = 'ohe_enc'
        self.encoder = encoder

    def _transform(self, dataset):
        new_dataset = self.transform_simple(dataset.copy(), self.one_hot_enc, 'nominal', 'matrix', 'boolean')
        return new_dataset
        
    def one_hot_enc(self, categorical_features):
        if categorical_features.shape[1] == 0:
            return categorical_features

        if self.is_copied_instance == False or self.encoder is None:
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)
            self.encoder = self.encoder.fit(categorical_features)
        
        return self.encoder.transform(categorical_features)

    def _copy(self):
        t = TransformationOneHotEncoder(self.encoder)
        self.encoder = None
        return t

class TransformationImpute(TransformationBase):
    def __init__(self, imputer = None):
        super().__init__()
        self.imputer = imputer

    def _transform(self, dataset):
        if dataset.contains_missing_values() == False:
            return dataset

        dataset.readd_missing_values()
        
        if self.is_copied_instance == False or self.imputer is None:
            self.imputer = self._get_imputer()
            if isinstance(self.imputer, TransformationImputeSimple.Imputer):
                self.imputer.fit(dataset)
            else:            
                self.imputer.fit(dataset.X)

        X = self.imputer.transform(dataset.X)
        new_dataset = Dataset(X, dataset.y, dataset.data_info, dataset.missing_value_mark)
        new_dataset.mark_as_imputed()

        return new_dataset

    def _get_imputer(self):
        pass

class TransformationImputeKnn(TransformationImpute):
    def __init__(self, imputer = None):
        super().__init__(imputer)
        self.name = 'knn_imp'

    def _copy(self):
        t = TransformationImputeKnn(self.imputer)
        self.imputer = None
        return t

    def _get_imputer(self):
        return KNNImputer(n_neighbors=5)

class TransformationImputeIterative(TransformationImpute):
    def __init__(self, imputer = None):
        super().__init__(imputer)
        self.name = 'ite_imp'

    def _copy(self):
        t = TransformationImputeIterative(self.imputer)
        self.imputer = None
        return t

    def _get_imputer(self):
        return IterativeImputer(max_iter=10)

class TransformationImputeSimple(TransformationImpute):
    def __init__(self, imputer = None):
        super().__init__(imputer)
        self.name = 'sim_imp'

    def _copy(self):
        t = TransformationImputeSimple(self.imputer)
        self.imputer = None
        return t

    def _get_imputer(self):
        return self.Imputer()

    class Imputer():
        def __init__(self):
            self.nominal_imputer = SimpleImputer(strategy='most_frequent')
            self.numeric_imputer = SimpleImputer(strategy='median')
            self.to_mf = None
            self.to_median = None
        
        def fit(self, dataset):
            self.to_mf = np.sort(dataset.get_type_indexes('nominal') + dataset.get_type_indexes('boolean'))
            self.to_mf.dtype = 'int32'
            self.to_median = np.sort(dataset.get_type_indexes('numeric') + dataset.get_type_indexes('timestamp'))
            self.to_median.dtype = 'int32'
            if self.to_mf.shape[0] > 0:
                self.nominal_imputer.fit(dataset.X[:,self.to_mf])
            if self.to_median.shape[0] > 0:
                self.numeric_imputer.fit(dataset.X[:,self.to_median])

        def transform(self, X, y = None):
            if self.to_mf.shape[0] > 0:
                most_freq = self.nominal_imputer.transform(X[:,self.to_mf])
                X[:,self.to_mf] = most_freq

            if self.to_median.shape[0] > 0:
                med = self.numeric_imputer.transform(X[:,self.to_median])
                X[:,self.to_median] = med
                
            return X

class TransformationBinning(TransformationBase):
    def __init__(self, return_type, binner = None):
        super().__init__()
        self.binner = binner    
        self.return_type = return_type

    def _transform(self, dataset):
        return self.transform_simple(dataset, self._discretize, 'numeric', 'matrix', self.return_type)

    def _discretize(self, X):
        if self.is_copied_instance == False or self.binner is None:
            self.binner = self._get_binner()
            self.binner.fit(X)
        
        return self.binner.transform(X)

    def _get_binner(self):
        pass   

class TransformationBinningNominal(TransformationBinning):
    def __init__(self, binner = None):
        super().__init__('nominal', binner)
        self.name = 'bin_nom'
    
    def _get_binner(self):
        return KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')  

    def _copy(self):
        t = TransformationBinningNominal(self.binner)
        self.binner = None
        return t

class TransformationBinningMean(TransformationBinning):
    def __init__(self, binner = None):
        super().__init__('numeric', binner)
        self.name = 'bin_mean' 

    def _get_binner(self):
        return self.MeanBinner() 

    def _copy(self):
        t = TransformationBinningMean(self.binner)
        self.binner = None
        return t

    class MeanBinner():
        def __init__(self):
            self.binner = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')  

        def fit(self, X, y = None):
            self.binner.fit(X)

        def transform(self, X, y = None):
            binned = self.binner.transform(X)
            return self.binner.inverse_transform(binned)

class TransformationTimeBinning(TransformationBase):
    def __init__(self):
        super().__init__()
        self.name = 'tbi'

    def time_binning(self, timestamp):
        binned = []
        date = dt.datetime.fromtimestamp(timestamp).timetuple()
        binned.append(date.tm_yday)
        binned.append(date.tm_wday)
        binned.append(date.tm_year)
        binned.append(date.tm_mon)
        binned.append(date.tm_mday)
        binned.append(date.tm_hour)
        binned.append(date.tm_min)
        binned.append(date.tm_sec)
        return binned

    def _transform(self, dataset):
        return self.transform_simple(dataset, self.time_binning, 'timestamp')

    def _copy(self):
        return TransformationTimeBinning()

        
        

