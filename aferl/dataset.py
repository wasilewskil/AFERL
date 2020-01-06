import numpy as np
from sklearn.preprocessing import LabelEncoder
from aferl.utils import can_convert_to_float
import datetime as dt

class Dataset:
    def __init__(self, X, y, data_info = [], missing_value_mark = None, datetime_format = None, is_initial = None):
        if is_initial == True:
            self.X = np.array(X, dtype='O')    
        else:    
            self.X = np.array(X, dtype='float64')
        self.y = np.array(y)
        self.missing_value_mark = missing_value_mark
        self.datetime_format = datetime_format
        self.missing_values = None

        if len(data_info) == 0:
            data_info = ['numeric'] * self.X.shape[1]
        if isinstance(data_info, np.ndarray):
            data_info = data_info.tolist()
        self.data_info = data_info        
        self._data_info_set = set(self.data_info)

        if is_initial == True:
            self._initial_transform()
    
    def filter_data(self, data_types):
        return np.array(np.delete(self.X, [i for i, x in enumerate(self.data_info) if x.lower() not in data_types], 1), dtype='float64')

    def get_type_indexes(self, data_type):
        return [i for i, x in enumerate(self.data_info) if x.lower() == data_type]

    def contains_type(self, data_type):
        return data_type in self._data_info_set  

    def contains_missing_values(self):
        return self.missing_values is not None and len(self.missing_values[0]) > 0 and len(self.missing_values[1]) > 0
    
    def copy(self, X = None, y = None, data_info = None, missing_value_mark = None):
        X_copy = np.copy(self.X) if X is None else X
        y_copy = np.copy(self.y) if y is None else y
        data_info_copy = np.copy(self.data_info) if data_info is None else data_info
        new_dataset =  Dataset(X_copy, y_copy, data_info_copy, self.missing_value_mark if missing_value_mark is None else missing_value_mark)
        new_dataset.missing_values = (self.missing_values[0].copy(), self.missing_values[1].copy()) if self.missing_values is not None else None
        return new_dataset

    def remove_type(self, data_type, not_remove_nan = False):
        indexes = self.get_type_indexes(data_type)
        if not_remove_nan:
            nans = np.where(~np.any(np.isnan(self.X[:,indexes]), axis=0))[0]
            indexes = [indexes[i] for i in nans]
        self.remove_columns(indexes)

    def extend(self, X, data_types):
        X = X.astype('float64')
        self.X = np.hstack((self.X, X))
        self.data_info.extend(data_types)
        for dtype in data_types:
            self._data_info_set.add(dtype)

    def remove_columns(self, indexes):
        self.X = np.delete(self.X, indexes, 1)
        self.data_info = np.delete(self.data_info, indexes, 0).tolist()
        self._data_info_set = set(self.data_info)
        if self.missing_values is not None and len(self.missing_values[1]) > 0:
            mask = np.in1d(self.missing_values[1], indexes)
            rows = self.missing_values[0][mask == False]
            columns = self.missing_values[1][mask == False]    
            col_copy = columns.copy()    
            for idx in indexes:
                columns[col_copy > idx] -= 1

            self.missing_values = (rows, columns)

    def readd_missing_values(self):
        if self.missing_values is not None:
            self.X[self.missing_values] = np.nan

    def mark_as_imputed(self):
        self.missing_values = None

    def _initial_transform(self):
        self._transform_missing_values()
        self._transform_nominal()
        self._transform_datetime()
        self.X = self.X.astype('float64')
        self._remove_missing_values()

    def _remove_missing_values(self):
        self.missing_values = np.where(np.isnan(self.X))
        self.X[self.missing_values] = 0

    def _transform_missing_values(self):
        self.X[self.X == self.missing_value_mark] = np.nan
        self.missing_value_mark = np.nan

    def _transform_nominal(self):
        LE = LabelEncoder()
        nominal_indexes = self.get_type_indexes('nominal')
        string_indexes = self.get_type_indexes('string')
        nominal_indexes = nominal_indexes + string_indexes
        for idx in nominal_indexes:
            X = [val for val in self.X[:,idx] if val is not self.missing_value_mark]

            LE = LabelEncoder()
            LE.fit(X)            
            self.X[:,idx] = [np.where(LE.classes_ == val)[0][0] if val is not self.missing_value_mark else self.missing_value_mark for val in self.X[:,idx]]

    def _transform_datetime(self):
        date_indexes = self.get_type_indexes('date')
        for idx in date_indexes:
            self.data_info[idx] = 'timestamp'            
            self.X[:,idx] = [dt.datetime.strptime(val, self.datetime_format).timestamp() if val is not self.missing_value_mark else self.missing_value_mark for val in self.X[:,idx]]
        if 'date' in self._data_info_set:
            self._data_info_set.remove('date')
            self._data_info_set.add('timestamp')


        
