from abc import ABCMeta, abstractmethod
import pandas as pd

class data_preprocessing(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name
        
        @abstractmethod
        def drop_columns(self):
            return NotImplementedError
        
        @abstractmethod
        def transform_columns(self):
            return NotImplementedError
        
    
class drop(data_preprocessing):
    def __init__(self, data, columns_to_drop):
        self.data = data
        self.columns_to_drop = columns_to_drop
        
    def drop_columns(self):
        if isinstance(self.columns_to_drop, str):  # If a single column name is provided
            self.columns_to_drop = [self.columns_to_drop]  # Convert it to a list

        missing_columns = [col for col in self.columns_to_drop if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

        self.data.drop(columns=self.columns_to_drop, axis=1, inplace=True)
        return self.data
    
class transform_columns(data_preprocessing):
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def transform_ms_to_minutes(self):
        if self.columns not in self.data.columns:
            raise ValueError(f"Column '{self.columns}' not found in the DataFrame.")

        # Extract the first part of the column name before the underscore
        column_prefix = self.columns.split('_')[0]

        # Create a new column with the duration in minutes
        self.data[f'{column_prefix}_minutes'] = self.data[self.columns] / 60000
        self.data.drop(self.columns, axis=1, inplace=True)

        return self.data
