from abc import ABCMeta, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class data_preprocessing(metaclass=ABCMeta):
    """
    Abstract base class for data preprocessing operations
    """
    def __init__(self, name):
        self.name = name
        
        @abstractmethod
        def drop_columns(self):
            return NotImplementedError
        
        @abstractmethod
        def transform_columns(self):
            return NotImplementedError
        
        @abstractmethod

        def missing_values_table(self):
             return NotImplementedError
        
        @abstractmethod
        def one_hot(self):
             return NotImplementedError
        
        @abstractmethod
        def add_mean_column(self):
            return NotImplementedError



        def encode(self):
            return NotImplementedError

        
    
class drop(data_preprocessing):
    """
    Class for dropping specified columns of a DataFrame
        
    Args:
        data (pd.DataFrame): The DataFrame to be processed
        columns_to_drop (str or list): The column(s) to be dropped        
    """
    def __init__(self, data, columns_to_drop):
        self.data = data
        self.columns_to_drop = columns_to_drop
        
    def drop_columns(self):
        """
        FUnction to drop specified columns from the DataFrame
        
        
        Raises:
            ValueError: If any specified column is not found in the DataFrame

        Returns:
            pd.DataFrame: The DataFrame after dropping the specified columns
        """
        if isinstance(self.columns_to_drop, str):  # If a single column name is provided
            self.columns_to_drop = [self.columns_to_drop]  # Convert it to a list

        missing_columns = [col for col in self.columns_to_drop if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

        self.data.drop(columns=self.columns_to_drop, axis=1, inplace=True)
        return self.data
    
class transform_columns(data_preprocessing):
    """
    Class for transforming specified columns in a DataFrame

    Args:
        data (pd.DataFrame): The DataFrame to be processed
        columns (str): The column to be transformed
    """
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def transform_ms_to_minutes(self):
        """
        Transform a specified column by converting milliseconds to minutes and drop the original column

        Raises:
            ValueError: If the specified column is not found in the DataFrame

        Returns:
            pd.DataFrame: The DataFrame after the specified transformation
        """
        if self.columns not in self.data.columns:
            raise ValueError(f"Column '{self.columns}' not found in the DataFrame.")

        # Extract the first part of the column name before the underscore
        column_prefix = self.columns.split('_')[0]

        # Create a new column with the duration in minutes
        self.data[f'{column_prefix}_minutes'] = self.data[self.columns] / 60000
        self.data.drop(self.columns, axis=1, inplace=True)

        return self.data



class missing_values(data_preprocessing):

    """
    Class to work with missing values
        
    Args:
        data (pd.DataFrame): The DataFrame to be processed       
    """

    def __init__(self, data):
        self.data = data

        """
        Function to to create table of missing values

        Returns:
            return the table with missing information
        """     

    def missing_values_table(self):
        # Total missing values
        mis_val = self.data.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * self.data.isnull().sum() / len(self.data)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        # Print some summary information
        print ("Dataframe has " + str(self.data.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


class dummies(data_preprocessing):

    """
    Class to create dummy variables.
        
    Args:
        data (pd.DataFrame): The DataFrame to be processed 
        columns (str): The names of the columns to create dummies.      
    """

    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

        """
        Function to to create table of missing values

        Returns:
            return the table with missing information
        """     

    def one_hot(self):
        dummy = pd.get_dummies(self.data[self.columns])
        encoded_df = pd.concat([self.data, dummy], axis=1)
        return encoded_df

class target_encoding(data_preprocessing):

    """
    Class to execute target encoding.
        
    Args:
        - data (pd.DataFrame): The DataFrame to be processed 
        - group_column (str): The name of the column by which the DataFrame should be grouped.
    - target_column (str): The name of the column for which the mean values should be calculated.
    - new_column_name (str): The name of the new column to be added to the DataFrame.      
    """

    def __init__(self, data, group_column, target_column, new_column_name):
        self.data = data
        self.group_column = group_column
        self.target_column = target_column
        self.new_column_name = new_column_name

        """
        Function to add a new column to a DataFrame with the mean values of a target column grouped by another column.

        Raises Error: if the column of interest doesn't exist in the DataFrame.

        Returns:
            The DataFrame with the new column added, containing the mean values of the target column grouped by the specified column.
        """     

    def add_mean_column(self):
        # Ensure the specified columns exist
        if self.group_column not in self.data.columns or self.target_column not in self.data.columns:
            raise ValueError(f"One or more specified columns do not exist in the DataFrame.")

        # Calculate the mean values and create a new column
        self.data[self.new_column_name] = self.data.groupby(self.group_column)[self.target_column].transform('mean')


    
class labelencode(data_preprocessing):

    def __init__(self, data, columns, encoder):
        self.data = data
        self.columns = columns
        self.encoder = encoder

    def encode(self):
        """
        FUnction to encode specified columns from the DataFrame
        
        Raises:
            ValueError: If any specified column is not found in the DataFrame

        Returns:
            pd.DataFrame: The DataFrame after adding encoded columns
        """
        if isinstance(self.columns, str):  # If a single column name is provided
            self.columns = [self.columns]  # Convert it to a list

        missing_columns = [col for col in self.columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

        self.encoder = LabelEncoder()
        self.data[f'{self.columns}_encoded'] = self.encoder.fit_transform(self.data[self.columns])

        return self.data
