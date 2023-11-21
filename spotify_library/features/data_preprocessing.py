from abc import ABCMeta, abstractmethod
import pandas as pd

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
        
    def missing_values_table(self):
                """
        Function to to create table of missing values

        Returns:
            pd.DataFrame: return the dataframe with missing information
        """
        # Total missing values
                mis_val = self.data.isnull().sum()
        
        # Percentage of missing values
                mis_val_percent = 100 * self.data.isnull().sum() / len(df)
        
        # Make a table with the results
                mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
                mis_val_table_ren_columns = mis_val_table.rename(
                columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
                mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

        # Print some summary information
                print ("Dataframe has " + str(df.shape[1]) + " columns.\n"
                       "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        
        # Return the dataframe with missing information
                return mis_val_table_ren_columns
