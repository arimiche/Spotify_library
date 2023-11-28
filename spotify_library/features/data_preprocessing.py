import pandas as pd

def drop_columns(data, columns_to_drop):
    """
    FUnction to drop specified columns from the DataFrame
    Args:

    data (pd.DataFrame): The DataFrame to be processed
    columns_to_drop (str): The column to be dropped
        
    Raises:
        ValueError: If any specified column is not found in the DataFrame

    Returns:
        pd.DataFrame: The DataFrame after dropping the specified columns
    """
    if isinstance(columns_to_drop, str):  # If a single column name is provided
        columns_to_drop = [columns_to_drop]  # Convert it to a list

    missing_columns = [col for col in columns_to_drop if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Columns {missing_columns} not found in the DataFrame.")

    data.drop(columns= columns_to_drop, axis=1, inplace=True)
    return data


def transform_ms_to_minutes(data, columns):
    """
    Transform a specified column by converting milliseconds to minutes and drop the original column
    Args:
    data (pd.DataFrame): The DataFrame to be processed
    columns (str): The column to be transformed
    Raises:
        ValueError: If the specified column is not found in the DataFrame

    Returns:
        pd.DataFrame: The DataFrame after the specified transformation
    """
    if columns not in data.columns:
        raise ValueError(f"Column '{columns}' not found in the DataFrame.")

    # Extract the first part of the column name before the underscore
    column_prefix = columns.split('_')[0]

    # Create a new column with the duration in minutes
    data[f'{column_prefix}_minutes'] = data[columns] / 60000
    data.drop(columns, axis=1, inplace=True)

    return data


def missing_values_table(data):

    """
    Function to to create table of missing values
    Args:
        data (pd.DataFrame): The DataFrame to be processed         
    Returns:
        return the table with missing information
    """    
    # Total missing values
    mis_val = data.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * data.isnull().sum() / len(data)
        
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Dataframe has " + str(data.shape[1]) + " columns.\n"
    "There are " + str(mis_val_table_ren_columns.shape[0]) + " columns that have missing values.")
        
    # Return the dataframe with missing information
    return mis_val_table_ren_columns



def one_hot(data, columns):

    """
    Function to to create table of missing values

    Args:
        data (pd.DataFrame): The DataFrame to be processed 
        columns (str): The names of the columns to create dummies.    

    Returns:
        return the table with missing information
    """     

    dummy = pd.get_dummies(data[columns])
    encoded_df = pd.concat([data, dummy], axis=1)
    return encoded_df

def add_mean_column(data, group_column, target_column, new_column_name):
    """
    Function to add a new column to a DataFrame with the mean values of a target column grouped by another column.

    Args:
        - data (pd.DataFrame): The DataFrame to be processed 
        - group_column (str): The name of the column by which the DataFrame should be grouped.
        - target_column (str): The name of the column for which the mean values should be calculated.
        - new_column_name (str): The name of the new column to be added to the DataFrame.

    Raises Error: if the column of interest doesn't exist in the DataFrame.

    Returns:
        The DataFrame with the new column added, containing the mean values of the target column grouped by the specified column.
    """   
    # Ensure the specified columns exist
    if group_column not in data.columns or target_column not in data.columns:
        raise ValueError(f"One or more specified columns do not exist in the DataFrame.")

    # Calculate the mean values and create a new column
    data[new_column_name] = data.groupby(group_column)[target_column].transform('mean')

    return data


    
