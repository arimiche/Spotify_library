from sklearn.model_selection import train_test_split

#function to split the data into train and test data
def splitdata(df: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """
    Split a DataFrame into training and testing sets

    Args:
        df (pd.DataFrame): The DataFrame to be split
        test_size (float): The proportion of the dataset to include in the test split
        random_state (int): Seed used by the random number generator for reproducibility

    Returns:
        tuple: A tuple containing the training DataFrame and testing DataFrame
    """
    traindf, testdf = train_test_split(df, test_size=test_size, random_state=random_state) 
    traindf = traindf.reset_index(drop=True) 
    testdf = testdf.reset_index(drop=True)
    return traindf, testdf
