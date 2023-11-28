import pandas as pd

# Function to load the data
def loaddata(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a Pandas DataFrame

    Args:
        path (str): The file path to the CSV file

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data from the CSV file
    """    
    
    dfraw = pd.read_csv(path)
    df = pd.DataFrame(dfraw)
    return df
