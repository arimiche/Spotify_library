import pandas as pd

def create_binary_columns_for_keywords(df: pd.DataFrame, column:str) -> pd.DataFrame:
    """
    Create binary columns in a DataFrame based on the presence of given keywords in a specified column

    Args:
        df (pd.DataFrame): The DataFrame to be processed
        column (str): The column containing text data for keyword analysis

    Returns:
        pd.DataFrame: The DataFrame with added binary columns indicating the presence of specific keywords
    """
    # Check if the specified column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Create binary columns based on specific words
    df['contains_love'] = df[column].str.contains('love', case=False).astype(int)
    df['contains_dance'] = df[column].str.contains('dance|dancing', case=False, regex=True).astype(int)
    df['contains_I_you_we'] = df[column].str.contains(r'\bI\b|\byou\b|\bwe\b', case=False, regex=True).astype(int)


    return df


def add_party_music_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary variable column 'party_music' to a DataFrame based on specified conditions

    Args:
        df (pd.DataFrame): The DataFrame containing columns 'danceability', 'loudness', and 'energy'

    Returns:
        pd.DataFrame: The DataFrame with an added 'party_music' column
    """
    conditions = (df['danceability'] > 0.5) & (df['loudness'] > -5.131) & (df['energy'] > 0.5)

    # Add a binary variable column 'party_music' based on the specified conditions
    df['party_music'] = conditions.astype(int)

    return df