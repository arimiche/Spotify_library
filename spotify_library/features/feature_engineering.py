import pandas as pd
from nltk.tokenize import word_tokenize
from collections import Counter


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


def get_top_words(df, column_name, top_n=20):

    """
    Tokenize and count the words in a specified column of a DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column in the DataFrame to analyze.
    - top_n(int): optional (default=20). The number of top words to retrieve.

    Returns:
    - top_words (list): A list of the top N most frequent words in the specified column.
    """

    # Extract and clean the data from the specified column
    column_data = df[column_name].dropna().astype(str)

    # Tokenize and lowercase the words
    all_words = [word.lower() for item in column_data for word in word_tokenize(item)]

    # Count the occurrences of each word
    word_counts = Counter(all_words)

    # Get the top N most frequent words
    top_words = [word for word, count in word_counts.most_common(top_n)]

    return top_words

def create_dummy_variables(df, column_name, words_list):

    """
    Create dummy variables for specified words in a DataFrame column.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column in the DataFrame to analyze.
    - words_list (list): A list of words for which dummy variables should be created.

    Raises Error: if the column does not exist in the DataFrame

    Returns:
    - df_with_dummies (pd.DataFrame): The DataFrame with additional columns for each word in words_list as dummy variables.
    """

    # Ensure the specified column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Iterate over the words in the list and create dummy variables
    for word in words_list:
        df[word] = df[column_name].str.contains(word, case=False).astype(int)

    return df


def add_sleep_music_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary variable column 'sleep_music' to a DataFrame based on specified conditions

    Args:
        df (pd.DataFrame): The DataFrame containing columns 'instrumentalness', 'duration_minutes', and 'energy'

    Returns:
        pd.DataFrame: The DataFrame with an added 'sleep_music' column
    """
    conditions = (df['instrumentalness'] > 0.6) & (df['duration_minutes'] > 5) & (df['energy'] < 0.5)

    # Add a binary variable column 'sleep_music' based on the specified conditions
    df['sleep_music'] = conditions.astype(int)

    return df

def create_word_count_trackcolumn(df: pd.DataFrame, column:str) -> pd.DataFrame:
    """
    Create a new column 'word_count_track' with the total word count for each track name

    Args:
        df (pd.DataFrame): The DataFrame containing the column 'track_name'

    Returns:
        pd.DataFrame: The DataFrame with an added 'word_count_track' column
    """
    # Check if the specified column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")
    
    #create the new column that counts how many words has each song
    df['word_count_track'] = df[column].apply(lambda text: len(text.split()))
    return df

