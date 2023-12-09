import sys
sys.path.append(r'C:/Users/arimi/Documents/BSE-term1/ComputingData/Final_project_Arianna_Vanessa_Tatiana')
from spotify_library.features.feature_engineering import *
import unittest
import pandas as pd
from pandas.testing import *
import nltk
nltk.download('punkt')


class TestAddPartyMusicColumn(unittest.TestCase):

    def test_normal_add_party_music_column(self):
        example_data = {'danceability': [0.6, 0.4, 0.7],
                'loudness': [-4, -6, -3],
                'energy': [0.6, 0.7, 0.8]}
        example_input = pd.DataFrame(example_data)
        output = add_party_music_column(example_input)
        expected_result = pd.DataFrame({
            'danceability': [0.6, 0.4, 0.7],
            'loudness': [-4, -6, -3],
            'energy': [0.6, 0.7, 0.8],
            'party_music': [1, 0, 1]})
        expected_result['party_music'] = expected_result['party_music'].astype('int32')
        assert_frame_equal(output, expected_result)

    def test_novalue_add_party_music_column(self):
        example_data = {'danceability': [0.2, 0.4, 0.1],
                'loudness': [-7, -6, -7],
                'energy': [0.2, 0.2, 0.3]}
        example_input = pd.DataFrame(example_data)
        output = add_party_music_column(example_input)
        expected_result = pd.DataFrame({
            'danceability': [0.2, 0.4, 0.1],
            'loudness': [-7, -6, -7],
            'energy': [0.2, 0.2, 0.3],
            'party_music': [0, 0, 0]})
        expected_result['party_music'] = expected_result['party_music'].astype('int32')
        assert_frame_equal(output, expected_result) 

class TestGetTopWords(unittest.TestCase):

    def test_normal_get_top_words(self):
        data = {'text_column': ['This is a sample sentence',
                                'Another sentence for testing',
                                'This is the third sentence'],
                'duration_minutes': [6, 2, 3]}
        example_input = pd.DataFrame(data)
        top_words = get_top_words(example_input, column_name='text_column', top_n=3)
        expected_results = ['sentence', 'this', 'is']
        self.assertEqual(top_words, expected_results)

class TestCreateDummyVariables(unittest.TestCase):

    def test_create_dummy_variables(self):
        example = {'text_column': ['This is a sample sentence.',
                                'Another sentence for testing.',
                                'This is the third sentence.']}
        data_input = pd.DataFrame(example)
        words_list = ['sentence', 'this', 'is']
        df_with_dummies = create_dummy_variables(data_input, column_name='text_column', words_list=words_list)
        expected_columns = ['text_column', 'sentence', 'this', 'is']
        expected_df = pd.DataFrame({
            'text_column': ['This is a sample sentence.',
                            'Another sentence for testing.',
                            'This is the third sentence.'],
            'sentence': [1, 1, 1],
            'this': [1, 0, 1],
            'is': [1, 0, 1]})
        column_problem = ['sentence', 'this', 'is']
        expected_df[column_problem] = expected_df[column_problem].astype('int32')
        pd.testing.assert_frame_equal(df_with_dummies[expected_columns], expected_df)

    def test_column_not_exist_create_dummy_variables(self):
        input = {'another_column': ['This is a sample sentence.',
                                   'Another sentence for testing.',
                                   'This is the third sentence.']}
        expected_df = pd.DataFrame(input)
        words_list = ['sentence']
        with self.assertRaises(ValueError):
            create_dummy_variables(expected_df, column_name='text_column', words_list=words_list)

    
class TestAddSleepMusicColumn(unittest.TestCase):

    def test_normal_add_sleep_music_column(self):
        example_data = {'instrumentalness': [0.8, 0.3, 0.2],
                'duration_minutes': [6, 2, 3],
                'energy': [0.2, 0.7, 0.8]}
        example_input = pd.DataFrame(example_data)
        output = add_sleep_music_column(example_input)
        expected_result = pd.DataFrame({
            'instrumentalness': [0.8, 0.3, 0.2],
            'duration_minutes': [6, 2, 3],
            'energy': [0.2, 0.7, 0.8],
            'sleep_music': [1, 0, 0]})
        expected_result['sleep_music'] = expected_result['sleep_music'].astype('int32')
        pd.testing.assert_frame_equal(output, expected_result)

    def test_empty_add_sleep_music_column(self):
        example_data = {'instrumentalness': [],
                'duration_minutes': [],
                'energy': []}
        example_input = pd.DataFrame(example_data)
        output = add_sleep_music_column(example_input)
        expected_result = pd.DataFrame({
            'instrumentalness': [],
            'duration_minutes': [],
            'energy': [],
            'sleep_music': []})
        expected_result['sleep_music'] = expected_result['sleep_music'].astype('int32')
        self.assertTrue(expected_result.equals(output)) 

class TestCreateWordCountTrackColumn(unittest.TestCase):

    def test_normal_create_word_count_track_column(self):
        data = {'track_name': ['Song One', 'Another Song', 'Third Song']}
        df = pd.DataFrame(data)
        result_df = create_word_count_trackcolumn(df, column='track_name')
        expected_df = pd.DataFrame({
            'track_name': ['Song One', 'Another Song', 'Third Song'],
            'word_count_track': [2, 2, 2]})
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_int_create_word_count_track_column(self):
        with self.assertRaises(AttributeError):
            data = {'track_name': ['Song One', 62, 'Third Song']}
            df = pd.DataFrame(data)
            result_df = create_word_count_trackcolumn(df, column='track_name')

