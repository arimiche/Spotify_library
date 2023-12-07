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

    def test_string_add_party_music_column(self):
        with self.assertRaises(TypeError):
            example_data = {'danceability': [0.3, 0.4, 0.7],
                'loudness': ['loud', 'soft', 'loud'],
                'energy': [0.6, 0.7, 0.1]}
            example_input = pd.DataFrame(example_data)
            add_party_music_column(example_input)

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

    def test_empty_get_top_words(self):
        with self.assertRaises(KeyError):
            empty_df = pd.DataFrame()
            top_words = get_top_words(empty_df, column_name='text_column', top_n=3)


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
