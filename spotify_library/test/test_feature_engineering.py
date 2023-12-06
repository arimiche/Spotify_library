#%%
import sys
sys.path.append(r'C:/Users/arimi/Documents/BSE-term1/ComputingData/Final_project_Arianna_Vanessa_Tatiana')
from spotify_library.features.feature_engineering import *
import unittest
import pandas as pd
from pandas.testing import *


#%%
class Test_Add_Party_Music_Column(unittest.TestCase):

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
        pd.testing.assert_frame_equal(output, expected_result)

    def test_novalue_add_party_music_column(self):
        example_data = {'danceability': [0.3, 0.4, 0.7],
                'loudness': [-4, -6, -3],
                'energy': [0.6, 0.7, 0.1]}
        example_input = pd.DataFrame(example_data)
        output = add_party_music_column(example_input)
        expected_result = pd.DataFrame({
            'danceability': [0.6, 0.4, 0.7],
            'loudness': [-4, -6, -3],
            'energy': [0.6, 0.7, 0.8],
            'party_music': [0, 0, 0]})
        pd.testing.assert_frame_equal(output, expected_result)

    def test_string_add_party_music_column(self):
        with self.assertRaises(TypeError):
            example_data = {'danceability': [0.3, 0.4, 0.7],
                'loudness': ['loud', 'soft', 'loud'],
                'energy': [0.6, 0.7, 0.1]}
            example_input = pd.DataFrame(example_data)
            add_party_music_column(example_input)


#%%
class Test_Add_Sleep_Music_Column(unittest.TestCase):

    def test_normal_add_sleep_music_column(self):
        example_data = {'instrumentalness': [0.8, 0.3, 0.2],
                'duration_minutes': [6, 2, 3],
                'energy': [0.2, 0.7, 0.8]}
        example_input = pd.DataFrame(example_data)
        output = add_party_music_column(example_input)
        expected_result = pd.DataFrame({
            'instrumentalness': [0.8, 0.3, 0.2],
            'duration_minutes': [6, 2, 3],
            'energy': [0.2, 0.7, 0.8],
            'party_music': [1, 0, 0]})
        pd.testing.assert_frame_equal(output, expected_result)

    def test_empty_add_sleep_music_column(self):
        example_data = {'instrumentalness': [],
                'duration_minutes': [],
                'energy': []}
        example_input = pd.DataFrame(example_data)
        output = add_party_music_column(example_input)
        expected_result = pd.DataFrame({
            'instrumentalness': [],
            'duration_minutes': [],
            'energy': [],
            'party_music': []})
        pd.testing.assert_frame_equal(output, expected_result)

