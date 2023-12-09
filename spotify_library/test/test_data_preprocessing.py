import sys
sys.path.append(r'C:/Users/arimi/Documents/BSE-term1/ComputingData/Final_project_Arianna_Vanessa_Tatiana')
from spotify_library.features.data_preprocessing import *
import pandas as pd
import unittest
from pandas.testing import *

class TestDropColumns(unittest.TestCase):

    def test_norma_drop_columns(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        df = pd.DataFrame(data)
        result_df = drop_columns(df, columns_to_drop='B')
        expected_df = pd.DataFrame({'A': [1, 2, 3], 'C': [7, 8, 9]})
        assert_frame_equal(result_df, expected_df) 

    def test_multiple_columns_drop_columns(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        df = pd.DataFrame(data)
        result_df = drop_columns(df, columns_to_drop=['B', 'C'])
        expected_df = pd.DataFrame({'A': [1, 2, 3]})
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_drop_columns_missing_column(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError):
            drop_columns(df, columns_to_drop='D')

class TestTransformMsToMinutes(unittest.TestCase):

    def test_normal_transform_ms_to_minutes(self):
        data = {'duration_ms': [120000, 180000, 90000]}
        df = pd.DataFrame(data)
        result_df = transform_ms_to_minutes(df, columns='duration_ms')
        expected_df = pd.DataFrame({'duration_minutes': [2, 3, 1.5]})
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_wrong_input_transform_ms_to_minutes(self):
        data = {'duration_ms': ['twelve', 180000, 90000]}
        df = pd.DataFrame(data)
        with self.assertRaises(TypeError):
            transform_ms_to_minutes(df, columns='duration_ms')

class TestMissingValuesTable(unittest.TestCase):

    def test_normal_missing_values_table(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
        df = pd.DataFrame(data)
        result_table = missing_values_table(df)
        expected_table = pd.DataFrame(index=[], columns=['Missing Values', '% of Total Values'])
        expected_table['Missing Values'] = expected_table['Missing Values'].astype('int64') 
        expected_table['% of Total Values'] = expected_table['% of Total Values'].astype('float64') 
        pd.testing.assert_frame_equal(result_table, expected_table)

    def test_with_missing_values_missing_values_table(self):
        data = {'A': [1, 2, None], 'B': [4, None, 6], 'C': [7, 8, 9]}
        df = pd.DataFrame(data)
        result_table = missing_values_table(df)
        expected_table = pd.DataFrame({
            'Missing Values': [1, 1],
            '% of Total Values': [33.3, 33.3]
        }, index=['A', 'B'])
        assert_frame_equal(result_table, expected_table)

class TestOneHot(unittest.TestCase):

    def test_normal_one_hot(self):
        data = {'Category': ['A', 'B', 'A', 'C']}
        df = pd.DataFrame(data)
        result_df = one_hot(df, columns='Category')
        result_df['A'] = result_df['A'].astype('int64')
        result_df['B'] = result_df['B'].astype('int64')
        result_df['C'] = result_df['C'].astype('int64')
        expected_df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C'],
            'A': [1, 0, 1, 0],
            'B': [0, 1, 0, 0],
            'C': [0, 0, 0, 1]})
        pd.testing.assert_frame_equal(result_df, expected_df)


    def test_multiple_columns_one_hot(self):
        data = {'Color': ['Red', 'Green', 'Blue'],
                'Size': ['Small', 'Medium', 'Large']}
        df = pd.DataFrame(data)
        result_df = one_hot(df, columns=['Color', 'Size'])
        result_df['Color_Blue'] = result_df['Color_Blue'].astype('int64')
        result_df['Color_Green'] = result_df['Color_Green'].astype('int64')
        result_df['Color_Red'] = result_df['Color_Red'].astype('int64')
        result_df['Size_Large'] = result_df['Size_Large'].astype('int64')
        result_df['Size_Medium'] = result_df['Size_Medium'].astype('int64')
        result_df['Size_Small'] = result_df['Size_Small'].astype('int64')
        expected_df = pd.DataFrame({
                'Color': ['Red', 'Green', 'Blue'],
                'Size': ['Small', 'Medium', 'Large'],
                'Color_Blue': [0, 0, 1],
                'Color_Green': [0, 1, 0],
                'Color_Red': [1, 0, 0],
                'Size_Large': [0, 0, 1],
                'Size_Medium': [0, 1, 0],
                'Size_Small': [1, 0, 0]})
        pd.testing.assert_frame_equal(result_df, expected_df)

class TestAddMeanColumn(unittest.TestCase):

    def test_normal_add_mean_column(self):
        data = {'Category': ['A', 'B', 'A', 'B'],
                'Value': [10, 20, 30, 40]}
        df = pd.DataFrame(data)
        result_df = add_mean_column(df, group_column='Category', target_column='Value', new_column_name='Mean_Value')
        result_df['Value'] = result_df['Value'].astype('int64')
        result_df['Mean_Value'] = result_df['Mean_Value'].astype('int64')
        expected_df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'],
            'Value': [10, 20, 30, 40],
            'Mean_Value': [20, 30, 20, 30]
        })
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_missing_column_add_mean_column(self):
        data = {'Category': ['A', 'B', 'A', 'B'],
                'Value': [10, 20, 30, 40]}
        df = pd.DataFrame(data)
        with self.assertRaises(ValueError):
            add_mean_column(df, group_column='Category', target_column='Missing_Value', new_column_name='Mean_Value')


