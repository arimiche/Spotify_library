# Spotify Library
by Arianna Michelangelo, Vanessa Kromm and Tatiana Bakwenye

# Purpose
The purpose of this library is to predict the popularity of a song with a regression model based on spotify data of 2022. The used data can be found in this folder.

# Structure
The library is structured as following:
- features
  - data_preprocessing
  - feature_engineering
- modelling
  - data_splitting
  - model
- preparation
  - data_loading
  - EDA
- test
  - test_data_preprocessing
  - test_feature_engineering

All the functions of the library are executed in the file pipeline.ipynb, where the steps are justified and results are commented.

# Installing the library
To install the library, in your terminal navigate to this folder and execute "pip install -e ."

# Changing the library
Before adding a function to the library, check whether it is not yet existing in the library to prevent redundancy.
When you add the function, make sure to place it into the right file and folder accoring to it's purpose and to describe it in the same pattern as the existing functions (short description, arguments, output).
Also remember to check the functionality of the function with tests, that you document in the test folder in the corresponding test file.
When you then use the function in the pipeline, also make comments on the results and if necessary change comments at other places.
