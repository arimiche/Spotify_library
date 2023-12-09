from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

class Modeling(metaclass=ABCMeta):
    def __init__(self, name):
        """
        Initialize a Modeling object

        Args:
            name (str): The name of the model
        """
        self.name = name

    @abstractmethod
    def trainmodel(self):
        """
        Abstract method for training the model
        """
        return NotImplementedError
            
    @abstractmethod
    def predictmodel(self):
        """
        Abstract method for making predictions using the trained model
        """
        return NotImplementedError
    
    @abstractmethod
    def coefficients(self):
        """
        Abstract method for retrieving model coefficients
        """
        return NotImplementedError


class LinearModel(Modeling):

    def __init__(self, traindata, testdata):
        """
        Initialize a LinearModel object

        Args:
            traindata (pd.DataFrame): The training data
            testdata (pd.DataFrame): The testing data
        """
        self._features = ['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_minutes', 'acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal', 'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house', 'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal', 'metalcore', 'minimal-techno', 'new-age', 'opera', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop', 'mean_speechiness', 'remix', 'feat', 'live', 'love', 'my', 'vivo', 'mix', 'version', 'remastered', 'your', 'we', 'edit', 'like', 'one', 'night', 'life', 'go', 'rain', 'good', 'let', 'mixed', 'original', 'world', 'new', 'remaster', 'never', 'die']
        self._target = ['popularity']
        self.traindata = traindata
        self.testdata = testdata
        self._model = LinearRegression()

    def trainmodel(self):
        """
        Train the linear regression model
        """
        X_train = self.traindata[self._features] 
        Y_train = self.traindata[self._target]
        self._model.fit(X_train, Y_train)

    
    def predictmodel(self):
        """
        Make predictions using the trained linear regression model

        Returns:
            np.ndarray: Predicted values
        """
        X_test = self.testdata[self._features] 
        y_pred = self._model.predict(X_test)
        return y_pred
    
    def coefficients(self):
        """
        Get linear regression model coefficients and display the top positive and negative coefficients
        """  
        X_train = self.traindata[self._features] 
        feature_names = X_train.columns
        feature_values_lin = self._model.coef_

        # Create a DataFrame with feature names and coefficients
        linear_coeff = pd.DataFrame({'Feature': feature_names, 'Coefficient': feature_values_lin.flatten()})


        # Sort the DataFrame by the values of coefficients in descending order
        linear_coeff = linear_coeff.sort_values(by='Coefficient', ascending=False)


        # Print the sorted DataFrame
        print("Top 10 positive coefficients:")
        print(linear_coeff.head(10))

        print("\nTop 10 negative coefficients:")
        print(linear_coeff.tail(10)[::-1])  
        


class LassoModel(Modeling):

    def __init__(self, traindata, testdata, alpha=0.5):
        """
        Initialize a LassoModel object

        Args:
            traindata (pd.DataFrame): The training data
            testdata (pd.DataFrame): The testing data
        """
        self._features = ['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_minutes', 'acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal', 'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house', 'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal', 'metalcore', 'minimal-techno', 'new-age', 'opera', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop', 'mean_speechiness', 'remix', 'feat', 'live', 'love', 'my', 'vivo', 'mix', 'version', 'remastered', 'your', 'we', 'edit', 'like', 'one', 'night', 'life', 'go', 'rain', 'good', 'let', 'mixed', 'original', 'world', 'new', 'remaster', 'never', 'die']
        self._target = ['popularity']
        self.traindata = traindata
        self.testdata = testdata
        self.alpha = alpha
        self._model = Lasso(alpha = self.alpha)

    def trainmodel(self):
        """
        Train the Lasso regression model
        """
        X_train = self.traindata[self._features] 
        Y_train = self.traindata[self._target]
        self._model.fit(X_train, Y_train)

    
    def predictmodel(self):
        """
        Make predictions using the trained Lasso regression model

        Returns:
            np.ndarray: Predicted values
        """
        X_test = self.testdata[self._features] 
        y_pred = self._model.predict(X_test)
        return y_pred
    
    def coefficients(self):
        """
        Get Lasso regression model coefficients and display the top positive and negative coefficients
        """   
        X_train = self.traindata[self._features] 
        feature_names = X_train.columns
        feature_values_lin = self._model.coef_

        # Create a DataFrame with feature names and coefficients
        Lasso_coeff = pd.DataFrame({'Feature': feature_names, 'Coefficient': feature_values_lin.flatten()})


        # Sort the DataFrame by the values of coefficients in descending order
        Lasso_coeff = Lasso_coeff.sort_values(by='Coefficient', ascending=False)


        # Print the sorted DataFrame
        print("Top 10 positive coefficients:")
        print(Lasso_coeff.head(10))

        print("\nTop 10 negative coefficients:")
        print(Lasso_coeff.tail(10)[::-1])  
    
class RidgeModel(Modeling):

    def __init__(self, traindata, testdata, alpha=0.5):
        """
        Initialize a RidgeModel object

        Args:
            traindata (pd.DataFrame): The training data
            testdata (pd.DataFrame): The testing data
        """
        self._features = ['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_minutes', 'acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal', 'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house', 'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal', 'metalcore', 'minimal-techno', 'new-age', 'opera', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop', 'mean_speechiness', 'remix', 'feat', 'live', 'love', 'my', 'vivo', 'mix', 'version', 'remastered', 'your', 'we', 'edit', 'like', 'one', 'night', 'life', 'go', 'rain', 'good', 'let', 'mixed', 'original', 'world', 'new', 'remaster', 'never', 'die']
        self._target = ['popularity']
        self.traindata = traindata
        self.testdata = testdata
        self.alpha = alpha
        self._model = Ridge(alpha = self.alpha)

    def trainmodel(self):
        """
        Train the Ridge regression model
        """
        X_train = self.traindata[self._features] 
        Y_train = self.traindata[self._target]
        self._model.fit(X_train, Y_train)

    
    def predictmodel(self):
        """
        Make predictions using the trained Ridge regression model

        Returns:
            np.ndarray: Predicted values
        """
        X_test = self.testdata[self._features] 
        y_pred = self._model.predict(X_test)
        return y_pred
    
    def coefficients(self):  
        """
        Get Ridge regression model coefficients and display the top positive and negative coefficients
        """ 
        X_train = self.traindata[self._features] 
        feature_names = X_train.columns
        feature_values_lin = self._model.coef_

        # Create a DataFrame with feature names and coefficients
        Ridge_coeff = pd.DataFrame({'Feature': feature_names, 'Coefficient': feature_values_lin.flatten()})


        # Sort the DataFrame by the values of coefficients in descending order
        Ridge_coeff = Ridge_coeff.sort_values(by='Coefficient', ascending=False)


        # Print the sorted DataFrame
        print("Top 10 positive coefficients:")
        print(Ridge_coeff.head(10))

        print("\nTop 10 negative coefficients:")
        print(Ridge_coeff.tail(10)[::-1])  
    
def model_performance(y_true, y_pred):
    """
    Calculate and display model performance metrics: MSE and R-squared

    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values    
    """
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Calculate R-squared
    r_squared = r2_score(y_true, y_pred)
    print("R-squared:", r_squared)


def hyperparameter_tuning(prediction_model, X_train, y_train, X_test, y_test):
    """
    Tune hyperparameters of the given prediction model using alpha values

    Args:
        prediction_model: The prediction model class
        X_train (pd.DataFrame): Training features
        y_train (np.ndarray): Training target
        X_test (pd.DataFrame): Testing features
        y_test (np.ndarray): Testing target    
    """

    # Define a range of alpha values to try
    alphas = np.linspace(0.001, 10, 200)

    # Initialize variables to store the best alpha and minimum MSE
    best_alpha = None
    min_mse = float('inf')


    # Loop over the alpha values
    for alpha in alphas:
        # Fit model
        model = prediction_model(alpha=alpha, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Update best alpha and minimum MSE 
        if mse < min_mse:
            min_mse = mse
            best_alpha = alpha

    # Print the best alpha and minimum MSE 
    print("Best alpha:", best_alpha)
    print("Best MSE:", min_mse,)  