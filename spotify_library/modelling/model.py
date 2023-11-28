from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

class Modeling(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def trainmodel(self):
        return NotImplementedError
        
    @abstractmethod
    def predictmodel(self):
        return NotImplementedError


class LinearModel(Modeling):

    def __init__(self, traindata, testdata):
        self._features = ['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_minutes', 'acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal', 'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house', 'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal', 'metalcore', 'minimal-techno', 'new-age', 'opera', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop', 'mean_speechiness', 'remix', 'feat', 'live', 'love', 'my', 'vivo', 'mix', 'version', 'remastered', 'your', 'we', 'edit', 'like', 'one', 'night', 'life', 'go', 'rain', 'good', 'let', 'mixed', 'original', 'world', 'new', 'remaster', 'never', 'die']
        self._target = ['popularity']
        self.traindata = traindata
        self.testdata = testdata
        self._model = LinearRegression()

    def trainmodel(self):
        X_train = self.traindata[self._features] 
        Y_train = self.traindata[self._target]
        self._model.fit(X_train, Y_train)

    
    def predictmodel(self):
        X_test = self.testdata[self._features] 
        y_pred = self._model.predict(X_test)
        return y_pred
    
class LassoModel(Modeling):

    def __init__(self, traindata, testdata, alpha=0.5):
        self._features = ['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_minutes', 'acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal', 'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house', 'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal', 'metalcore', 'minimal-techno', 'new-age', 'opera', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop', 'mean_speechiness', 'remix', 'feat', 'live', 'love', 'my', 'vivo', 'mix', 'version', 'remastered', 'your', 'we', 'edit', 'like', 'one', 'night', 'life', 'go', 'rain', 'good', 'let', 'mixed', 'original', 'world', 'new', 'remaster', 'never', 'die']
        self._target = ['popularity']
        self.traindata = traindata
        self.testdata = testdata
        self.alpha = alpha
        self._model = Lasso(alpha = self.alpha)

    def trainmodel(self):
        X_train = self.traindata[self._features] 
        Y_train = self.traindata[self._target]
        self._model.fit(X_train, Y_train)

    
    def predictmodel(self):
        X_test = self.testdata[self._features] 
        y_pred = self._model.predict(X_test)
        return y_pred
    
class RidgeModel(Modeling):

    def __init__(self, traindata, testdata, alpha=0.5):
        self._features = ['year', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'duration_minutes', 'acoustic', 'afrobeat', 'alt-rock', 'ambient', 'black-metal', 'blues', 'breakbeat', 'cantopop', 'chicago-house', 'chill', 'classical', 'club', 'comedy', 'country', 'dance', 'dancehall', 'death-metal', 'deep-house', 'detroit-techno', 'disco', 'drum-and-bass', 'dub', 'dubstep', 'edm', 'electro', 'electronic', 'emo', 'folk', 'forro', 'french', 'funk', 'garage', 'german', 'gospel', 'goth', 'grindcore', 'groove', 'guitar', 'hard-rock', 'hardcore', 'hardstyle', 'heavy-metal', 'hip-hop', 'house', 'indian', 'indie-pop', 'industrial', 'jazz', 'k-pop', 'metal', 'metalcore', 'minimal-techno', 'new-age', 'opera', 'party', 'piano', 'pop', 'pop-film', 'power-pop', 'progressive-house', 'psych-rock', 'punk', 'punk-rock', 'rock', 'rock-n-roll', 'romance', 'sad', 'salsa', 'samba', 'sertanejo', 'show-tunes', 'singer-songwriter', 'ska', 'sleep', 'songwriter', 'soul', 'spanish', 'swedish', 'tango', 'techno', 'trance', 'trip-hop', 'mean_speechiness', 'remix', 'feat', 'live', 'love', 'my', 'vivo', 'mix', 'version', 'remastered', 'your', 'we', 'edit', 'like', 'one', 'night', 'life', 'go', 'rain', 'good', 'let', 'mixed', 'original', 'world', 'new', 'remaster', 'never', 'die']
        self._target = ['popularity']
        self.traindata = traindata
        self.testdata = testdata
        self.alpha = alpha
        self._model = Ridge(alpha = self.alpha)

    def trainmodel(self):
        X_train = self.traindata[self._features] 
        Y_train = self.traindata[self._target]
        self._model.fit(X_train, Y_train)

    
    def predictmodel(self):
        X_test = self.testdata[self._features] 
        y_pred = self._model.predict(X_test)
        return y_pred
    
def model_performance(y_true, y_pred):
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_true, y_pred)
    print("Mean Squared Error (MSE):", mse)

    # Calculate R-squared
    r_squared = r2_score(y_true, y_pred)
    print("R-squared:", r_squared)


def hyperparameter_tuning(prediction_model, X_train, y_train, X_test, y_test):

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