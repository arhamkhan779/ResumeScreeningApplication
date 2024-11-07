import re
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tensorflow import keras
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from RESUMESCREENINGAPP.entity.config_entity import DataPreprocessConfig
from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP import logger
import joblib


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, config:DataPreprocessConfig):
        """
        Initialize the DataPreprocessor class with configuration settings.
        """
        self.config = config
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.voc_size = self.config.voc_size  # Configuration for vocabulary size

    def fit(self, X, y=None):
        """
        Fit method for scikit-learn compatibility. No fitting is required for this transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the input text data by applying multiple preprocessing steps.
        """
        # Step 1: Clean Text (Remove non-alphabetic characters, convert to lowercase)
        cleaned_text = self._clean_text(X)

        # Step 2: Apply Stopwords Removal and Stemming
        stemmed_text = self._remove_stopwords_and_stem(cleaned_text)

        # Step 3: One-hot Encoding and Padding
        padded_docs = self._one_hot_and_pad(stemmed_text)

        return padded_docs

    def _clean_text(self, X):
        """
        Clean the text by removing non-alphabetic characters and converting to lowercase.
        """
        logger.info("Cleaning Text Data")
        cleaned_text = []
        for text in X:
            text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
            text = text.lower()  # Convert to lowercase
            cleaned_text.append(text)
        return cleaned_text

    def _remove_stopwords_and_stem(self, X):
        """
        Remove stopwords and apply stemming to the text data.
        """
        logger.info("Applying Stopwords Removal and Stemming")
        processed_text = []
        for text in X:
            words = text.split()
            words = [self.ps.stem(word) for word in words if word not in self.stop_words]
            processed_text.append(' '.join(words))
        return processed_text

    def _one_hot_and_pad(self, X):
        """
        Convert the text data into one-hot representations and pad the sequences.
        """
        logger.info("Convert Text Data into One Hot Representation")
        one_hot_rep = [keras.preprocessing.text.one_hot(words, self.voc_size) for words in X]
        logger.info("Embedding Document with Pre Padding")
        max_length = max([len(doc) for doc in one_hot_rep])  # Find the max sentence length
        padded_docs = keras.preprocessing.sequence.pad_sequences(one_hot_rep, padding='pre', maxlen=max_length)
        return padded_docs


# Separate pipeline for handling the target variable (Y)
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Updated argument name from 'sparse' to 'sparse_output'
        self.encoder = OneHotEncoder(sparse_output=False)  # OneHotEncoder, non-sparse matrix (dense)

    def fit(self, Y, y=None):
        """
        Fit method for OneHotEncoder.
        Convert Y into a numpy array before reshaping and fitting.
        """
        Y = np.array(Y)  # Convert Y to a numpy array
        self.encoder.fit(Y.reshape(-1, 1))  # Reshape because OneHotEncoder expects a 2D array
        return self

    def transform(self, Y):
        """
        Transform the target variable Y into one-hot encoded format.
        Convert Y into a numpy array before reshaping and transforming.
        """
        Y = np.array(Y)  # Convert Y to a numpy array
        encoded_Y = self.encoder.transform(Y.reshape(-1, 1))  # Reshape because Y is 1D
        return encoded_Y


# Define the full pipeline, including both text preprocessing and target encoding
def create_full_pipeline(config):
    # Text preprocessing pipeline for feature X
    text_pipeline = Pipeline(steps=[
        ('preprocessor', DataPreprocessor(config))  # Preprocess text features (X)
    ])
    
    # Target encoding pipeline for target Y
    target_pipeline = Pipeline(steps=[
        ('target_encoder', TargetEncoder())  # Encode target variable (Y)
    ])
    
    joblib.dump(text_pipeline,config.text_pipeline)
    joblib.dump(target_pipeline,config.target_pipeline)


# Example Usage
if __name__ == "__main__":
    conifg=ConfigurationManager()
    config_entity=conifg.get_data_preprocessing_config()
    text_pipeline, target_pipeline = create_full_pipeline(config_entity)

