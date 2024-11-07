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
        logger.info("Fetching Configuration From Data Preprocessing Config")
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
        try:
           logger.info("Start Transforming The Data")

           logger.info("Applying Regular Expression")
           cleaned_text = self._clean_text(X)
           logger.info("Applying Stop Words And Stemming")
           stemmed_text = self._remove_stopwords_and_stem(cleaned_text)
           logger.info("Applying One Hot and Pre Padding")
           padded_docs = self._one_hot_and_pad(stemmed_text)
           return padded_docs
        except Exception as e:
            logger.info(e)
            raise e

    def _clean_text(self, X):
        """
        Clean the text by removing non-alphabetic characters and converting to lowercase.
        """
        try:
            cleaned_text = []
            for text in X:
                text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
                text = text.lower()  # Convert to lowercase
                cleaned_text.append(text)
            return cleaned_text
        except Exception as e:
            logger.info(e)
            raise e

    def _remove_stopwords_and_stem(self, X):
        """
        Remove stopwords and apply stemming to the text data.
        """
        try:
            processed_text = []
            for text in X:
                words = text.split()
                words = [self.ps.stem(word) for word in words if word not in self.stop_words]
                processed_text.append(' '.join(words))
            return processed_text
        except Exception as e:
            logger.info(e)
            raise e
        

    def _one_hot_and_pad(self, X):
        """
        Convert the text data into one-hot representations and pad the sequences.
        """
        try:
            one_hot_rep = [keras.preprocessing.text.one_hot(words, self.config.voc_size) for words in X]
            logger.info("Embedding Document with Pre Padding")
            max_length = max([len(doc) for doc in one_hot_rep])  # Find the max sentence length
            padded_docs = keras.preprocessing.sequence.pad_sequences(one_hot_rep, padding='pre', maxlen=max_length)
            return padded_docs
        except Exception as e:
            logger.info(e)
            raise e


class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False) 

    def fit(self, Y, y=None):
        """
        Fit method for OneHotEncoder.
        Convert Y into a numpy array before reshaping and fitting.
        """
        try:
            Y = np.array(Y)
            self.encoder.fit(Y.reshape(-1, 1))  
            return self
        except Exception as e:
            logger.info(e)
            raise e
        
    def transform(self, Y):
        """
        Transform the target variable Y into one-hot encoded format.
        Convert Y into a numpy array before reshaping and transforming.
        """
        try:
           Y = np.array(Y)  # Convert Y to a numpy array
           encoded_Y = self.encoder.transform(Y.reshape(-1, 1))  # Reshape because Y is 1D
           return encoded_Y
        except Exception as e:
            logger.info(e)
            raise e

def create_full_pipeline(config,text_preprocessor,target_preprocessor):

    try:
        logger.info("PipeLine Creation For Text Column ----- > Started")
        text_pipeline = Pipeline(steps=[
          ('preprocessor', DataPreprocessor(config))  
        ])
        logger.info("PipeLine Creation For Text Column ----- > Completed")
        joblib.dump(text_pipeline,text_preprocessor)
        logger.info(f"Target Preprocessor Pipeline save at {text_preprocessor}")

        logger.info("PipeLine Creation For Target Variable ----- > Start")
        target_pipeline = Pipeline(steps=[
          ('target_encoder', TargetEncoder())  
        ])
        logger.info("PipeLine Creation For Target Variable ----- > Completed")
        joblib.dump(target_pipeline,target_preprocessor)
        logger.info(f"Target Preprocessor Pipeline save at {target_preprocessor}")

    except Exception as e:
        logger.info(e)
        raise e



