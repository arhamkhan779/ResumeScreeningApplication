from RESUMESCREENINGAPP.entity.config_entity import ModelConfig
from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP import logger
from tensorflow import keras
import tensorflow as tf
import joblib

class BaseModel:
    def __init__(self,config:ModelConfig):
        self.config=config
    
    def Create_Base_Model(self):
        try:
            logger.info("LSTM Architecture For Model Development")
            model=keras.models.Sequential()
            model.add(keras.layers.Embedding(input_dim=self.config.Voc_size, output_dim=self.config.Max_features))
            model.add(keras.layers.LSTM(units=128, return_sequences=False)) 
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense(25, activation='softmax'))  
            model.compile(optimizer=self.config.optimizer, loss=self.config.loss, metrics=self.config.metrics)
            print(model.summary())
            logger.info(f"Saving base model as Model.h5 at {self.config.model_path}")
            joblib.dump(model,self.config.model_path)
            

        except Exception as e:
            logger.info(e)
            raise e
        
