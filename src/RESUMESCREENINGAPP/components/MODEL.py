from RESUMESCREENINGAPP.entity.config_entity import ModelConfig
from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP import logger
from tensorflow import keras
import tensorflow as tf

class BaseModel:
    def __init__(self,config:ModelConfig):
        self.config=config
    
    def Create_Base_Model(self):
        try:
            model=keras.models.Sequential()
            model.add(keras.layers.Embedding(input_dim=self.config.Voc_size, output_dim=self.config.Max_features, input_length=self.config.max_length))
            model.add(keras.layers.LSTM(units=128, return_sequences=True))  # return_sequences=True allows the next layer to receive sequences
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dropout(0.2))

# Output layer with softmax activation for multi-class classification
            model.add(keras.layers.Dense(3, activation='softmax'))  
            model.compile(optimizer=self.config.optimizer, loss=self.config.loss, metrics=self.config.metrics)

        except Exception as e:
            logger.info(e)
            raise e
        

if __name__ == "__main__":
    config=ConfigurationManager()
    configuration=config.get_base_model_config()
    obj=BaseModel(configuration)
    obj.Create_Base_Model()
