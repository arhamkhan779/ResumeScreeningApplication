from RESUMESCREENINGAPP.constants import *
from RESUMESCREENINGAPP.utils.common import read_yaml,create_directories
from RESUMESCREENINGAPP.entity.config_entity import DataIngstionConfig,DataPreprocessConfig,ModelConfig,TrainingConfig
import os
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
                 config_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngstionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngstionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) ->DataPreprocessConfig:
          config=self.config.preprocessing_dir
          create_directories([config.root_dir])
          
          data_preprocessing_config=DataPreprocessConfig(
               root_dir=config.root_dir,
               source_dir=config.Unprocess_dir,
               voc_size=self.params.VOC_SIZE,
               target_preprocessor=config.target_preprocessor_file,
               text_preprocessor=config.text_preprocessor_file,
               max_length=self.params.max_length
          )
          return data_preprocessing_config
    
    def get_base_model_config(self) -> ModelConfig:
         config=self.config.Model
         create_directories([config.root_dir])

         model_config=ModelConfig(
              root_dir=config.root_dir,
              model_path=config.model_file,
              batch=self.params.BATCH,
              epochs=self.params.EPOCHS,
              Max_features=self.params.MAX_FEATURES,
              optimizer=self.params.OPTIMIZER,
              loss=self.params.loss,
              metrics=self.params.metrics,
              Voc_size=self.params.VOC_SIZE,
              max_length=self.params.max_length)
         
         return model_config
    
    def get_training_config(self) -> TrainingConfig:
         config=self.config.Training
         create_directories([config.root_dir])
         training_config=TrainingConfig(
              root_dir=config.root_dir,
              trained_model_path=config.model_file,
              data_set_dir=config.data_dir,
              target_preprocessor_path=config.target_preprocessor,
              text_preprocessor_path=config.text_preprocessor,
              results_path=config.results,
              optimizer=self.params.OPTIMIZER,
              loss=self.params.loss,
              metrics=self.params.metrics,
              batch=self.params.BATCH,
              epochs=self.params.EPOCHS,
              base_model=config.base_model
         )
        
         return training_config
    
   