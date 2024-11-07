from RESUMESCREENINGAPP.constants import *
from RESUMESCREENINGAPP.utils.common import read_yaml,create_directories
from RESUMESCREENINGAPP.entity.config_entity import DataIngstionConfig,DataPreprocessConfig
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
               text_preprocessor=config.text_preprocessor_file
          )
          return data_preprocessing_config