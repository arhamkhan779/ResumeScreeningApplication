from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP.components.DARA_PREPROCESSING import create_full_pipeline

class DataProcessingPipeline:
    def __init__(self):
        pass
    def main(self):
        conifg=ConfigurationManager()
        config_entity=conifg.get_data_preprocessing_config()
        create_full_pipeline(config_entity,config_entity.text_preprocessor,config_entity.target_preprocessor)