from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP.components.MODEL_TRAINING import ModelTrainer
from RESUMESCREENINGAPP import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self,plots=True):
         try:
            conf=ConfigurationManager()
            config=conf.get_training_config()
            obj=ModelTrainer(config=config)
            obj.Train_Model_On_Custom_Dataset()
            if plots:
               obj.Save_Plot_Results()
         except Exception as e:
             logger.info(e)
             raise e
