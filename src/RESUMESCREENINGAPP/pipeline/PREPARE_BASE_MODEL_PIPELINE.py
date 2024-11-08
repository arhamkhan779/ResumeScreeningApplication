from RESUMESCREENINGAPP.config.configuration import ConfigurationManager
from RESUMESCREENINGAPP.components.MODEL import BaseModel
from RESUMESCREENINGAPP import logger

class PrepareBaseModelPipeline:
    def __init__(self):
        pass
    def main(self):
        config=ConfigurationManager()
        configuration=config.get_base_model_config()
        obj=BaseModel(configuration)
        obj.Create_Base_Model()