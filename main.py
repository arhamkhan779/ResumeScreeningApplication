from RESUMESCREENINGAPP import logger
from RESUMESCREENINGAPP.pipeline.DATA_INGESTION_PIPELINE import DataIngestionTrainingPipeline
from RESUMESCREENINGAPP.pipeline.DATA_PREPROCESSING_PIPELINE import DataProcessingPipeline
from RESUMESCREENINGAPP.pipeline.PREPARE_BASE_MODEL_PIPELINE import PrepareBaseModelPipeline
from RESUMESCREENINGAPP.pipeline.MODEL_TRAINER_PIPELINE import ModelTrainingPipeline

STAGE_NAME="DATA INGESTION STAGE"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>>")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>>")

except Exception as e:
    logger.info(e)
    raise e

STAGE_NAME="DATA PREPROCESSING STAGE"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>>")
    obj=DataProcessingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>>")
except Exception as e:
    raise e

STAGE_NAME="PREPARE BASE MODEL "

try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>>")
    obj=PrepareBaseModelPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>>")
except Exception as e:
    raise e

STAGE_NAME="MODEL TRAINING STAGe "

try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>>")
    obj=ModelTrainingPipeline()
    obj.main(plots=True)
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>>")
except Exception as e:
    raise e