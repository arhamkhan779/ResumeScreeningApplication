from RESUMESCREENINGAPP import logger
from RESUMESCREENINGAPP.pipeline.DATA_INGESTION_PIPELINE import DataIngestionTrainingPipeline
from RESUMESCREENINGAPP.pipeline.DATA_PREPROCESSING_PIPELINE import DataProcessingPipeline

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