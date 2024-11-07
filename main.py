from RESUMESCREENINGAPP import logger
from RESUMESCREENINGAPP.pipeline.DATA_INGESTION_PIPELINE import DataIngestionTrainingPipeline

STAGE_NAME="DATA INGESTION STAGE"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started >>>>>>")
    obj=DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed >>>>>>")

except Exception as e:
    logger.info(e)
    raise e