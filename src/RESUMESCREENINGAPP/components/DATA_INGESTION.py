import os
import zipfile
from RESUMESCREENINGAPP import logger
from RESUMESCREENINGAPP.entity.config_entity import DataIngstionConfig
import kaggle

class DataIngestion:
    def __init__(self, config: DataIngstionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch Data From URL
        '''
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.root_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            logger.info(f"Downloading Data from {dataset_url} into {zip_download_dir}")

            # Downloading the dataset
            os.system(f'kaggle datasets download -d {dataset_url} -p {zip_download_dir}')  
            
            # Constructing the expected original zip file path
            original_zip_path = os.path.join(zip_download_dir, 'resume-dataset.zip')
            new_zip_path = os.path.join(zip_download_dir, 'dataset.zip')

            # Check if the original zip file exists
            if os.path.exists(original_zip_path):
                # Rename the downloaded zip file
                os.rename(original_zip_path, new_zip_path)
                logger.info(f"Downloaded data from {dataset_url} into file {new_zip_path}")
            else:
                logger.error(f"Expected zip file not found: {original_zip_path}")
                raise FileNotFoundError(f"Expected zip file not found: {original_zip_path}")

        except Exception as e:
            logger.error(f"Error occurred while downloading the dataset: {str(e)}")
            raise e
        
    def extract_zip_file(self):
        '''
        This function will be responsible 
        for unzipping the data directory
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        try:
            with zipfile.ZipFile(os.path.join(self.config.root_dir, 'dataset.zip'), 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Extracted files to {unzip_path}")

        except Exception as e:
            logger.error(f"Error occurred while extracting the zip file: {str(e)}")
            raise e
