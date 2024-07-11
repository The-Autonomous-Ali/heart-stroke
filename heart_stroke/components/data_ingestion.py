import os 
import sys
from pandas import DataFrame

from heart_stroke.constants.training_pipeline import SCHEMA_FILE_PATH
from heart_stroke.entity.config_entity import DataIngestionConfig
from heart_stroke.entity.artifact_entity import DataIngestionArtifact
from heart_stroke.exception import HeartStrokeException
from heart_stroke.logger import logging
from heart_stroke.data_access.heart_stroke_data import StrokeData
from heart_stroke.utils.main import *
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise HeartStrokeException(e,sys)
        
    
    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name : export_data_into_feature_store
        Description : This method exports the dataframe from mongodb feature store as dataframe


        Output : Dataframe
        On Failure : Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from Mongodb")
            heart_stroke_data=StrokeData()
            dataframe = heart_stroke_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path= self.data_ingestion_config.feature_store_file_path
            dir_path= os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(
                f"Saving exported data into feature store file path: {feature_store_file_path}"
            )
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise HeartStrokeException(e,sys)
        

    def split_data_as_train_test(self,dataframe: DataFrame)-> None:
        """
        Method Name : Split_data_as_train_test
        Description : This method splits the dataframe into train set and test set based on split ratio

        Output : Folder is created in s3 bucket
        On failure : Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(
                dataframe,test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Existed split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            logging.info(f"Exporting train and test file path")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,index=False,header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,index=False,header= True
            )

            logging.info(f"Exported train and test file path")
        except Exception as e:
            raise HeartStrokeException(e,sys) from e
        

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        """
        Method Name : Initiate_data_ingestion
        Description : This method initiates the ingestion components of training pipeline

        Output: train set and test set are retured as the artifacts of data ingestion components
        On Failure : Write as exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_ingestion class")

        try:
            dataframe = self.export_data_into_feature_store()
            _schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

            dataframe = dataframe.drop(_schema_config['Drop_columns'],axis=1)

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Existed initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path,
            )

            logging.info(f"Data Ingestion artifacts : {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise HeartStrokeException(e,sys) from e
 
