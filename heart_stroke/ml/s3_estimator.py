import os
import sys

from pandas import DataFrame
from heart_stroke.cloud_storage.aws_storage import SimpleStorageService
from heart_stroke.ml.estimator import HeartStrokeModel
from heart_stroke.exception import HeartStrokeException


class StrokeEstimator:
    """ Load from S3 and then make it usable for predition"""
    def __init__(self,
                 bucket_name,
                 model_path,):
        """
        :params bucket_name : Name of your model bucket
        :params model_path : Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model = HeartStrokeModel = None

    def is_model_present(self,model_path):
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,s3_key=model_path
            )
        except Exception as e:
            print(e)
            return False
        
    
    def load_model(self) -> HeartStrokeModel:
        """
        Load the model from the model_path
        """
        return self.s3.load_model(self.model_path,bucket_name = self.bucket_name)
    
    def save_model(self,from_file,remove:bool=False) -> None:
        """
        Save the model to the model_path
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            raise HeartStrokeException(e,sys)
    

    def predict(self,dataframe: DataFrame):
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise HeartStrokeException(e,sys)
