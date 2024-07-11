import sys
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from heart_stroke.exception import HeartStrokeException
from heart_stroke.logger import logging


class HeartStrokeModel:
    def __init__(
            self, preprocessing_object: ColumnTransformer,trained_model_object: object
    ):
        """
        : params preprocessing_object : Input Object of preprocesser
        : params trained_model_object : Input Pbject of trained model
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Function accepts raw inputs and then transformed rsw input usjg preprocessing_object
        qhich gurantees that the inputs are in the same fotmat as the trained_data
        """
        logging.info("Entered predict method of HeartStroke class")

        try:
            logging.info("Using the trained model to get prediction")

            transformed_feature = self.preprocessing_object.transform(dataframe)

            logging.info("Used the trained model to get prediction")
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise HeartStrokeException(e,sys) from e
        
    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
        