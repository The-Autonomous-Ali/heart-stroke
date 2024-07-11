import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sklearn.metrics import f1_score

from heart_stroke.constants.training_pipeline import TARGET_COLUMN
from heart_stroke.entity.artifact_entity import (
    DataIngestionArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact
)

from heart_stroke.entity.config_entity import ModelEvaluationConfig
from heart_stroke.exception import HeartStrokeException
from heart_stroke.logger import logging
from heart_stroke.ml.s3_estimator import StrokeEstimator


@dataclass
class EvaluationResponse:
    trained_model_f1_score:float
    best_model_f1_score: float
    is_model_accepted: bool
    difference:float


class ModelEvaluation:
    def __init__(
            self,
            model_eval_config: ModelEvaluationConfig,
            data_ingestion_artifact: DataIngestionArtifact,
            model_trainer_artifact: ModelTrainerArtifact
    ):
        """
        Params model_evaluation_config : Output reference of data evaluation artiffact stage
        param data_ingestion_artifact: Output reference of data ingestion artifact stage
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise HeartStrokeException(e,sys)
        
    
    def get_best_model_(self) -> Optional[StrokeEstimator]:
        """
        Method Name: get_best_model
        Description : This function is used to get model in production
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            heart_stroke_estimator = StrokeEstimator(
                bucket_name=bucket_name,model_path=model_path
            )

            if heart_stroke_estimator.is_model_present(model_path=model_path):
                return heart_stroke_estimator
            return None
        
        except Exception as e:
            raise HeartStrokeException(e,sys)
        

    def evaluate_model(self) -> EvaluationResponse:
        """Description : This func is used to evaluate trained model with production model and choose best model"""

        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x,y = test_df.drop(TARGET_COLUMN,axis=1),test_df[TARGET_COLUMN]
            trained_model_f1_score = (
                self.model_trainer_artifact.metric_artifact.f1_score
            )

            best_model_f1_score = None
            best_model =self.get_best_model_()
            if best_model is  not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y,y_hat_best_model)


            tmp_best_model_score =(
                0 if best_model_f1_score is None else best_model_f1_score
            )
            result = EvaluationResponse(
                trained_model_f1_score = trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference= trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            raise HeartStrokeException(e,sys)
        

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path= s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy= evaluate_model_response.difference,
            )
            logging.info(f"Model evaluation artifact : {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise HeartStrokeException(e,sys) from e



            