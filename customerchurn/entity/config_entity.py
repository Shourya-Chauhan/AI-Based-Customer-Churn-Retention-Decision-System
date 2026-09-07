from datetime import datetime

import os

from customerchurn.constant import training_pipeline



class TrainingPipelineConfig:
    def __init__(self, timestamp=None):
         if timestamp is None:
            timestamp = datetime.now()

         self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
         self.pipeline_name = training_pipeline.PIPELINE_NAME
         self.artifact_name = training_pipeline.ARTIFACT_DIR
         self.target_column = training_pipeline.TARGET_COLUMN
         self.churn_prediction_window_days = (
            training_pipeline.CHURN_PREDICTION_WINDOW_DAYS
        )
         self.artifact_dir = os.path.join(
            self.artifact_name,
            self.timestamp,
        )
         self.model_dir = os.path.join(training_pipeline.SAVED_MODEL_DIR)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )

        self.feature_store_dir = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
        )

        self.ingested_data_dir = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
        )

        self.training_file_path = os.path.join(
            self.ingested_data_dir,
            training_pipeline.TRAIN_FILE_NAME,
        )

        self.testing_file_path = os.path.join(
            self.ingested_data_dir,
            training_pipeline.TEST_FILE_NAME,
        )

        self.database_name = training_pipeline.DATA_INGESTION_DATABASE_NAME
        self.target_column = training_pipeline.TARGET_COLUMN
        self.churn_prediction_window_days = (
            training_pipeline.CHURN_PREDICTION_WINDOW_DAYS
        )

        # Key is the simple dataset name.
        # Value is the MongoDB collection name.
        self.collection_names = (
            training_pipeline.DATA_INGESTION_COLLECTION_NAMES
        )

        self.feature_store_file_paths = {
            dataset_name: os.path.join(
                self.feature_store_dir,
                f"{dataset_name}.csv",
            )
            for dataset_name in self.collection_names
        }

        self.train_test_split_ratio = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
