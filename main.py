from customerchurn.components.data_ingestion import DataIngestion
from customerchurn.exception.exception import CustomerChurnException
from customerchurn.logging.logger import logging
from customerchurn.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
)

import sys


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config
        )

        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating data ingestion")

        data_ingestion_artifact = (
            data_ingestion.initiate_data_ingestion()
        )
        print(data_ingestion_artifact)

    except Exception as e:
        raise CustomerChurnException(e, sys)
