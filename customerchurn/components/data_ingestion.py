import os
import sys

import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from customerchurn.entity.artifact_entity import DataIngestionArtifact
from customerchurn.entity.config_entity import DataIngestionConfig 
from customerchurn.exception.exception import CustomerChurnException
from customerchurn.logging.logger import logging


load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def export_collections_as_dataframes(self):
        try:
            mongo_db_url = os.getenv("MONGO_DB_URL")

            if not mongo_db_url:
                raise ValueError("MONGO_DB_URL is not available in the .env file")

            dataframes = {}

            with pymongo.MongoClient(
                mongo_db_url,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=30000,
            ) as mongo_client:
                mongo_client.admin.command("ping")
                database = mongo_client[
                    self.data_ingestion_config.database_name
                ]

                for dataset_name, collection_name in (
                    self.data_ingestion_config.collection_names.items()
                ):
                    collection = database[collection_name]
                    records = list(collection.find({}, {"_id": 0}))

                    if not records:
                        raise ValueError(
                            f"MongoDB collection '{collection_name}' is empty"
                        )

                    dataframe = pd.DataFrame(records)
                    dataframe.replace("na", pd.NA, inplace=True)
                    dataframes[dataset_name] = dataframe

                    logging.info(
                        f"Extracted {len(dataframe)} rows from {collection_name}"
                    )

            return dataframes

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def export_data_into_feature_store(self, dataframes):
       
        try:
            os.makedirs(
                self.data_ingestion_config.feature_store_dir,
                exist_ok=True,
            )

            for dataset_name, dataframe in dataframes.items():
                file_path = (
                    self.data_ingestion_config.feature_store_file_paths[
                        dataset_name
                    ]
                )
                dataframe.to_csv(file_path, index=False, header=True)
                logging.info(f"Saved {dataset_name} data at {file_path}")

            return self.data_ingestion_config.feature_store_file_paths

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def create_customer_churn_data(self, dataframes):
        
        try:
            customers = dataframes["customers"].copy()
            orders = dataframes["orders"].copy()

            required_customer_columns = {
                "customer_id",
                "customer_unique_id",
            }
            required_order_columns = {
                "order_id",
                "customer_id",
                "order_status",
                "order_purchase_timestamp",
            }

            if not required_customer_columns.issubset(customers.columns):
                raise ValueError("Required customer columns are missing")

            if not required_order_columns.issubset(orders.columns):
                raise ValueError("Required order columns are missing")

            orders["order_purchase_timestamp"] = pd.to_datetime(
                orders["order_purchase_timestamp"],
                errors="coerce",
            )

            # Only completed purchases are used for churn calculation.
            orders = orders[
                (orders["order_status"] == "delivered")
                & orders["order_purchase_timestamp"].notna()
            ]

            customer_orders = orders.merge(
                customers[["customer_id", "customer_unique_id"]],
                on="customer_id",
                how="inner",
            )

            if customer_orders.empty:
                raise ValueError("No customer orders were available after merging")

            target_column = self.data_ingestion_config.target_column
            prediction_days = (
                self.data_ingestion_config.churn_prediction_window_days
            )

            data_end_date = customer_orders[
                "order_purchase_timestamp"
            ].max()
            cutoff_date = data_end_date - pd.Timedelta(days=prediction_days)

            history = customer_orders[
                customer_orders["order_purchase_timestamp"] <= cutoff_date
            ]
            future = customer_orders[
                customer_orders["order_purchase_timestamp"] > cutoff_date
            ]

            customer_data = (
                history.groupby("customer_unique_id")
                .agg(
                    total_orders=("order_id", "nunique"),
                    last_purchase_timestamp=(
                        "order_purchase_timestamp",
                        "max",
                    ),
                )
                .reset_index()
            )

            future_customers = set(future["customer_unique_id"])
            customer_data[target_column] = (
                ~customer_data["customer_unique_id"].isin(future_customers)
            ).astype(int)

            customer_data["days_since_last_purchase"] = (
                cutoff_date - customer_data["last_purchase_timestamp"]
            ).dt.days

            logging.info(
                f"Created target column '{target_column}' using a "
                f"{prediction_days}-day prediction window"
            )

            return customer_data

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def split_data_as_train_test(self, dataframe):
        
        try:
            target_column = self.data_ingestion_config.target_column
            target_counts = dataframe[target_column].value_counts()

            # Stratification keeps a similar churn ratio in both files.
            stratify_column = None
            if len(target_counts) > 1 and target_counts.min() >= 2:
                stratify_column = dataframe[target_column]

            train_set, test_set = train_test_split(
                dataframe,
                test_size=(
                    self.data_ingestion_config.train_test_split_ratio
                ),
                random_state=42,
                stratify=stratify_column,
            )

            os.makedirs(
                self.data_ingestion_config.ingested_data_dir,
                exist_ok=True,
            )

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True,
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True,
            )

            logging.info("Saved the training and testing datasets")

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_data_ingestion(self):
        
        try:
            logging.info("Data ingestion started")

            dataframes = self.export_collections_as_dataframes()
            feature_store_file_paths = (
                self.export_data_into_feature_store(dataframes)
            )

            customer_data = self.create_customer_churn_data(dataframes)
            self.split_data_as_train_test(customer_data)

            data_ingestion_artifact = DataIngestionArtifact(
                training_file_path=(
                    self.data_ingestion_config.training_file_path
                ),
                testing_file_path=(
                    self.data_ingestion_config.testing_file_path
                ),
                feature_store_file_paths=feature_store_file_paths,
            )

            logging.info("Data ingestion completed")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomerChurnException(e, sys)
