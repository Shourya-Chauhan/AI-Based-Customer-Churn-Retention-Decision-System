"""Constants used in the customer churn training pipeline."""

# Common pipeline constants
PIPELINE_NAME: str = "CustomerChurn"
ARTIFACT_DIR: str = "artifacts"
SAVED_MODEL_DIR: str = "final_model"

# Target column created after customers and orders are joined
TARGET_COLUMN: str = "is_churned"
CHURN_PREDICTION_WINDOW_DAYS: int = 90

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"


# Data ingestion constants
DATA_INGESTION_DATABASE_NAME: str = "CustomerChurnDB"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# MongoDB collections used by the project
DATA_INGESTION_COLLECTION_NAMES: dict = {
    "customers": "olist_customers_dataset",
    "orders": "olist_orders_dataset",
    "order_items": "olist_order_items_dataset",
    "payments": "olist_order_payments_dataset",
    "reviews": "olist_order_reviews_dataset",
    "products": "olist_products_dataset",
    "sellers": "olist_sellers_dataset",
    "category_translation": "product_category_name_translation",
}
