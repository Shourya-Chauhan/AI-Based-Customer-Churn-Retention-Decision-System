import os
import sys
import json
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv
from customerchurn.exception.exception import CustomerChurnException

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class CustomerDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]
            
            # Avoid re-inserting into a collection that already has data
            if self.collection.count_documents({}) == 0:
                self.collection.insert_many(records)
                print(f"Inserted {len(records)} records into collection: {collection}")
            else:
                print(f"Collection {collection} already contains data. Skipped.")
                
            return len(records)
        except Exception as e:
            raise CustomerChurnException(e, sys)

if __name__ == "__main__":
    # Choose your database name here
    DATABASE = "CustomerChurnDB"
    
    # Path where all raw CSV files are stored
    RAW_DATA_DIR = os.path.join("Data", "raw")
    
    extractor = CustomerDataExtract()
    
    # Loop over every CSV in the Data/raw directory
    for file_name in os.listdir(RAW_DATA_DIR):
        if file_name.endswith(".csv"):
            file_path = os.path.join(RAW_DATA_DIR, file_name)
            
            # Use the filename (without .csv) as the collection name
            collection_name = os.path.splitext(file_name)[0]
            
            print(f"Processing: {file_name} -> Collection: {collection_name}")
            records = extractor.csv_to_json_convertor(file_path=file_path)
            extractor.insert_data_mongodb(records=records, database=DATABASE, collection=collection_name)