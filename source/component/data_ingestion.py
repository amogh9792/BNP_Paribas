import os
import pandas as pd
from pandas import DataFrame
from source.exception import BnpException
from pymongo.mongo_client import MongoClient
from sklearn.model_selection import train_test_split
from source.utility.utility import import_csv_file
from source.utility.utility import export_csv_file
from source.logger import logging

class DataIngestion:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def export_data_into_feature_store(self, key) -> DataFrame:
        """
        Export data from MongoDB into a feature store.

        Parameters:
        key (str): Indicates the type of data to export ('train' or other).

        Returns:
        DataFrame: The exported data as a DataFrame.

        Raises:
        BnpException: If an error occurs during export.
        """

        try:
            logging.info("Start: Data Load From MongoDB")

            if key == 'train':
                collection_name = self.utility_config.train_collection_name
                feature_store_file_path = self.utility_config.train_feature_store_dir_path
                feature_store_file_name = self.utility_config.train_feature_store_file_name
            else:
                # collection_name = self.utility_config.predict_collection_name
                # feature_store_file_path = self.utility_config.predict_di_feature_store_file_path
                # feature_store_file_name = self.utility_config.predict_di_feature_store_file_name

                data = pd.read_csv('BNP EDA/predict.csv')

                return data

            client = MongoClient(self.utility_config.mongodb_url_key)
            database = client[self.utility_config.database_name]
            train_collection = database[collection_name]

            cursor = train_collection.find()

            data = pd.DataFrame(list(cursor))

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok = True)

            export_csv_file(data, feature_store_file_name, feature_store_file_path)

            return data

            logging.info("Complete: Data Load From MongoDB")

        except BnpException as e:
            raise e

    def split_train_test(self, data: DataFrame):
        try:
            train_data, test_data = train_test_split(data, test_size = self.utility_config.train_test_split_ratio)

            return train_data, test_data

        except BnpException as e:
            raise e


    def clean_data(self, data, key):
        """
        Clean data based on the specified key.

        Parameters:
        data (DataFrame): The input data to be cleaned.
        key (str): Indicates the type of data ('train' or 'test').

        Returns:
        DataFrame: The cleaned data.

        Raises:
        BnpException: If an error occurs during the cleaning process.
        """
        try:

            logging.info("start: clean data")

            if key == 'train':
                data = data.drop_duplicates()
                data = data.drop(self.utility_config.di_col_drop_in_clean, axis = 1)

                data = data.loc[:, data.nunique() > 1]

                drop_column = []

                for col in data.select_dtypes(include=['object']).columns:
                    unique_count = data[col].nunique()

                    if unique_count / len(data) > 0.5:
                        data.drop(col, axis=1, inplace=True)
                        drop_column.append(col)

                logging.info(f"dropped columns: {drop_column}")

            logging.info("complete: clean data")

            return data

        except BnpException as e:
            raise e

    def process_data(self, data, key):
        """
        Process data based on the specified key.

        Parameters:
            data (DataFrame): The input data to be processed.
            key (str): Indicates the type of data ('train', 'test', or 'predict').

        Returns:
            DataFrame: The processed data.

        Raises:
            Exception: If a mandatory column is missing.
            ValueError: If there's an error converting column data type.
        """

        try:
            logging.info("start: process data")

            if key == 'train':
                mandatory_cols = self.utility_config.mandatory_col_list.copy()

            if key in ['predict', 'test']:

                mandatory_cols = self.utility_config.mandatory_col_list.copy()

                mandatory_cols.remove('target')

                data = data.drop(self.utility_config.di_col_drop_in_clean, axis=1)

            for col in mandatory_cols:

                if col not in data.columns:
                    raise Exception(f"missing mandatory column: {col}")

                if data[col].dtype != self.utility_config.mandatory_col_data_type[col]:
                    try:
                        data[col] = data[col].astype(self.utility_config.mandatory_col_data_type[col])
                    except ValueError as e:
                        raise Exception(f"ERROR: converting data type for column: {col}")

            data = data[mandatory_cols]

            logging.info("complete: process data")

            return data

        except BnpException as e:
            raise e

    def initiate_data_ingestion(self, key):

        """
        Initiate data ingestion process.

        Parameters:
            key (str): Indicates the type of data ('train', 'test', or 'predict').

        Raises:
            BnpException: If an error occurs during data ingestion.
        """

        try:
            logging.info(">>>>>> INITIATE DATA INGESTION <<<<<<")

            data = self.export_data_into_feature_store(key)
            data = self.clean_data(data, key)
            data = self.process_data(data, key)

            if key == 'train':
                train_data, test_data = self.split_train_test(data)
                export_csv_file(train_data, self.utility_config.train_file_name, self.utility_config.train_di_train_file_path)
                export_csv_file(test_data, self.utility_config.test_file_name, self.utility_config.train_di_test_file_path)

            if key == 'predict':

                export_csv_file(data, self.utility_config.predict_file, self.utility_config.predict_file_path)

            logging.info(">>>>>>>> COMPLETE DATA INGESTION <<<<<<<<<")

        except BnpException as e:
            raise e