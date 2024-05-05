import numpy as np
import os
import pandas as pd
import pickle
import category_encoders as ce
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from source.exception import BnpException
from source.logger import logging
from source.utility.utility import export_csv_file, import_csv_file

warnings.filterwarnings('ignore')

class DataTransformation:
    def __init__(self, utility_config):
        self.utility_config = utility_config

    def feature_encoding(self, data, save_encoder_path = None, load_encoder_path = None, key = None):
        """
        Perform feature encoding using TargetEncoder.

        Parameters:
            data (DataFrame): The input DataFrame containing the data.
            save_encoder_path (str): The file path to save the encoder (optional).
            load_encoder_path (str): The file path to load a pre-trained encoder (optional).
            key (str): The key indicating the mode of operation (optional).

        Returns:
            DataFrame: The DataFrame with feature encoding applied.

        Raises:
            Exception: If an error occurs during the feature encoding process.
        """
        try:
            if save_encoder_path:
                encoder = ce.TargetEncoder(cols = self.utility_config.dt_multi_class_col)
                target_variable = data['target']
                data_encoded = encoder.fit_transform(data[self.utility_config.dt_multi_class_col], target_variable)

                with open(save_encoder_path, 'wb') as f:
                    pickle.dump(encoder, f)

            if load_encoder_path:
                with open(load_encoder_path, 'rb') as f:
                    encoder = pickle.load(f)

                data_encoded = encoder.transform(data[self.utility_config.dt_multi_class_col])

            data = pd.concat([data.drop(columns = self.utility_config.dt_multi_class_col), data_encoded], axis = 1)

            return data

        except BnpException as e:
            raise e

    def min_max_scaler(self, data, key = None):
        """
        Applies Min-Max scaling to numerical features in the dataset.

        Parameters:
        - data (DataFrame): The input dataset containing numerical features to be scaled.
        - key (str): Specifies the mode of operation. If 'train', fits the scaler to the data and saves scaler details.
                     If None or any other value, applies scaling using pre-computed scaler details.

        Returns:
        - data (DataFrame): The dataset with numerical features scaled using Min-Max scaling.

        Notes:
        - The 'target' column is assumed to be present in the dataset and is not scaled.
        - For 'train' mode, scaler details are saved to a CSV file specified by self.utility_config.scaler_details_path.
        - For non-'train' mode, scaler details are loaded from the saved CSV file.

        """

        if key == 'train':

            numeric_columns = data.select_dtypes(include = ['float64', 'int64', 'int32']).columns

            scaler = MinMaxScaler()

            scaler.fit(data[numeric_columns])

            scaler_details = pd.DataFrame({'Feature' : numeric_columns,
                                           'Scaler_min': scaler.data_min_,
                                           'Scaler_max': scaler.data_max_})

            scaler_details.to_csv(self.utility_config.scaler_details_path, index = False)

            scaled_data = scaler.transform(data[numeric_columns])

            data.loc[:, numeric_columns] = scaled_data
            data['target'] = self.utility_config.target_column

        else:

            scaler_details = pd.read_csv(self.utility_config.scaler_details_path)

            for col in data.select_dtypes(include=['float64', 'int64', 'int32']).columns:
                data[col] = data[col].astype('float')

                temp = scaler_details[scaler_details['Feature'] == col]

                if not temp.empty:

                    min = temp.loc[temp.index[0], 'Scaler_min']
                    max = temp.loc[temp.index[0], 'Scaler_max']

                    data[col] = (data[col] - min) / (max - min)

                else:
                    print(f"No Scaling details available for feature: {col}")

            data['target'] = self.utility_config.target_column

        return data

    def oversample_smote(self, data):
        """
           Performs oversampling using the Synthetic Minority Over-sampling Technique (SMOTE).

           Parameters:
           - data (DataFrame): The input dataset containing features and the target variable.

           Returns:
           - DataFrame: The oversampled dataset with balanced classes.

           Raises:
           - BnpException: If an error occurs during the oversampling process.

           Notes:
           - SMOTE is used to create synthetic samples for the minority class to balance the class distribution.
           - Assumes the target variable is binary (0 and 1).
           """

        try:

            np.random.seed(42)

            x = data.drop(columns=['target'])
            y = data['target']

            smote = SMOTE()

            x_resampled, y_resampled = smote.fit_resample(x, y)

            return pd.concat([pd.DataFrame(x_resampled, columns=x.columns), pd.DataFrame(y_resampled, columns=['target'])],axis=1)

        except BnpException as e:
            raise e


    def initiate_data_transformation(self, key):
        """
            Initiates the data transformation process for training or testing datasets.

            Parameters:
            - key (str): Specifies the mode of operation. 'train' for training data transformation,
                         'test' for testing data transformation.
         """

        if key == 'train':

            logging.info("Start: Data Transformation For Training")

            train_data = import_csv_file(self.utility_config.train_file_name, self.utility_config.train_dv_train_file_path)
            test_data = import_csv_file(self.utility_config.test_file_name, self.utility_config.train_dv_test_file_path)

            train_data = self.feature_encoding(train_data, save_encoder_path = self.utility_config.dt_multi_class_encoder_path, key = 'train')
            test_data = self.feature_encoding(test_data, load_encoder_path = self.utility_config.dt_multi_class_encoder_path, key = 'test')

            self.utility_config.target_column = train_data['target']
            train_data.drop('target', axis = 1, inplace = True)
            train_data = self.min_max_scaler(train_data, key = 'train')

            self.utility_config.target_column = test_data['target']
            test_data.drop('target', axis = 1, inplace = True)
            test_data = self.min_max_scaler(test_data, key = 'test')

            train_data = self.oversample_smote(train_data)

            export_csv_file(train_data, self.utility_config.train_file_name, self.utility_config.train_dt_train_file_path)
            export_csv_file(test_data, self.utility_config.test_file_name, self.utility_config.train_dt_test_file_path)

        if key == 'predict':
            logging.info("Start: Data Transformation for prediction")

            predict_data = import_csv_file(self.utility_config.predict_file, self.utility_config.predict_dv_file_path)
            predict_data = self.feature_encoding(predict_data, load_encoder_path=self.utility_config.dt_multi_class_encoder_path, key = "predict")
            predict_data = self.min_max_scaler(predict_data, key = 'predict')

            export_csv_file(predict_data, self.utility_config.predict_file, self.utility_config.predict_dt_file_path)

            logging.info("Complete: Data Transformation")
