import logging
import pickle
from source.exception import BnpException
from source.utility.utility import export_csv_file, import_csv_file

class ModelPrediction:

    def __init__(self, utility_config):
        self.utility_config = utility_config

    def load_model_pickle(self):
        try:
            with open(self.utility_config.final_model_path + '\\' + self.utility_config.final_model_file_name, 'rb') as file:

                return pickle.load(file)

        except BnpException as e:
            raise e

    def make_prediction(self, model, data):
        try:

            return model.predict(data)

        except BnpException as e:
            raise e

    def initiate_model_prediction(self):

        logging.info("Start: Model Prediction")

        predict_data = import_csv_file(self.utility_config.predict_file, self.utility_config.predict_dt_file_path)
        predict_data = predict_data.drop('target', axis = 1)

        model = self.load_model_pickle()

        feature_data = import_csv_file(self.utility_config.predict_data_file_name, self.utility_config.predict_di_feature_store_file_path)

        feature_data['target'] = self.make_prediction(model, predict_data)

        export_csv_file(feature_data, self.utility_config.predict_file, self.utility_config.predict_mp_file_path)

        print("MODEL PREDICTION DONE")

        logging.info("Complete: Model Prediction")