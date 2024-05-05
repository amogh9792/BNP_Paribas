import os
from source.constant import constant

class PipelineConfig:
    def __init__(self, global_timestamp):

        self.artifact_dir = os.path.join(constant.ARTIFACT_DIR, global_timestamp)
        self.global_timestamp = global_timestamp
        self.target_column = constant.TARGET_COLUMN
        self.train_pipeline = constant.TRAIN_PIPELINE_NAME
        self.train_file_name = constant.TRAIN_FILE_NAME
        self.test_file_name = constant.TEST_FILE_NAME

        # Data Ingestion Constant

        self.train_di_dir = os.path.join(self.artifact_dir, constant.DI_DIR_NAME)
        self.train_test_split_ratio = constant.TRAIN_TEST_SPLIT_RATIO

        self.train_feature_store_dir_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DI_DIR_NAME, constant.DI_FEATURE_STORE_DIR)
        self.train_feature_store_file_name = constant.TRAIN_FILE_NAME
        self.test_feature_store_file_name = constant.TEST_FILE_NAME
        self.train_di_train_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DI_DIR_NAME,constant.DI_INGESTED_DIR)
        self.train_di_test_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DI_DIR_NAME,constant.DI_INGESTED_DIR)
        self.di_col_drop_in_clean = constant.DI_DROP_COL_IN_CLEAN

        self.mongodb_url_key = os.environ[constant.MONGODB_KEY]
        self.database_name = constant.DATABASE_NAME
        self.train_collection_name = constant.TRAIN_DI_COLLECTION_NAME
        self.mandatory_col_list = constant.DI_MANDATORY_COLUMN_LIST
        self.mandatory_col_data_type = constant.DI_MANDATORY_COLUMN_DATA_TYPE

        # Data Validation
        self.imputation_values_file_path = constant.DV_IMPUTATION_VALUES_FILE_PATH
        self.ml_path = os.path.join(os.getcwd(), constant.DV_IMPUTATION_VALUES_FILE_PATH)
        os.makedirs(self.ml_path, exist_ok = True)
        self.init_file_path = os.path.join(self.ml_path, "__init__.py")
        open(self.init_file_path, "a").close()
        self.imputation_values_file_name = os.path.join(self.ml_path, constant.DV_IMPUTATION_VALUES_FILE_NAME)
        self.outlier_params_file = os.path.join(self.ml_path, constant.DV_OUTLIER_PARAMS_FILE)

        self.train_dv_train_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DV_DIR_NAME)
        self.train_dv_test_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DV_DIR_NAME)


        # Data Transformation

        self.dt_binary_class_col = constant.BINARY_COLUMN
        self.dt_multi_class_col = constant.MULTI_CLASS_COLUMNS
        self.dt_multi_class_encoder_name = constant.DT_ENCODER_NAME
        self.dt_multi_class_encoder_path = os.path.join(self.ml_path, self.dt_multi_class_encoder_name)
        self.train_dt_train_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DT_DIR_NAME)
        self.train_dt_test_file_path = os.path.join(self.artifact_dir, self.train_pipeline, constant.DT_DIR_NAME )
        self.scaler_details_path = os.path.join(self.ml_path, constant.SCALER_DETAILS)

        # Model Train And Evaluation

        self.model_path = os.path.join(self.ml_path, constant.MODELS_DIR_NAME)
        os.makedirs(self.model_path, exist_ok=True)
        self.final_model_path = os.path.join(self.ml_path, constant.FINAL_MODEL_DIR_NAME)
        os.makedirs(self.final_model_path, exist_ok=True)
        self.final_model_file_name = constant.FINAL_MODEL_FILE_NAME
        self.model_evaluation_report = os.path.join(self.ml_path, constant.MODEL_EVALUATION_REPORT)

        # Model Prediction

        self.predict_collection_name = constant.PREDICT_DI_COLLECTION_NAME
        self.predict_di_dir = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME, constant.DI_DIR_NAME)
        self.predict_di_feature_store_file_name = os.path.join(self.predict_di_dir, constant.DI_FEATURE_STORE_DIR)
        self.predict_file = constant.PREDICT_FILE
        self.predict_data_file_name = constant.PREDICT_DATA_FILE_NAME

        self.predict_file_path = os.path.join(self.predict_di_dir, constant.DI_INGESTED_DIR)
        self.predict_dv_file_path = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME, constant.DV_DIR_NAME)
        self.predict_dt_file_path = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME,constant.DT_DIR_NAME)
        self.predict_mp_file_path = os.path.join(self.artifact_dir, constant.PREDICT_PIPELINE_NAME, constant.MP_DIR_NAME)

