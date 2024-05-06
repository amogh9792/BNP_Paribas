
# Common Constant
TARGET_COLUMN = 'target'
TRAIN_PIPELINE_NAME = 'train'
ARTIFACT_DIR = "artifact"

TRAIN_FILE_NAME = 'train.csv'
TEST_FILE_NAME = 'test.csv'

MONGODB_KEY = "MONGODB_KEY"
DATABASE_NAME = "BNP-Paribas"

# Data Ingestion Constant

TRAIN_DI_COLLECTION_NAME = "Train"
DI_DIR_NAME = "data_ingestion"
TRAIN_TEST_SPLIT_RATIO = 0.2
DI_FEATURE_STORE_DIR = "feature_store"
DI_INGESTED_DIR = 'ingested'
DI_DROP_COL_IN_CLEAN = ['ID']

DI_MANDATORY_COLUMN_LIST = ['target', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v14', 'v15', 'v16', 'v18', 'v19', 'v20', 'v22', 'v23', 'v24', 'v26', 'v30', 'v31', 'v33', 'v34', 'v35', 'v36', 'v38', 'v39', 'v42', 'v45', 'v47', 'v50', 'v52', 'v56', 'v57', 'v58', 'v62', 'v66', 'v69', 'v70', 'v71', 'v72', 'v74', 'v75', 'v79', 'v80', 'v82', 'v88', 'v90', 'v91', 'v97', 'v99', 'v102', 'v107', 'v110', 'v112', 'v113', 'v120', 'v125', 'v129', 'v131']
DI_MANDATORY_COLUMN_DATA_TYPE = {'target': 'int64', 'v1': 'float64', 'v2': 'float64', 'v3': 'object', 'v4': 'float64', 'v5': 'float64', 'v6': 'float64', 'v7': 'float64', 'v8': 'float64', 'v9': 'float64', 'v10': 'float64', 'v11': 'float64', 'v13': 'float64', 'v14': 'float64', 'v15': 'float64', 'v16': 'float64', 'v18': 'float64', 'v19': 'float64', 'v20': 'float64', 'v22': 'object', 'v23': 'float64', 'v24': 'object', 'v26': 'float64', 'v30': 'object', 'v31': 'object', 'v33': 'float64', 'v34': 'float64', 'v35': 'float64', 'v36': 'float64', 'v38': 'int64', 'v39': 'float64', 'v42': 'float64', 'v45': 'float64', 'v47': 'object', 'v50': 'float64', 'v52': 'object', 'v56': 'object', 'v57': 'float64', 'v58': 'float64', 'v62': 'int64', 'v66': 'object', 'v69': 'float64', 'v70': 'float64', 'v71': 'object', 'v72': 'int64', 'v74': 'object', 'v75': 'object', 'v79': 'object', 'v80': 'float64', 'v82': 'float64', 'v88': 'float64', 'v90': 'float64', 'v91': 'object', 'v97': 'float64', 'v99': 'float64', 'v102': 'float64', 'v107': 'object', 'v110': 'object', 'v112': 'object', 'v113': 'object', 'v120': 'float64', 'v125': 'object', 'v129': 'int64', 'v131': 'float64'}

# Data Validation Constant

DV_IMPUTATION_VALUES_FILE_NAME = 'imputation_values.csv'
DV_IMPUTATION_VALUES_FILE_PATH = 'source/ml'
DV_OUTLIER_PARAMS_FILE = "outlier_details.csv"

DV_DIR_NAME = "data_validation"

# Data Transformation Constant

DT_DIR_NAME: str = "data_transformation"
DT_ENCODER_NAME = 'multi_class_encoder.pkl'

MULTI_CLASS_COLUMNS = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v14', 'v15', 'v16', 'v18', 'v19',
                        'v20', 'v22', 'v23', 'v24', 'v26', 'v30', 'v31', 'v33', 'v34', 'v35', 'v36', 'v38', 'v39', 'v42', 'v45', 'v47', 'v50',
                        'v52', 'v56', 'v57', 'v58', 'v62', 'v66', 'v69', 'v70', 'v71', 'v72', 'v74', 'v75', 'v79', 'v80', 'v82', 'v88', 'v90',
                        'v91', 'v97', 'v99', 'v102', 'v107', 'v110', 'v112', 'v113', 'v120', 'v125', 'v129', 'v131']
BINARY_COLUMN = ['target']

SCALER_DETAILS = 'scaler_details.csv'

# Model Train And Evaluation

MODELS_DIR_NAME = 'artifact'
FINAL_MODEL_DIR_NAME = 'final_model'
MODEL_EVALUATION_REPORT = "model_evaluation_report.csv"

# Model Predict
PREDICT_DI_COLLECTION_NAME = "Predict"
PREDICT_PIPELINE_NAME = 'predict'
PREDICT_DATA_FILE_NAME = 'predict.csv'
PREDICT_FILE = 'predict.csv'
FINAL_MODEL_FILE_NAME = 'SVC.pkl'
MP_DIR_NAME = "model_prediction"


