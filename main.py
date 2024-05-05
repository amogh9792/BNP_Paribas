import pandas as pd
from source.utility.utility import generate_global_timestamp
from source.entity.config_entity import PipelineConfig
from source.logger import setup_logger
from source.logger import logging
from source.pipeline.pipeline import DataPipeline


global_timestamp = generate_global_timestamp()

setup_logger(global_timestamp)
logging.info("Logger Timestamp Setup Complete")

train_pipeline_config_obj = PipelineConfig(global_timestamp)
# print(train_pipeline_config_obj.__dict__)

logging.info("Training Pipeline Config Created..")

pipeline_obj = DataPipeline(global_timestamp)
pipeline_obj.run_train_pipeline()
# pipeline_obj.run_predict_pipeline()

print('Done...')