# BNP Paribas Cardif Claims Prediction Project

## Project Overview
This project aims to accelerate the claims approval process for BNP Paribas Cardif by predicting the category of claims based on features available early in the process. By automating this prediction, BNP Paribas Cardif can potentially speed up the claims process and provide better service to its customers.

## Project Structure
The project is structured as follows:

- `data/`: Contains the dataset provided by BNP Paribas Cardif.
- `src/`: Contains the source code for the project.
  - `data_ingestion.py`: Module for ingesting the dataset.
  - `data_validation.py`: Module for validating the dataset.
  - `data_transformation.py`: Module for transforming and preprocessing the dataset.
  - `model_train_evaluate.py`: Module for training, evaluating, and selecting machine learning models.
  - `model_prediction.py`: Module for using trained models to make predictions.
- `models/`: Contains trained machine learning models.
- `results/`: Contains evaluation results and predictions.
- `README.md`: This file.

## Implemented Solution
- **Model Used:** Support Vector Classifier (SVC)
- **Experimentation:** Experimented with multiple machine learning models to identify the most suitable one for the task.
- **Performance Metrics:** Evaluated model performance using appropriate metrics such as accuracy, precision, recall, and F1-score.
- **Hyperparameter Tuning:** Tuned the hyperparameters of the SVC model to optimize its performance.

## Usage
To replicate the project or use the provided functionalities, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Run the necessary modules/scripts in the `src/` directory.
4. Access the stored results and predictions in MongoDB.
