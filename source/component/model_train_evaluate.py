import pickle
import pandas as pd
import warnings
from source.logger import logging
from source.exception import BnpException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

class ModelTrainEvaluate:
    def __init__(self, utility_config):
        self.utility_config = utility_config

        self.models = {
            "LogisticRegression": LogisticRegression(),
            "SVC": SVC(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "GaussianNB": GaussianNB(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "XGBClassifier": XGBClassifier()
        }

        self.model_evaluation_report = pd.DataFrame(columns = ["model_name", "accuracy", "precision", "recall", "f1", "class_report", "confu_matrix"])

    def hyper_parameter_tuning(self, x_train, y_train):
        try:
            model = SVC()

            param_grid = {
                'C': [0.1, 1, 10],  # Regularization parameter
                'kernel': ['linear', 'rbf'],  # Kernel type
                'gamma': ['scale', 'auto']  # Kernel coefficient
            }

            f1_scorer = make_scorer(f1_score, average='macro')
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=f1_scorer)

            grid_search.fit(x_train, y_train)

            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            return best_params, best_score

        except BnpException as e:
            raise e
    def model_training(self, train_data, test_data):
        try:
            x_train = train_data.drop('target', axis = 1)
            y_train = train_data['target']
            test_data = test_data[:-76]
            x_test = test_data.drop('target', axis = 1)
            y_test = test_data['target']

            for name, model in self.models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                with open(f"{self.utility_config.model_path}/{name}.pkl", "wb") as f:
                    pickle.dump(model, f)

                self.metrics_and_log(y_test, y_pred, name)

        except BnpException as e:
            raise e

    def metrics_and_log(self, y_test, y_pred, model_name):
        try:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            confu_matrix = confusion_matrix(y_test, y_pred)

            logging.info(
                f"model: {model_name}, accuracy:{accuracy}, precision:{precision}, recall:{recall}, f1_score: {f1}, classification_report:{class_report}, confusion matrix:{confu_matrix}")
            new_row = [model_name, accuracy, precision, recall, f1, class_report, confu_matrix]
            self.model_evaluation_report = self.model_evaluation_report._append(pd.Series(new_row, index=self.model_evaluation_report.columns), ignore_index=True)

        except BnpException as e:
            print(e)
            raise e

    def retrain_final_model(self, train_data, test_data):
        try:
            x_train = train_data.drop('target', axis = 1)
            y_train = train_data['target']
            x_test = test_data.drop('target', axis = 1)
            y_test = test_data['target']

            best_params, best_score = self.hyper_parameter_tuning(x_train, y_train)

            final_model = SVC(**best_params)
            final_model_name = 'SVC'

            final_model.fit(x_train, y_train)

            test_score = final_model.score(x_test, y_test)

            logging.info(f"final model: SVC, test score: {test_score}")

            with open(f"{self.utility_config.final_model_path}/{final_model_name}.pkl", "wb") as f:
                pickle.dump(final_model, f)

        except BnpException as e:
            raise e

    def initiate_model_training(self):
        try:

            logging.info("Start: Model Training And Evaluation")

            train_data = pd.read_csv(self.utility_config.train_dt_train_file_path + '/' + self.utility_config.train_file_name)
            test_data = pd.read_csv(self.utility_config.train_dt_test_file_path+'/'+self.utility_config.test_file_name)

            self.model_training(train_data, test_data)
            self.model_evaluation_report.to_csv(self.utility_config.model_evaluation_report, index = False)

            self.retrain_final_model(train_data, test_data)

            print('>>>>>>>> Model Train Done <<<<<<<<')

            logging.info("Complete: Model Training And Evaluation")

        except BnpException as e:
            raise e