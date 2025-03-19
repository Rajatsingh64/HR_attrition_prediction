from src.entity import artifact_entity, config_entity
from src.exception import ProjectException
from src.logger import logging
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from src.utils import load_numpy_array_data, save_object
import warnings
warnings.filterwarnings("ignore")
import sys


class ModelTrainer:

    def __init__(self, model_trainer_config: config_entity.ModelTrainerConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f'{">"*20} Model Trainer {"<"*20}')
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ProjectException(e, sys)

    def fine_tune_train_best_model(self, X, y):
        try:
            param_grid = {
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0]
            }

            xgb_clf = XGBClassifier(eval_metric='logloss')  # avoid warning
            logging.info(f"Hyperparameter Tuning Started")
            grid_search = GridSearchCV(
                estimator=xgb_clf,
                param_grid=param_grid,
                cv=3,
                scoring='accuracy',
                verbose=1,   # For logging
                n_jobs=-1 )
            grid_search.fit(X, y)
            logging.info(f"Best Params: {grid_search.best_params_}")
            best_model = grid_search.best_estimator_
            return best_model
        except Exception as e:
            raise ProjectException(e, sys)

    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Loading Train array and Test array")
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target features")
            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            logging.info(f"Training the model")
            model = self.fine_tune_train_best_model(X=x_train, y=y_train)

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_train, yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_test, yhat_test)

            logging.info(f"Train f1 score: {f1_train_score}, Test f1 score: {f1_test_score}")

            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model accuracy {f1_test_score} is less than expected {self.model_trainer_config.expected_score}")

            diff = abs(f1_train_score - f1_test_score)
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Model overfitting detected! Diff: {diff} exceeds threshold {self.model_trainer_config.overfiting_threshold}")

            # Save model
            logging.info(f"Saving the trained model")
            save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare artifact
            logging.info(f"Preparing model trainer artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_path=self.model_trainer_config.model_path,
                f1_train_score=f1_train_score,
                f1_test_score=f1_test_score
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise ProjectException(e, sys)
