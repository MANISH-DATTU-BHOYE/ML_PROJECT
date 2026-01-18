import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()   

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "RandomForest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            report = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            best_model_name = max(report, key=report.get)
            best_model = models[best_model_name]

            best_model.fit(X_train, y_train)

            save_object(self.config.model_path, best_model)

            print(f" Best model saved: {best_model_name}")

        except Exception as e:
            raise CustomException(e, sys)
