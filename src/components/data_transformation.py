import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path = "artifacts/preprocessor.pkl"


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def clean_data(self, df):
        df["kmDriven"] = (
            df["kmDriven"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" km", "", regex=False)
        )
        df["kmDriven"] = pd.to_numeric(df["kmDriven"], errors="coerce")

        df["AskPrice"] = (
            df["AskPrice"].astype(str)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["AskPrice"] = pd.to_numeric(df["AskPrice"], errors="coerce")

        return df

    def get_preprocessor(self):
        cat_cols = ["Brand", "model", "Transmission", "Owner", "FuelType"]
        num_cols = ["Year", "Age", "kmDriven"]

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("scaler", StandardScaler(with_mean=False))
        ])

        return ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ])

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = self.clean_data(pd.read_csv(train_path))
            test_df = self.clean_data(pd.read_csv(test_path))

            X_train = train_df.drop(columns=["AskPrice", "PostedDate", "AdditionInfo"])
            y_train = train_df["AskPrice"]

            X_test = test_df.drop(columns=["AskPrice", "PostedDate", "AdditionInfo"])
            y_test = test_df["AskPrice"]

            preprocessor = self.get_preprocessor()
            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            save_object(self.config.preprocessor_path, preprocessor)

            return X_train_arr, X_test_arr, y_train.values, y_test.values

        except Exception as e:
            raise CustomException(e, sys)
