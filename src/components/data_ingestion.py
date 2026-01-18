import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"
    raw_path = "artifacts/raw.csv"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv("used_car_dataset.csv")

            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.config.raw_path, index=False)

            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_df.to_csv(self.config.train_path, index=False)
            test_df.to_csv(self.config.test_path, index=False)

            return self.config.train_path, self.config.test_path

        except Exception as e:
            raise CustomException(e, sys)
