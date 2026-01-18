import pandas as pd
from src.utils import load_object


class PredictPipeline:
    def predict(self, df):
        preprocessor = load_object("artifacts/preprocessor.pkl")
        model = load_object("artifacts/model.pkl")

        data_transformed = preprocessor.transform(df)
        return model.predict(data_transformed)


class CustomData:
    def __init__(
        self,
        Brand,
        model,
        Transmission,
        Owner,
        FuelType,
        Year,
        Age,
        kmDriven
    ):
        self.data = {
            "Brand": [Brand],
            "model": [model],
            "Transmission": [Transmission],
            "Owner": [Owner],
            "FuelType": [FuelType],
            "Year": [Year],
            "Age": [Age],
            "kmDriven": [kmDriven],
        }

    def get_data_as_dataframe(self):
        return pd.DataFrame(self.data)
