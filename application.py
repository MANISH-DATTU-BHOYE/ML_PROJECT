from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = CustomData(
            Brand=request.form["Brand"],
            model=request.form["model"],
            Transmission=request.form["Transmission"],
            Owner=request.form["Owner"],
            FuelType=request.form["FuelType"],
            Year=int(request.form["Year"]),
            Age=int(request.form["Age"]),
            kmDriven=int(request.form["kmDriven"]),
        )

        df = data.get_data_as_dataframe()
        prediction = PredictPipeline().predict(df)[0]

        return render_template(
            "index.html",
            result=round(prediction, 2)
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
