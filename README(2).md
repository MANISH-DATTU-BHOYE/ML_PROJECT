
# Used_Car_Price_Prediction
Hello 👋, I’m Manish, A Computer Science student exploring Machine Learning and Data Science.
This repository is dedicated to building and documenting an end-to-end Used Car Price Prediction system using real-world data.

### Aim of the Project

 Aim of this project is to design and develop an end-to-end Machine Learning system that can accurately predict the asking price of a used car based on historical data and key vehicle attributes.


### Contents

1)Dataset/ – Used car dataset (CSV)

2)notebooks/

    Exploratory Data Analysis (EDA)

    Model training and evaluation

3)src/ – Modular ML pipeline

    Data ingestion

    Data transformation

    Model training

    Prediction pipeline

4)artifacts/ – Saved models and preprocessors

5)application.py – Flask-based web application

6)templates/ – HTML files for UI

7)Cloud deployment using AWS Elastic Beanstalk

8)README.md – Project documentation

### Project Covers

Data cleaning (currency symbols, units like km)

Feature engineering for categorical and numerical data

Training and comparing multiple regression models

Selecting the best-performing model

Deploying the model using a Flask web application

### Deployment on AWS Elastic Beanstalk (Using GitHub Repository)
This project is deployed on Amazon Web Services (AWS) using Elastic Beanstalk, with the application source code hosted on a GitHub repository.

Elastic Beanstalk manages the complete infrastructure, including EC2 instance creation, dependency installation, application deployment, scaling, and monitoring.

Deployment Workflow

1)The complete project is pushed to a GitHub repository

2)AWS Elastic Beanstalk is configured with Python platform

3)GitHub repository is connected as the source code provider

4)Elastic Beanstalk automatically:

    Pulls the latest code from GitHub

    Installs dependencies from requirements.txt

    Runs the Flask application

    Exposes a public application URL

### Output

Deployed an end-to-end Used Car Price Prediction ML application on AWS Elastic Beanstalk using a GitHub-based deployment workflow.

The system delivers real-time price predictions via a scalable Flask web application, following production-ready ML practices.