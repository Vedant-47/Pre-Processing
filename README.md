# Data Pre-Processing Pipeline

## Project Overview
This project is a web-based machine learning app built with Dash, scikit-learn, and Plotly.
It provides an end-to-end workflow for:

~ Loading data (from CSV, MySQL, or MongoDB)

~ Cleaning and preprocessing data

~ Performing Exploratory Data Analysis (EDA) through interactive histograms

~ Training a Random Forest Classifier on user-selected target columns

~ Making real-time predictions with a simple input form

The app is designed for data analysts, students, and ML beginners who want to quickly train models and visualize datasets without writing code.

## Features
    Multiple Data Sources
     ~ Upload CSV files
     ~ Load from MySQL databases
     ~ Load from MongoDB collections
    Preprocessing
      ~ Handles missing values
      ~ Standardizes features with StandardScaler
    Interactive EDA
      ~ Select any column to generate histograms using Plotly
    Model Training
      ~ Train a Random Forest Classifier with one click
      ~ Displays model accuracy on test data
    Prediction
      ~ Dynamically generates input fields for features
      ~ Provides predictions on new user inputs


## Project Structure
      Data-PreProcessing-Pipeline/
      │── app.py              # Main Dash application (this file)
      │── model.pkl           # Trained model + scaler (auto-saved after training)
      │── requirements.txt    # Python dependencies
      │── README.md           # Project documentation




## Author
Vedant Gandhi (Data Science Student @ Navrachana University)
