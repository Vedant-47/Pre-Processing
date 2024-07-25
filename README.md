# Data Pre-Processing

## Project Overview
This project provides a Python class for processing and analyzing data. It includes functionality for data cleaning, handling missing values, encoding categorical variables, detecting outliers, and splitting datasets for machine learning tasks.

## Features
- MongoDB integration for data input and output
- Data cleaning and preprocessing
- Handling of null values
- Removal of duplicate entries
- Categorical data encoding (Label, One-Hot, Binary, Frequency)
- Outlier detection and handling (Z-Score, IQR, Percentile methods)
- Dataset splitting for machine learning
- Output to CSV or MongoDB

## Dependencies
- pymongo
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- category_encoders
- scipy

## Usage
The `CarDataProcessor` class provides an interactive interface for data processing. Users can:
1. Connect to a MongoDB database
2. Choose columns to drop
3. Handle null values
4. Remove duplicates
5. Encode categorical variables
6. Detect and handle outliers
7. Split the dataset into training and testing sets
8. Output processed data to CSV or MongoDB

## Key Methods
- `dropping_values()`: Allows user to drop specified columns
- `handle_null_values()`: Handles null values using mode, mean, or removal
- `remove_duplicates()`: Removes duplicate entries from the dataset
- `handle_categorical_values()`: Encodes categorical variables
- `outlier_detection()`: Detects and handles outliers
- `split_dataset()`: Splits data into training and testing sets
- `output_file()`: Saves processed data to CSV or MongoDB

## Limitations
- The current implementation is designed for a specific car dataset structure. It may require modifications for use with different datasets.
- The code relies on user input for many operations, which may not be ideal for automated data processing pipelines.
- Error handling is basic and could be improved for more robust operation.
- The code assumes a local MongoDB instance is available and accessible.
- Some operations, like outlier detection, use fixed methods that may not be suitable for all types of data distributions.

## Future Improvements
- Enhance flexibility to handle various dataset structures
- Add options for automated processing with less user input
- Implement more sophisticated error handling and input validation
- Add support for different database types beyond MongoDB
- Implement more advanced statistical methods for data analysis

## Author
Vedant Gandhi (Data Science Student @ Navrachana University)
