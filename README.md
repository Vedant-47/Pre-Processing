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


## Dataset Overview
This project processes a car dataset and generates three main outputs:

1. Processed Data: The full dataset after all cleaning and preprocessing steps.
2. Train Data: A subset of the processed data for training machine learning models.
3. Test Data: A subset of the processed data for testing machine learning models.

Note: These datasets are provided in the attached zip file. They demonstrate the results of the data processing pipeline and can be used for further analysis or model training.

## Installation
To use this project, follow these steps:

1. Ensure you have Python 3.7+ installed on your system.
2. Clone this repository to your local machine.
3. Install the required dependencies:
   - pymongo
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - category_encoders
   - scipy
4. Install MongoDB on your system if not already present. Follow the official MongoDB installation guide for your operating system.
5. Start the MongoDB service on your local machine.
6. Update the MongoDB connection string in the code if necessary (default is "mongodb://localhost:27017/").

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
