import pymongo as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import BinaryEncoder,TargetEncoder
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler,FunctionTransformer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
import scipy.stats as stats

class CarDataProcessor:
    database_Name = input("Enter the name of the Database (Caution: It is CASE SENSITIVE !): ")
    print("")
    collection_Name = input("Enter the name of the collection (Caution: It is CASE SENSITIVE !): ")
    print("")
    def __init__(self, db_name=database_Name, collection_name=collection_Name):
        try:
            self.mongo_client = pm.MongoClient("mongodb://localhost:27017/")
            self.db = self.mongo_client[db_name]
            self.collection = self.db[collection_name]
            self.df = pd.DataFrame(list(self.collection.find({})))
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
        except pm.errors.ConnectionFailure as e:
            print("Failed to connect to MongoDB:", e)

    def fit(self):
        try:
            self.dropping_values()
            self.handle_null_values()
            self.remove_duplicates()
            self.handle_categorical_values()
            self.outlier_detection()
            self.split_dataset()
            self.output_file()
        except Exception as e:
            print("An error occurred during data processing:", e)
            self.fit()

    def dropping_values(self):
        try:
            drop = int(input("Do you want to drop any columns? ('1' for yes and '0' for no): "))
            print("")
            if drop == 1:
                while True:
                    col_drop = input("Enter 'yes' to drop a column or 'no' to stop: ")
                    print("")
                    if col_drop in ['yes','Yes','YES']:
                        column = input("Enter the column you want to drop: ")
                        print("")
                        self.df = self.df.drop(column, axis=1)
                    elif col_drop in ['no','No','NO']:
                        break
                    else:
                        print("Invalid input. Please enter 'yes' or 'no'.")
                        self.dropping_values()
            elif drop == 0:
                pass
            else:
                print("Please enter a valid input ('1' or '0')")
                self.dropping_values()
        except Exception as e:
            print("An error occurred in dropping_values method:", e)
            self.dropping_values()
            pass

    def handle_null_values(self):
        try:
            # Handling Null Values
            print("Handling Null Values...")
            missing_data = self.df.isnull()
            headers = self.df.columns
            show_null = input("Show count of NaN values? ('Yes' or 'No'): ")
            if show_null in ['yes','YES','Yes']:
                for i in headers:
                     print(missing_data[i].value_counts())
                     print("")
            elif show_null in ['no','No','NO']:
                pass
            else:
                for i in headers:
                    null_count = missing_data[i].value_counts()
                    self.handle_null_values()

            mode_mean_remove = input("What do you want to calculate 'Mode' or 'Mean' or 'Remove' ")
            object1 = input("Enter feature to calculate mode/mean: ")
            if object1 in ['Company', 'Model', 'Fuel_Type', 'Transmission','Mileage','Engine','Power','Seats']:
                if mode_mean_remove  in ['mode','Mode','MODE']:
                    dtype_change = int(input("Do you want to change data-type of the column? ('1' for yes or '0' for no)"))
                    if dtype_change == 1:
                        mode_value = self.df[object1].astype("string").mode()[0]
                        self.df[object1].fillna(mode_value, inplace=True)
                        print("Success!! NaN is filled with the mode of the respective column")
                    elif dtype_change == 0:
                        mode_value = self.df[object1].mode()[0]
                        self.df[object1].fillna(mode_value, inplace=True)
                        print("Success!! NaN is filled with the mode of the respective column")
                    else:
                        print("Enter a valid choice")
                        print("Restarting the function !")
                        self.handle_null_values()
                elif mode_mean_remove in ['mean','Mean','MEAN']:
                    dtype_change = int(input("Do you want to change data-type of the column? ('1' for yes or '0' for no)"))
                    if dtype_change == 1:
                        mode_value = self.df[object1].astype("string").mode()[0]
                        self.df[object1].fillna(mode_value, inplace=True)
                        print("Success!! NaN is filled with the mode of the respective column")
                    elif dtype_change == 0:
                        mode_value = self.df[object1].mode()[0]
                        self.df[object1].fillna(mode_value, inplace=True)
                        print("Success!! NaN is filled with the mode of the respective column")
                    else:
                        print("Enter a valid choice")
                        print("Restarting the function !")
                        self.handle_null_values()
                elif mode_mean_remove in ['remove','Remove','REMOVE']:
                    dtype_change = int(input("Do you want to change data-type of the column? ('1' for yes or '0' for no)"))
                    if dtype_change == 1:
                        mode_value = self.df[object1].astype("string").mode()[0]
                        self.df[object1].fillna(mode_value, inplace=True)
                        print("Success!! NaN is filled with the mode of the respective column")
                    elif dtype_change == 0:
                        mode_value = self.df[object1].mode()[0]
                        self.df[object1].fillna(mode_value, inplace=True)
                        print("Success!! NaN is filled with the mode of the respective column")
                    else:
                        print("Enter a valid choice")
                        print("Restarting the function !")
                        self.handle_null_values()
                else:
                    print("Please enter 'Mode' or 'Mean' or 'Remove'")
                    print("Restarting the function !")
                    self.handle_null_values()
            else:
                print("Please enter a valid feature name")
                print("Restarting the function !")
                self.handle_null_values()
        except Exception as e:
            print("An error occurred in handle_null_values method:", e)
            self.handle_null_values()
            pass

    def remove_duplicates(self):
        try:
            action = int(input("Do you want to remove duplicates? ('1' for yes and '0' for no): "))
            if action == 1:
                self.df.drop_duplicates(inplace=True)
                print("Duplicates dropped!")
            elif action == 0:
                pass
            else:
                print("Please enter a valid input!")
                print("Restarting the function !")
                self.remove_duplicates()
        except Exception as e:
            print("An error occurred in remove_duplicates method:", e)
            self.remove_duplicates()
            pass

    def handle_categorical_values(self):
        try:
        # Handling categorical values
            cat_method = int(input("Please Enter a method to handle categorical values\n1 -->Label Encoding\n2 -->One-Hot Encoding\n3 -->Binary Encoding\n4 -->Frequency Encoding\n-->"))
            col = input("Please Enter the column names from the list: ")
        
        # Check if the selected column is appropriate for categorical encoding
            if self.df[col].dtype == 'object':
                if cat_method == 1:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col])
                    print("Label Encoding Successful !!")
                elif cat_method == 2:
                # One-Hot Encoding
                    replace = input("Do you want to replace the dummy column with existing column ('Yes' or 'No'): ")
                    dummy = pd.get_dummies(self.df[col])
                    if replace.lower() == 'yes':
                        self.df = pd.concat([self.df, dummy], axis=1)
                        self.df.drop(col, axis=1, inplace=True)
                    elif replace.lower() == 'no':
                        self.df = pd.concat([self.df, dummy], axis=1)
                    else:
                        print("Please Enter 'Yes' or 'No'")
                        print("Restarting the function !")
                        self.handle_categorical_values()

                elif cat_method == 3:
                # Binary Encoding
                    replace = input("Do you want to replace the dummy column with existing column ('Yes' or 'No'): ")
                    be = BinaryEncoder()
                    binary_values = be.fit_transform(self.df[col])
                    if replace.lower() == 'yes':
                        self.df = pd.concat([self.df, binary_values], axis=1)
                        self.df.drop(col, axis=1, inplace=True)
                    elif replace.lower() == 'no':
                        self.df = pd.concat([self.df, binary_values], axis=1)
                    else:
                        print("Please Enter 'Yes' or 'No'")
                        print("Restarting the function !")
                        self.handle_categorical_values()
                    
                elif cat_method == 4:
                    freq = self.df.groupby(col).size() / len(self.df)
                    self.df.loc[:, "{}_frequency".format(col)] = self.df[col].map(freq)
                    print("Column for frequency of the data has been added to dataset !!")
                else:
                    print("Please give a valid input")
                    print("Restarting the function !")
                    self.handle_categorical_values()
            else:
                print("Selected column is not categorical. Please select a categorical column.")
                self.handle_categorical_values()
            
        except Exception as e:
            print("An error occurred in handle_categorical_values method:", e)
            self.handle_categorical_values()
            pass


    def outlier_detection(self):
        try:
            # Outlier Detection
            print("")
            print("Please Enter which method u want to perform for Outlier Detection and Removal")
            method = int(input("1 --> Z-Score Method \n2 --> Inter-Quartile Range Method \n3 --> Percentile Method \n-->"))
            col = input("Please Enter the column names from the list: ")
            if method == 1:
                upper_limit = self.df[col].mean() + 3*self.df[col].std()
                lower_limit = self.df[col].mean() - 3*self.df[col].std()
                print("The upper Limit is: ", upper_limit)
                print("The lower Limit is: ", lower_limit)
                # Handling outliers
                location = self.df.loc[(self.df[col] >= upper_limit) | (self.df[col] <= lower_limit)]
                action = int(input("What action u want on your data ? \n1 -->Data Trimming \n2 -->Data Capping \n-->"))
                if action == 1:
                    self.df = self.df.loc[(self.df[col] <= upper_limit) & (self.df[col] >= lower_limit)]
                    print("Trimming Successful!")
                elif action == 2:
                    self.df.loc[self.df[col] >= upper_limit, col] = upper_limit
                    self.df.loc[self.df[col] <= lower_limit, col] = lower_limit
                    print("Capping Successful!")
                else:
                    print("Please choose between '1' or '2'")
                    print("Restarting the function !")
                    self.outlier_detection()
            elif method == 2:
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                upper_limit = q3 + (1.5*iqr)
                lower_limit = q1 - (1.5*iqr)
                sns.boxplot(self.df[col])
                # Finding the outliers
                location = self.df.loc[(self.df[col]>=upper_limit)|(self.df[col]<=lower_limit)]
                action = int(input("What action u want on your data ? \n1 -->Data Trimming \n2 -->Data Capping \n-->"))
                if action == 1:
                    # Trimming --> Deleting
                    self.df = self.df.loc[(self.df[col]<=upper_limit)&(self.df[col]>=lower_limit)]
                    print("Trimming Successfull!")
                elif action == 2:
                    # Capping --> Change the value to upper limit or the lower limit
                    new_df = self.df.copy()
                    new_df.loc[new_df[col]>=upper_limit , col] = upper_limit
                    new_df.loc[new_df[col]<=lower_limit , col] = lower_limit
                    print("Capping Successfull!")  
                else:
                    print("Please choose between '1' or '2'")
            elif method == 3:
                # Percentile Method
                upper_limit = self.df[col].quantile(0.99)
                lower_limit = self.df[col].quantile(0.01)
                sns.boxplot(self.df[col])
                # Finding the outliers
                location = self.df.loc[(self.df[col]>=upper_limit)|(self.df[col]<=lower_limit)]
                action = int(input("What action u want on your data ? \n1 -->Data Trimming \n2 -->Data Capping \n-->"))
                if action == 1:
                    # Trimming --> Deleting
                    self.df = self.df.loc[(self.df[col]<=upper_limit)&(self.df[col]>=lower_limit)]
                    print("Trimming Successfull!")
                elif action == 2:
                    # Capping --> Change the value to upper limit or the lower limit

                    new_df = self.df.copy()
                    new_df.loc[new_df[col]>=upper_limit , col] = upper_limit
                    new_df.loc[new_df[col]<=lower_limit , col] = lower_limit
                    print("Capping Successfull!")  
                else:
                    print("Please choose between '1' or '2'")
        except Exception as e:
            print("An error occurred in outlier_detection method:", e)
            self.outlier_detection()
            pass

    def split_dataset(self):
        try:
            replace_target = int(input("Do you want to put the target column at the end? ('1' for Yes and '0' for No): "))
            print("")
            if replace_target == 1:
                target_column = input("Enter the name of the Target Column: ")
                print("")
                columns = [col for col in self.df.columns if col != target_column] + [target_column]
                self.df = self.df[columns]
                X = self.df.iloc[:, :-1]   # Independent variable
                y = self.df.iloc[:, -1]    # Target variable
                test_size = float(input("Enter the test_size for the data-set (e.g. 0.2 ; 0.1)"))
                print("")
                random = int(input("Enter Integer for randomisation: "))
                print("")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random)
                shape = int(input("Do you want to see the shape of your train and testing data-set? ('1' for yes and '0' for no): "))
                print("")
                if shape == 1:
                    train_shape = X_train.shape
                    test_shape = X_test.shape
                    print("The train shape is: ", train_shape, "\nThe test data shape is: ", test_shape)
                elif shape == 0:
                    pass
                else:
                    print("Please Enter a valid choice ('1' or '0')")
                    self.split_dataset()

            elif replace_target == 0:
                X = self.df.iloc[:, :-1]   # Independent variable
                y = self.df.iloc[:, -1]    # Target variable
                test_size = float(input("Enter the test_size for the data-set (e.g. 0.2 ; 0.1): "))
                print("")
                random = int(input("Enter Integer for randomisation: "))
                print("")
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random)
            else:
                print("Please Enter a valid input")
                self.split_dataset()

            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            db_name = input("Enter the name of the database: ")

            try:
                mongo_client = pm.MongoClient("mongodb://localhost:27017/")
                db = mongo_client[db_name]

                train_collection = db["Train_Data"]
                train_collection.insert_many(train_data.to_dict(orient='records'))
                print("Train data inserted into MongoDB collection successfully!")

                test_collection = db["Test_Data"]
                test_collection.insert_many(test_data.to_dict(orient='records'))
                print("Test data inserted into MongoDB collection successfully!")

            except Exception as e:
                print("An error occurred while inserting data into MongoDB collection:", e)

        except Exception as e:
            print("An error occurred in split_dataset method:", e)
            self.split_dataset()


    def output_file(self):
        choice = int(input("What type of output file do you want ?\n1--> CSV File\n2--> Database Collection\n--> "))
        print("")
        if choice == 1:
            try:
                file_name = input("Enter the file name: ")
                file_path = f"E:\\Projects\\PreProcessingFY1\\{file_name}.csv"
                print("Successfully saved as an excel sheet !")
                self.df.to_csv(file_path)
            except Exception as e:
                print("An error occurred in output_file method:", e)
                self.output_file()
        elif choice == 2:
            collection_name = input("Enter the name of the collection in the database: ")
            db_name = input("Enter the name of the database: ")
            try:
                mongo_client = pm.MongoClient("mongodb://localhost:27017/")
                db = mongo_client[db_name]
                collection = db[collection_name]
                data_dict = self.df.to_dict(orient='records')
                # Insert data into MongoDB collection
                collection.insert_many(data_dict)
                print("Data inserted into MongoDB collection successfully!")
            except Exception as e:
                print("An error occurred while inserting data into MongoDB collection:", e)
                self.output_file()
        else:
            print("Please enter a valid output source !")
            self.output_file()
            pass
        
if __name__ == "__main__":
    data_processor = CarDataProcessor()
    data_processor.fit()















