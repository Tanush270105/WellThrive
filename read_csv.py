# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import numpy as np

# variables_needed=['Q46','Q47','Q57','Q106','Q107','Q108','Q109','Q110','Q152','Q154','Q156','Q164','Q169','Q170','Q199','Q201','Q202','Q203','Q204','Q205','Q206','Q207','Q208','Q260','Q262','Q263','Q273','Q275','Q279','Q287']
# df= pd.read_csv('WVS_Cross-National_Wave_7_csv_v5_0 2.csv')
# subset=df[variables_needed]

# subset.to_csv('subset_world_values_survey.csv', index=False)
# missing_data = subset.isnull().sum()
# print("Missing data in each variable:")
# print(missing_data)

# # Drop rows with any missing values in any of the selected variables
# subset_data_cleaned = subset.dropna()

# # Display information about the cleaned data
# print("\nShape of data before cleaning:", subset.shape)
# print("Shape of data after removing missing values:", subset_data_cleaned.shape)
# print("Hello")

# # Assuming 'subset_data_cleaned' is your cleaned DataFrame without missing values

# # Select the variables for linear regression
# independent_vars = ['education_level', 'other_variables']  # Replace with your independent variables
# dependent_var = 'income'  # Replace with your dependent variable

# # Create a linear regression model
# model = LinearRegression()

# # Fit the model
# model.fit(subset_data_cleaned[independent_vars], subset_data_cleaned[dependent_var])

# # Predict the target variable
# subset_data_cleaned['predicted'] = model.predict(subset_data_cleaned[independent_vars])

# # Calculate differences between predicted and actual values
# subset_data_cleaned['difference'] = np.abs(subset_data_cleaned[dependent_var] - subset_data_cleaned['predicted'])

# # Find outliers using Z-score or any threshold
# outlier_threshold = 3  # Adjust this threshold as needed based on your data and context

# # Identify outliers
# outliers = subset_data_cleaned[subset_data_cleaned['difference'] > outlier_threshold]

# # Remove rows with outliers
# subset_data_no_outliers = subset_data_cleaned[subset_data_cleaned['difference'] <= outlier_threshold].copy()

# # Drop the temporary columns
# subset_data_no_outliers.drop(['predicted', 'difference'], axis=1, inplace=True)

# # Display information about the removed outliers
# print(f"Rows before removing outliers: {subset_data_cleaned.shape[0]}")
# print(f"Rows after removing outliers: {subset_data_no_outliers.shape[0]}")


