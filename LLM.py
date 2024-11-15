import os 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

#funtcion to load data 

def load_data(file_path, variables_needed):
    """Load the dataset here using file_path and variables needed""" 
    df=pd.read_csv(file_path)
    subset = df[variables_needed]
    subset.to_csv('subset_world_values_survey.csv', index=False)
    return subset

#function for empty data cleaned from subset

def clean_data(subset):
    """Clean the data by removing rows with missing values."""
    missing_data = subset.isnull().sum()
    print("Missing data in each variable: ")
    print(missing_data)
    subset_data_cleaned = subset.dropna()
    print("\nShape of data before cleaning:", subset.shape)
    print("Shape of data after removing missing values:", subset_data_cleaned.shape)
    
    subset_data_cleaned.to_excel('cleaned_world_values_survey.xlsx', index=False)
    return subset_data_cleaned

#function to split the clean data into training and testing 

def split_data(subset_data_cleaned, independent_vars, dependent_var):
    """Split the data into 60 percent training and 40 percent testing"""
    X= subset_data_cleaned[independent_vars]
    y=subset_data_cleaned[dependent_var]
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.4, random_state= 42)
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(X_train, y_train):
    """Tune hyperparameters for the Random Forest model using Grid Search."""
    param_grid= {
        'n_estimators':[50, 100, 200],
        'max_depth':[None, 10, 20, 30],
        'min_samples_split':[2, 5, 10],
        'min_samples_leaf':[1,2,4]
    }
    rf= RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator = rf, param_grid=param_grid, cv=3, n_jobs = -1, verbose= 2)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

def analyze_correlations(subset_data_cleaned, dependent_var, independent_vars, folder_path, country_code):
    """Analyze and plot correlations between the dependent variable and independent variables."""
    correlations = subset_data_cleaned[[dependent_var] + independent_vars].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlations, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix for {dependent_var} in {country_code}')
    plt.savefig(os.path.join(folder, f'Heatplot_{country_code}.png'))
    plt.close()

def create_output_folder(base_folder_name="LLM"):
    """Create an incremented folder for storing heatplots."""
    folder_number = 2
    while True:
        folder_name = f"{base_folder_name}_{folder_number}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            break
        folder_number += 1
    return folder_name

def main():
    file_path= '/Users/tanushzutshi/Downloads/Python_JOB code/WVS_Cross-National_Wave_7_csv_v5_0 2.csv'
    variables_needed = ['B_COUNTRY_ALPHA','Q46','Q47','Q57','Q106','Q107','Q108','Q109','Q110','Q152','Q154','Q156','Q164','Q169','Q170','Q199','Q201','Q202','Q203','Q204','Q205','Q206','Q207','Q208','Q260','Q262','Q263','Q273','Q275','Q279','Q287']
    subset = load_data(file_path, variables_needed)
    subset_data_cleaned = clean_data(subset)

    output_folder = create_output_folder()

    countries = subset_data_cleaned['B_COUNTRY_ALPHA'].unique()
    dependent_var = 'Q46'
    models = {}  # Dictionary to store trained models for each country

     # Iterate through each variable as the dependent variable but seperately for each country

    for country in countries:
        print(f"\nProcessing data for country: {country}")
        country_data = subset_data_cleaned[subset_data_cleaned['B_COUNTRY_ALPHA'] == country]
        
        # Define independent variables
        independent_vars = [var for var in variables_needed if var not in ['B_COUNTRY_ALPHA', dependent_var]]
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(country_data, independent_vars, dependent_var)
        
        # Hyperparameter tuning: to tune all the micro parameters that are instilled within this chunk
        best_model = hyperparameter_tuning(X_train, y_train)
        
        # Evaluate the tuned model
        evaluate_model(best_model, X_test, y_test)
        
        # Analyze correlations
        analyze_correlations(country_data, dependent_var, independent_vars, output_folder, country)
        
        # Store the trained model
        models[country] = best_model
    
    # Save the models to disk
    with open('country_models.pkl', 'wb') as f:
        pickle.dump(models, f)
    
def predict(country, inputs):
    """Predict the target variable for a given country and inputs."""
    # Load the models from disk
    with open('country_models.pkl', 'rb') as f:
        models = pickle.load(f)
    
    if country not in models:
        raise ValueError(f"No model found for country: {country}")
    
    model = models[country]
    independent_vars = [var for var in inputs.keys() if var in model.feature_names_in_]
    X_input = pd.DataFrame([inputs[independent_vars]], columns=independent_vars)
    prediction = model.predict(X_input)[0]
    
    return prediction

if __name__ == "__main__":
    main()