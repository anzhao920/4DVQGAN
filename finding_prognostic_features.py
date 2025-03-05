import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np

# Load your CSV file into a pandas DataFrame
df = pd.read_csv("C:\\My Data\\Leuven\\training_dataset_histogram_target.csv")

# Define the outcome and covariates
outcome_col = "dead"  # Mortality column (1 or 0)
covariates = ['sex', 'age', 'smoking', 'follow up time']  # Other covariates

# Prepare a dataframe to store the p-values and C-index values for each feature (Bin 0 - Bin 255)
results = pd.DataFrame(columns=["Bin", "P-value", "C-index", "Hazard Ratio"])

# Initialize Cox Proportional Hazards model
cph = CoxPHFitter()

# Columns for bins are 'Bin 0' to 'Bin 255'
bin_columns = [f"Bin {i}" for i in range(256)]

# Calculate the total sum across all bins for each patient
df['total_bins'] = df[bin_columns].sum(axis=1)

# For each bin, calculate the proportion relative to the total sum of all bins for that patient
for i in range(256):
    bin_column = f"Bin {i}"
    df[f"Bin {i}_proportion"] = df[bin_column] / df['total_bins']

# Loop through each Bin feature (from Bin 0 to Bin 255)
for i in range(256):
    # Create a model with the current Bin feature and the covariates
    feature_col = f"Bin {i}_proportion"

    # Prepare the feature and covariates for the Cox model
    model_data = df[[outcome_col, feature_col] + covariates].dropna()
    
    # Fit the Cox model
    cph.fit(model_data, duration_col="follow up time", event_col=outcome_col)
    
    # Get the p-value for the current feature
    p_value = cph.summary.loc[feature_col, "p"]

    # Get the Hazard Ratio for the current feature
    hazard_ratio = np.exp(cph.params_[feature_col])  # Exponentiate the coefficient to get the hazard ratio

    # Calculate the C-index for the current model
    # We use the concordance_index function from lifelines for this
    c_index = concordance_index(model_data['follow up time'], -cph.predict_partial_hazard(model_data), model_data[outcome_col])
    
    # Append the results to the results DataFrame
    results = results.append({"Bin proportion": feature_col, "P-value": p_value, "C-index": c_index, "Hazard Ratio": hazard_ratio}, ignore_index=True)

# Filter for significant features (p-value < 0.01)
significant_results = results[results["P-value"] < 0.01]
significant_results = significant_results[significant_results["Hazard Ratio"] > 1]

# Save the results to a CSV file
results.to_csv("cox_model_results.csv", index=False)

# Save the significant results (p-value < 0.01 and Hazard Ratio > 1)
significant_results.to_csv("significant_cox_model_results.csv", index=False)

print("Results have been saved to 'cox_model_results.csv' and 'significant_cox_model_results.csv'.")

# --- Now select the top 10 features based on p-value and refit the final model ---

# Sort the significant results by p-value and select the top 10 features
top_10_features = significant_results.sort_values(by="P-value").head(10)

# Extract the feature names (Bin proportion columns)
selected_features = ["Bin proportion"].values

# Define the final model with the top 10 features
final_covariates = list(selected_features) + covariates  # Add the selected features to the list of covariates

# Prepare the final dataset with only the selected features and covariates
final_model_data = df[[outcome_col] + final_covariates].dropna()

# Fit the Cox model with the top 10 features
cph.fit(final_model_data, duration_col="follow up time", event_col=outcome_col)

# Calculate the C-index for the final model
c_index_final = concordance_index(final_model_data['follow up time'], -cph.predict_partial_hazard(final_model_data), final_model_data[outcome_col])

print("C-index of the final model with the top 10 features:", c_index_final)

# --- Now use all 256 features and the covariates to predict survival and calculate C-index ---

# Prepare the dataset with all the 256 features and covariates
# all_features = [f"Bin {i}_proportion" for i in range(256)]  # Using the proportions of Bin 0 - Bin 255
# final_covariates_all = all_features + covariates  # Add covariates as well

# # Prepare the final dataset with all features
# final_model_data_all = df[[outcome_col] + final_covariates_all].dropna()

# # Fit the Cox model with all 256 features
# cph.fit(final_model_data_all, duration_col="follow up time", event_col=outcome_col)

# # Calculate the C-index for the final model with all 256 features
# c_index_final_all = concordance_index(final_model_data_all['follow up time'], -cph.predict_partial_hazard(final_model_data_all), final_model_data_all[outcome_col])

# print("C-index of the final model with all 256 features:", c_index_final_all)
