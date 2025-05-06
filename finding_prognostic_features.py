import pandas as pd 
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np
from sklearn.model_selection import KFold

# Load your CSV file into a pandas DataFrame
df = pd.read_csv("C:\\My Data\\Leuven\\training_dataset_histogram_target.csv")
df = df[df["time_from_baseline"] == 0]

# Define the outcome and covariates
outcome_col = "dead"  # Mortality column (1 or 0)
covariates = ['sex', 'age', 'smoking', 'follow up time']  # Other covariates

# Prepare a dataframe to store the p-values, C-index values, and Hazard Ratios for each fold
results = pd.DataFrame(columns=["Fold", "Bin", "P-value", "Train C-index", "Test C-index", "Hazard Ratio"])

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

# Initialize 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store biomarkers that pass the criteria across all folds
biomarkers_passed_all_tests = []

# Loop through each Bin feature (from Bin 0 to Bin 255)
for i in range(256):
    feature_col = f"Bin {i}_proportion"

    # Skip the bin if the feature value is less than 0.00001
    if df[feature_col].mean() < 0.00001:
        continue  # Skip this feature if its mean value is less than 0.00001
    
    model_data = df[[outcome_col, feature_col] + covariates].dropna()


    # Initialize lists to store C-index values, p-values, and hazard ratios for each fold
    train_c_index_values = []
    test_c_index_values = []
    fold_p_values = []
    fold_hazard_ratios = []
    
    # Cross-validation loop
    passed = True  # A flag to check if all conditions are met for this feature
    for fold, (train_index, test_index) in enumerate(kf.split(model_data), 1):
        # Split data into training and testing sets based on KFold indices
        train_data = model_data.iloc[train_index]
        test_data = model_data.iloc[test_index]

        # Fit the Cox model on the training data
        cph.fit(train_data, duration_col="follow up time", event_col=outcome_col)

        # Get the p-value for the current feature in the fold
        fold_p_value = cph.summary.loc[feature_col, "p"]
        fold_p_values.append(fold_p_value)

        # Get the Hazard Ratio for the current feature in the fold
        fold_hazard_ratio = np.exp(cph.params_[feature_col])  # Exponentiate the coefficient to get the hazard ratio
        fold_hazard_ratios.append(fold_hazard_ratio)

        # Calculate the C-index on the training data
        train_c_index = concordance_index(train_data['follow up time'], -cph.predict_partial_hazard(train_data), train_data[outcome_col])
        train_c_index_values.append(train_c_index)

        # Calculate the C-index on the test data
        test_c_index = concordance_index(test_data['follow up time'], -cph.predict_partial_hazard(test_data), test_data[outcome_col])
        test_c_index_values.append(test_c_index)

        # Check if conditions are met for p-value, hazard ratio, and C-index
        if fold_p_value >= 0.05 or fold_hazard_ratio <= 1 or test_c_index <= 0.5:
            passed = False
            break  # No need to check further folds if one fold fails the criteria
    
    # If the feature passes the criteria for all folds, store its results
    if passed:
        avg_train_c_index = np.mean(train_c_index_values)
        avg_test_c_index = np.mean(test_c_index_values)
        avg_p_value = np.mean(fold_p_values)
        avg_hazard_ratio = np.mean(fold_hazard_ratios)

        # Append the feature that passed all tests
        biomarkers_passed_all_tests.append({
            "Bin": feature_col,
            "Avg P-value": avg_p_value,
            "Avg Train C-index": avg_train_c_index,
            "Avg Test C-index": avg_test_c_index,
            "Avg Hazard Ratio": avg_hazard_ratio
        })
        
        # Append fold-specific results to the results DataFrame
        for fold in range(1, 6):
            results = results.append({
                "Fold": fold,
                "Bin": feature_col,
                "P-value": fold_p_values[fold-1],
                "Train C-index": train_c_index_values[fold-1],
                "Test C-index": test_c_index_values[fold-1],
                "Hazard Ratio": fold_hazard_ratios[fold-1]
            }, ignore_index=True)

# Convert biomarkers that passed the test into a DataFrame
biomarkers_df = pd.DataFrame(biomarkers_passed_all_tests)

# Sort biomarkers by Avg Test C-index in descending order
biomarkers_df_sorted = biomarkers_df.sort_values(by="Avg Test C-index", ascending=False)

# Select the top 5 biomarkers
top_5_biomarkers = biomarkers_df_sorted.head(5)

# Compute the correlation matrix among the top 5 biomarkers' proportions
top_5_columns = [biomarker for biomarker in top_5_biomarkers["Bin"]]
correlation_matrix = df[top_5_columns].corr()

# Save the results to CSV files
results.to_csv("cox_model_results_cv_with_train_test_cindex_and_hazard_ratio.csv", index=False)
biomarkers_df.to_csv("biomarkers_passed_all_tests.csv", index=False)
top_5_biomarkers.to_csv("top_5_biomarkers.csv", index=False)
correlation_matrix.to_csv("correlation_matrix_top_5_biomarkers.csv", index=True)

print("Results have been saved to 'cox_model_results_cv_with_train_test_cindex_and_hazard_ratio.csv', 'biomarkers_passed_all_tests.csv', 'top_5_biomarkers.csv', and 'correlation_matrix_top_5_biomarkers.csv'.")