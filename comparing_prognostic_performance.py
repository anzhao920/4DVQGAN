import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# Load the true biomarker values and predicted biomarker values from CSV files
true_df = pd.read_csv('C:\\My Data\\Leuven\\test_dataset_histogram_target.csv')  # True biomarker values (from target.csv)
predicted_df = pd.read_csv('C:\\My Data\\Leuven\\test_dataset_histogram_predicted.csv')  # Predicted biomarker values (from prediction.csv)

true_df = true_df.groupby('Patient ID').nth(2)
predicted_df = predicted_df.groupby('Patient ID').nth(2)

# Columns for bins are 'Bin 0' to 'Bin 255'l
bin_columns = [f"Bin {i}" for i in range(256)]

# Calculate the total sum across all bins for each patient
true_df['total_bins'] = true_df[bin_columns].sum(axis=1)
predicted_df['total_bins'] = predicted_df[bin_columns].sum(axis=1)

# For each bin, calculate the proportion relative to the total sum of all bins for that patient

for i in range(256):
    bin_column = f"Bin {i}"
    true_df[f"Bin_{i}_proportion"] = true_df[bin_column] / true_df['total_bins']
    predicted_df[f"Bin_{i}_proportion"] = predicted_df[bin_column] / predicted_df['total_bins']



# Selected features (indices) for the biomarkers
# bin_indices = [196,238,204,229,128,251,85,197,194,182]
bin_indices = [196,238,204,229,128]
# bin_indices = [196]
selected_features = [f"Bin_{bin_index}_proportion" for bin_index in bin_indices]
true_biomarkers = true_df.loc[:, selected_features]
predicted_biomarkers = predicted_df.loc[:, selected_features]

# Step 1: Calculate MAE for the selected 10 biomarkers
mae_values = []
for feature in range(len(bin_indices)):
    true_values = true_biomarkers.iloc[:, feature]
    predicted_values = predicted_biomarkers.iloc[:, feature]
    mae = mean_absolute_error(true_values, predicted_values)
    mae_values.append(mae)

# Display the MAE for each feature
mae_df = pd.DataFrame({
    "Feature": [f"Bin_{bin_indices[i]}_proportion" for i in range(len(bin_indices))],
    "MAE": mae_values
})
print(mae_df)

# Step 2: Calculate the correlation (Spearman's rank correlation)
correlation_values = []
for feature in range(len(bin_indices)):
    true_values = true_biomarkers.iloc[:, feature]
    predicted_values = true_biomarkers.iloc[:, feature]
    corr, _ = spearmanr(true_values, predicted_values)
    correlation_values.append(corr)

# Display the correlation for each feature
correlation_df = pd.DataFrame({
    "Feature": [f"Bin_{bin_indices[i]}_proportion" for i in range(len(bin_indices))],
    "Spearman Correlation": correlation_values
})
print(correlation_df)

# Step 3: Prepare the data for Cox model analysis
covariates = ['age', 'sex', 'smoking']  # Add these covariates from your dataset
outcome_col = "dead"  # Mortality column
follow_up_time_col = "follow up time"

# Load covariates (age, sex, smoking) and the outcome
covariates_df = true_df[covariates + [outcome_col, follow_up_time_col]].dropna()
true_data = pd.concat([covariates_df, true_biomarkers], axis=1)
predicted_data = pd.concat([covariates_df, predicted_biomarkers], axis=1)

# Step 4: Fit the Cox Proportional Hazards model with the true biomarkers
cph_true = CoxPHFitter()
cph_true.fit(true_data, duration_col=follow_up_time_col, event_col=outcome_col)
c_index_true = cph_true.concordance_index_

# Step 5: Fit the Cox Proportional Hazards model with the predicted biomarkers
cph_pred = CoxPHFitter()
cph_pred.fit(predicted_data, duration_col=follow_up_time_col, event_col=outcome_col)
c_index_pred = cph_pred.concordance_index_

# Output C-index for both models
print(f"C-index for the model with true biomarkers: {c_index_true}")
print(f"C-index for the model with predicted biomarkers: {c_index_pred}")

# Optionally, save the results to CSV files
mae_df.to_csv("mae_results.csv", index=False)
correlation_df.to_csv("correlation_results.csv", index=False)

print("Results have been saved to 'mae_results.csv' and 'correlation_results.csv'.")
