import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

def filter_patient_for_longitudinal_analysis(df):  
    # # Step 1: Identify patients who have both time_point == 0 and time_point == 2
    # patients_with_both_timepoints = df[df['time_from_baseline'].isin([0, 2])]['Patient ID'].value_counts()
    # patients_with_both_timepoints = patients_with_both_timepoints[patients_with_both_timepoints == 2].index
    # df_filtered = df[df['Patient ID'].isin(patients_with_both_timepoints)]
    # # Step 2: Separate the DataFrame into time point 0 and time point 2
    # df_time0 = df_filtered[df_filtered['time_from_baseline'] == 0]
    # df_time2 = df_filtered[df_filtered['time_from_baseline'] == 2]

    df_time0 = []
    df_time2 = []

    # Process each patient group separately
    for _, group in df.groupby('Patient ID'):
        group = group.sort_values('time_from_baseline').reset_index(drop=True)
        
        if len(group) < 2:
            continue  # Skip patients with less than 2 rows
        
        second_row = group.iloc[1]  # Get the second row
        matching_row = group[group['time_from_baseline'] == second_row['time_from_baseline'] + 2]
        
        if not matching_row.empty:
            df_time0.append(second_row)
            df_time2.append(matching_row.iloc[0])  # Take the first match if multiple exist

    # Convert lists to DataFrames
    df_time0 = pd.DataFrame(df_time0)
    df_time2 = pd.DataFrame(df_time2)



    bin_columns = [col for col in df.columns if col.startswith('Bin_')]
    # Step 3: Merge the two DataFrames on 'patient_ID' to ensure we have both time 0 and time 2 data for each patient
    merged_df = pd.merge(df_time0, df_time2, on='Patient ID', suffixes=('_time0', '_time2'))
    for i in range(256):
        col = f"Bin_{i}_proportion"
        col_time0 = col + '_time0'
        col_time2 = col + '_time2'
        merged_df[col] = merged_df[col_time2] - merged_df[col_time0]
    
    # Step 5: Keep only 'time_point == 0' rows (we already have these)
    # Keep the patient_ID, the calculated Bin_proportions columns (the difference), and other columns from time 0
    final_df = merged_df[['Patient ID'] + bin_columns].copy()

    # If you want to keep other columns like 'other_column' from time 0:
    final_df['age'] = merged_df['age_time0']
    final_df['sex'] = merged_df['sex_time0']
    final_df['smoking'] = merged_df['smoking_time0']
    final_df['dead'] = merged_df['dead_time0']
    final_df['follow up time'] = merged_df['follow up time_time0']

    # Drop unnecessary columns (those with '_time0' or '_time2' suffix)
    # final_df.drop(columns=[col for col in merged_df.columns if col.endswith('_time0') or col.endswith('_time2')], inplace=True)
    return final_df


# Load the true biomarker values and predicted biomarker values from CSV files
true_df = pd.read_csv('C:\\My Data\\Leuven\\test_dataset_histogram_target.csv')  # True biomarker values (from target.csv)
predicted_df = pd.read_csv('C:\\My Data\\Leuven\\test_dataset_histogram_predicted.csv')  # Predicted biomarker values (from prediction.csv)

# true_df = true_df.groupby('Patient ID').nth(2)
# predicted_df = predicted_df.groupby('Patient ID').nth(2)

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

true_df = filter_patient_for_longitudinal_analysis(true_df)
predicted_df = filter_patient_for_longitudinal_analysis(predicted_df)

# Selected features (indices) for the biomarkers
# bin_indices = [196,238,204,229,128,251,85,197,194,182]
# bin_indices = [196,128,85,194,182]
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
