# code to combine the csvs from each f1 score
import os
import pandas as pd

# Path to the folder containing the CSV files
folder_path = '3xruns/3xruns_outputs/dblp'

# Initialize an empty DataFrame
combined_df = pd.DataFrame()

# Iterate over each CSV file in the folder and append its content to combined_df
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Drop the columns "Macro F1 Mean" and "Macro F1 Variance"
columns_to_drop = ["Micro F1 Mean", "Micro F1 Variance"]
combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('3xruns/combined_dblp_mac.csv', index=False)
