# code to combine the csvs from each f1 score
import os
import pandas as pd

# ath to folder containing the CSV files
folder_path = '3xruns/3xruns_outputs/dblp'

combined_df = pd.DataFrame()
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# drop columns "Macro F1 Mean" and "Macro F1 Variance"
columns_to_drop = ["Micro F1 Mean", "Micro F1 Variance"]
combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')

combined_df.to_csv('3xruns/combined_dblp_mac.csv', index=False)
