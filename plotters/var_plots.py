# make variance plots for all node types in a specific directory
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# source here
csv_directories = [
    '3xruns/3xruns_outputs/allnode_embeddings_nonorm/dblp_rgcn',
    '3xruns/3xruns_outputs/allnode_embeddings_nonorm/dblp_pnrgcn',
    '3xruns/3xruns_outputs/allnode_embeddings_nonorm/imdb_rgcn',
    '3xruns/3xruns_outputs/allnode_embeddings_nonorm/imdb_pnrgcn',
]

def plot_variance_from_csv(csv_files, output_filename, directory_name):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))
    
    for csv_file, color in zip(csv_files, colors):
        # Load the CSV file
        df = pd.read_csv(csv_file, index_col=0)
        
        # Exclude 'Node_Type' column for variance calculation
        df_features = df.drop(columns='Node_Type')
        
        # Calculate variance of each feature across nodes
        variances = df_features.var(axis=0)
        
        # Extract the number of layers (num_hops) from the filename
        num_hops = os.path.basename(csv_file).split('_')[2]  # Assuming layers are in the third position
        
        # Plot variance vs feature index
        plt.plot(range(1, len(variances) + 1), variances, marker='o', label=f'{num_hops} layers', color=color)

    # Title, labels, and legend with increased font size
    plt.title(f'Variance of Features Across All Nodes, All Node Types for {directory_name}', fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Variance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save and show the plot
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    for csv_directory in csv_directories:
        csv_pattern = os.path.join(csv_directory, '*.csv')
        csv_files = glob.glob(csv_pattern)
        
        # Sort files based on a specific part of the filename if needed
        try:
            csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).split('_')[2]))
        except ValueError:
            print("Warning: Sorting failed due to non-integer part. Sorting files by name.")
            csv_files = sorted(csv_files)
        
        # Extract directory name for title
        directory_name = os.path.basename(csv_directory)
        
        output_filename = os.path.join(csv_directory, 'feature_variance_plot.png')
        
        # Generate and save the plots
        plot_variance_from_csv(csv_files, output_filename, directory_name)