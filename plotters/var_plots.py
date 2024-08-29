# plot variance of features in final layer embeddings, change directory lcoation for different ones
# do it per directory and save it for directory
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

# source here

csv_directories = {
    'trained_no_norm': [
        'results/train_embeddings/all_node_embeddings_no_norm/dblp_pnrgcn',
        'results/train_embeddings/all_node_embeddings_no_norm/dblp_rgcn',
        'results/train_embeddings/all_node_embeddings_no_norm/imdb_pnrgcn',
        'results/train_embeddings/all_node_embeddings_no_norm/imdb_pnrgcn'
    ],
    'trained_norm': [
        'results/train_embeddings/all_node_embeddings_norm/dblp_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/dblp_rgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_pnrgcn'
    ],
    'propagated_no_norm' : [
        'results/prop_embeddings/all_node_embeddings_no_norm/dblp_pnrgcn',
        'results/prop_embeddings/all_node_embeddings_no_norm/dblp_rgcn',
        'results/prop_embeddings/all_node_embeddings_no_norm/imdb_pnrgcn',
        'results/prop_embeddings/all_node_embeddings_no_norm/imdb_pnrgcn'
    ],
    'propagated_norm' : [
        'results/prop_embeddings/all_node_embeddings_norm/dblp_pnrgcn',
        'results/prop_embeddings/all_node_embeddings_norm/dblp_rgcn',
        'results/prop_embeddings/all_node_embeddings_norm/imdb_pnrgcn',
        'results/prop_embeddings/all_node_embeddings_norm/imdb_pnrgcn'
    ]
}
def plot_variance_from_csv(csv_files, output_filename, plot_title):
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
    plt.title(plot_title, fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Variance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Save and show the plot
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Plot variance from CSV files in specified directory.')
    parser.add_argument('--dataset_type', choices=['trained_no_norm', 'trained_norm', 'propagated_no_norm', 'propagated_norm'], required=True, help='Specify the dataset type to use.')

    args = parser.parse_args()

    # Get the selected CSV directories
    selected_csv_directories = csv_directories[args.dataset_type]

    for csv_directory in selected_csv_directories:
        csv_pattern = os.path.join(csv_directory, '*.csv')
        csv_files = glob.glob(csv_pattern)
        
        # Sort files based on a specific part of the filename if needed
        try:
            csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).split('_')[2]))
        except ValueError:
            print("Warning: Sorting failed due to non-integer part. Sorting files by name.")
            csv_files = sorted(csv_files)
        
        # Determine output file name and plot title based on dataset type
        output_filename = os.path.join(csv_directory, f'feature_variance_plot_{args.dataset_type}.png')
        plot_title = f'Variance of Features Across All Nodes - {args.dataset_type.replace("_", " ").title()}'
        
        # Generate and save the plots
        plot_variance_from_csv(csv_files, output_filename, plot_title)
