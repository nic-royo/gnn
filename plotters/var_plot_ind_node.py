# plot variance only for target nodes of a dataset
import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# dict with directories of CSV files
csv_directories = {
    'trained_no_norm': [
        'results/train_embeddings/all_node_embeddings_no_norm/dblp_pnrgcn',
        'results/train_embeddings/all_node_embeddings_no_norm/dblp_rgcn',
        'results/train_embeddings/all_node_embeddings_no_norm/imdb_pnrgcn',
        'results/train_embeddings/all_node_embeddings_no_norm/imdb_rgcn'
    ],
    'trained_norm': [
        'results/train_embeddings/all_node_embeddings_norm/dblp_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/dblp_rgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_rgcn'
    ],
    'propagated_no_norm': [
        'results/prop_embeddings/all_node_embeddings_no_norm/dblp_pnrgcn',
        'results/prop_embeddings/all_node_embeddings_no_norm/dblp_rgcn',
        'results/prop_embeddings/all_node_embeddings_no_norm/imdb_pnrgcn',
        'results/prop_embeddings/all_node_embeddings_no_norm/imdb_rgcn'
    ],
    'propagated_norm': [
        'results/train_embeddings/all_node_embeddings_norm/dblp_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/dblp_rgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_rgcn'
    ],
}

def plot_variance_from_csv(csv_files, output_filename, plot_title):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))

    for csv_file, color in zip(csv_files, colors):
        df = pd.read_csv(csv_file, index_col=0)

        # set Node_Type based on the CSV dataset type
        if 'dblp' in csv_file:
            node_type = 'author'
        elif 'imdb' in csv_file:
            node_type = 'movie'
        else:
            continue  

        # filter df so it only has the node type we want
        df_filtered = df[df['Node_Type'] == node_type]
        df_features = df_filtered.drop(columns='Node_Type')
        variances = df_features.var(axis=0)

        # get num_layers (num_hops) from the filename
        num_hops = os.path.basename(csv_file).split('_')[2]  # Adjust position if necessary

        plt.plot(range(1, len(variances) + 1), variances, marker='o', label=f'{num_hops} layers', color=color)

    plt.title(plot_title, fontsize=16)
    plt.xlabel('Feature Index', fontsize=14)
    plt.ylabel('Variance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot variance from CSV files in specified directory.')
    parser.add_argument('--dataset_type', choices=['trained_no_norm', 'trained_norm', 'propagated_no_norm', 'propagated_norm'], required=True, help='Specify the dataset type to use.')

    args = parser.parse_args()

    plots_dir = "plots/features_variance_target_node"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    selected_csv_directories = csv_directories[args.dataset_type]

    for csv_directory in selected_csv_directories:
        csv_pattern = os.path.join(csv_directory, '*.csv')
        csv_files = glob.glob(csv_pattern)

        # sort files based on a specific part of the filename if needed, otherwise we get an error
        try:
            csv_files = sorted(csv_files, key=lambda x: int(os.path.basename(x).split('_')[2]))
        except ValueError:
            print("Warning: Sorting failed due to non-integer part. Sorting files by name.")
            csv_files = sorted(csv_files)

        # get dataset name and model name from the directory path
        base_name = os.path.basename(csv_directory)
        dataset_name, model_name = base_name.split('_')

        # setup plot title based on dataset type, model name, and dataset name
        output_filename = os.path.join(plots_dir, f'feature_variance_plot_ind_node{args.dataset_type}_{dataset_name}_{model_name}.png')
        plot_title = f'Variance of Target Node Features - {args.dataset_type.replace("_", " ").title()} - {dataset_name.upper()} - {model_name.upper()}'

        plot_variance_from_csv(csv_files, output_filename, plot_title)