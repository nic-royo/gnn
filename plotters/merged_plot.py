import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_embeddings_from_csv(csv_files, output_filename, dataset_name, model_name):
    num_plots = len(csv_files)
    
    # Adjust figsize to make each subplot narrower
    fig_width = 7 * num_plots  # Adjust the width per plot
    fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, 10), sharey=True)
    
    if num_plots == 1:
        axes = [axes]  

    for ax, csv_file in zip(axes, csv_files):
        df = pd.read_csv(csv_file, index_col=0)
        
        # Choose node type depending on file name
        if 'dblp' in csv_file:
            node_type = 'author'
        elif 'imdb' in csv_file:
            node_type = 'movie'
        else:
            continue  

        # Filter df so it only has the node type we want
        df_filtered = df[df['Node_Type'] == node_type]
        df_features = df_filtered.drop(columns='Node_Type')

        # Sample 50 nodes randomly
        sampled_df = df_features.sample(n=50, random_state=42)

        # Plot using heatmap
        sns.heatmap(sampled_df, cmap='coolwarm', cbar=False, ax=ax, xticklabels=False, yticklabels=False)
        
        # Set title based on the num_layers (extracted from the filename)
        num_hops = 'Unknown'
        basename = os.path.basename(csv_file)
        parts = basename.split('_')
        for part in parts:
            if part.isdigit():
                num_hops = part
                break

        ax.set_title(f'{num_hops} Layers', fontsize=26)  # Adjust title font size
        ax.set_xlabel("Features", fontsize=26)  # Adjust xlabel font size
        ax.set_ylabel("Nodes", fontsize=26)  # Adjust ylabel font size

        # Remove y-axis ticks and labels
        ax.yaxis.set_ticks([])
        ax.yaxis.set_tick_params(length=0)  

    # Set general title
    dataset_name_upper = dataset_name.upper()
    model_name_upper = model_name.upper()
    general_title = f'Visualization of 50 Randomly Sampled Target Nodes, Trained Norm - {dataset_name_upper} - {model_name_upper}'
    plt.suptitle(general_title, fontsize=30)  # Adjust general title font size

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    csv_directories = [
        'results/train_embeddings/all_node_embeddings_norm/dblp_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/dblp_rgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_pnrgcn',
        'results/train_embeddings/all_node_embeddings_norm/imdb_rgcn'
    ]

    # Layers of interest
    layers_of_interest = {8, 16, 32}

    for csv_directory in csv_directories:
        # Get dataset name and model name from the directory
        directory_parts = csv_directory.split('/')
        dataset_name = directory_parts[-1].split('_')[0]  # 'dblp' or 'imdb'
        model_name = directory_parts[-1].split('_')[1]  # 'pnrgcn' or 'rgcn'
        
        # Pattern to match all CSV files
        csv_pattern = os.path.join(csv_directory, '*.csv')
        csv_files = glob.glob(csv_pattern)
        
        # Filter and sort files based on the num_layers
        csv_files = [f for f in csv_files if any(int(part) in layers_of_interest for part in os.path.basename(f).split('_') if part.isdigit())]
        csv_files = sorted(csv_files, key=lambda x: next((int(part) for part in os.path.basename(x).split('_') if part.isdigit()), 0))

        # Construct output filename in the desired format
        output_filename = f"water_plot_{dataset_name}_{model_name}.svg"        
        plot_embeddings_from_csv(csv_files, output_filename, dataset_name, model_name)
