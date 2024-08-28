import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_embeddings_from_csv(csv_files, output_filename):
    # Create a figure with subplots arranged horizontally
    num_plots = len(csv_files)
    fig, axes = plt.subplots(1, num_plots, figsize=(15 * num_plots, 10), sharey=True)
    
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, csv_file in zip(axes, csv_files):
        # Load the CSV file
        df = pd.read_csv(csv_file, index_col=0)
        
        # Plot heatmap
        sns.heatmap(df, cmap='coolwarm', cbar=True, ax=ax)
        
        # Set title based on the number of layers (extracted from the filename)
        num_hops = os.path.basename(csv_file).split('_')[3]
        ax.set_title(f'{num_hops} Layers', fontsize=24)
        
        ax.set_xlabel("Features")
        ax.set_ylabel("Nodes")

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_filename)
    plt.show()

if __name__ == "__main__":
    # List of directories containing CSV files
    csv_directories = [
        '3xruns/propagate/outputs/dblp/rgcn',
        '3xruns/propagate/outputs/dblp/pnrgcn',
        '3xruns/propagate/outputs/imdb/rgcn',
        '3xruns/propagate/outputs/imdb/pnrgcn'
    ]
    
    # Iterate over each directory, process the CSV files, and save the plots
    for csv_directory in csv_directories:
        # Pattern to match all CSV files
        csv_pattern = os.path.join(csv_directory, '*.csv')
        csv_files = sorted(glob.glob(csv_pattern), key=lambda x: int(os.path.basename(x).split('_')[3]))

        # Create an output filename for the combined plot specific to each directory
        output_filename = f"{csv_directory.replace('/', '_')}_prop_embeddings_plot.svg"
        
        # Generate and save the plots
        plot_embeddings_from_csv(csv_files, output_filename)
