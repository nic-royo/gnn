# plot variance of features in final layer embeddings, change directory lcoation for different ones
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_variance_from_csv(csv_files, output_filename):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(csv_files)))

    for csv_file, color in zip(csv_files, colors):
        # Load the CSV file
        df = pd.read_csv(csv_file, index_col=0)
        
        # Calculate variance of each feature across nodes
        variances = df.var(axis=0)
        
        # Extract the number of layers (num_hops) from the filename
        num_hops = os.path.basename(csv_file).split('_')[3]
        
        # Plot variance vs feature index
        plt.plot(range(1, len(variances) + 1), variances, marker='o', label=f'{num_hops} layers', color=color)

    # Title, labels, and legend
    plt.title('Variance of Features in Final Layer Embeddings')
    plt.xlabel('Feature Index')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)
    
    # Save and show the plot
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
        output_filename = f"{csv_directory.replace('/', '_')}_feature_variance_plot.svg"
        
        # Generate and save the plots
        plot_variance_from_csv(csv_files, output_filename)