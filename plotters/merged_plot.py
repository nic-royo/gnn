# code to make minecraft water plots of the embeddings, change so it has bigger fonta
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# put source here
csv_directories = [
    '3xruns/3xruns_outputs/allnode_embeddings/dblp_rgcn',
    '3xruns/3xruns_outputs/allnode_embeddings/dblp_pnrgcn',
    '3xruns/3xruns_outputs/allnode_embeddings/imdb_rgcn',
    '3xruns/3xruns_outputs/allnode_embeddings/imdb_pnrgcn',
]

def plot_embeddings_from_csv(csv_files, output_filename):
    # Assuming this function plots embeddings and saves the plot
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=(10, 6))
    
    for csv_file in csv_files:
        data = pd.read_csv(csv_file, index_col=0)  # Assuming the first column is the index
        plt.scatter(data.index, data.iloc[:, 0], label=os.path.basename(csv_file))

    plt.xlabel('Node Index')
    plt.ylabel('Embedding Value')
    plt.title('Embeddings Plot')
    plt.legend()
    plt.grid(True)
    
    # Save plot to the specified directory
    plt.savefig(output_filename)
    plt.close()

if __name__ == "__main__":
    # Iterate over each directory, process the CSV files, and save the plots
    for csv_directory in csv_directories:
        # Pattern to match all CSV files
        csv_pattern = os.path.join(csv_directory, '*.csv')
        csv_files = sorted(glob.glob(csv_pattern), key=lambda x: int(os.path.basename(x).split('_')[3]))

        # Create an output filename for the combined plot specific to each directory
        output_filename = f"{csv_directory.replace('/', '_')}_combined_embeddings_plot.svg"
        
        # Generate and save the plots
        plot_embeddings_from_csv(csv_files, output_filename)
