import wandb
from RGCN_train import main  # Import the main function from your original script
import argparse

def sweep():
    sweep_config = {
        'method': 'bayes',  # Change to grid search
        'metric': {
            'name': 'f1_score',
            'goal': 'maximize'
        },
        'parameters': {
            'hidden_dim': {'values': [32, 64, 128, 256]},
            'num_bases': {'values': [5, 10, None]},
            'num_epochs': {'values': [100, 200, 300, 400, 500]},
            'dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.9},
            'num_hops': {'values': [2, 5, 8]},
            'learning_rate': {'values': [0.001, 0.005, 0.01, 0.1]},
            'weight_decay': {'values': [0.00001, 0.00005, 0.0001, 0.001]},
            'patience': {'values': [50, 100]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=f"Hyper_search-{args.dataset}")

    def train():
        # Initialize with the same project name, but allow for different runs
        wandb.init(project=f"rgcn_base-{args.dataset}", group="rgcn_baseline")
        
        # Update args with values from sweep
        args.hidden_dim = wandb.config.hidden_dim
        args.num_bases = wandb.config.num_bases
        args.dropout = wandb.config.dropout
        args.num_hops = wandb.config.num_hops
        args.num_epochs = wandb.config.num_epochs
        args.learning_rate = wandb.config.learning_rate
        args.weight_decay = wandb.config.weight_decay
        args.patience = wandb.config.patience 
        
        main(args)

    wandb.agent(sweep_id, train, count = 50) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN Hyperparameter Sweep')
    parser.add_argument('--dataset', type=str, default='mag', choices=['imdb', 'dblp', 'yelp', 'mag', 'rcdd'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--print_every', type=int, default=10, help='Print every n epochs')

    args = parser.parse_args()
    sweep()
