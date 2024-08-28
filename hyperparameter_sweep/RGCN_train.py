import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
import torch_geometric.transforms as T

from torch_scatter import scatter
from torch_geometric.nn import RGCNConv
from torch_geometric.datasets import IMDB, DBLP, Yelp, AMiner, RCDD, OGB_MAG
from sklearn.metrics import f1_score
import argparse
import wandb
from typing import Optional
import torch.linalg as TLA
from torch import Tensor

def build_adj_dict(num_nodes: int, edge_index: Tensor) -> dict[int, list[int]]:
    """
    Build an adjacency dictionary from the given edge_index tensor.
    
    Args:
        num_nodes (int): The total number of nodes.
        edge_index (Tensor): The edge_index tensor of shape (2, num_edges).
        
    Returns:
        A dictionary where the keys are node IDs and the values are lists of neighboring node IDs.
    """
    # initialize adjacency dict with empty neighborhoods for all nodes
    adj_dict: dict[int, list[int]] = {nodeid: [] for nodeid in range(num_nodes)}

    # iterate through all edges and add head nodes to adjacency list of tail nodes
    for eidx in range(edge_index.shape[1]):
        ctail, chead = edge_index[0, eidx].item(), edge_index[1, eidx].item()

        if not chead in adj_dict[ctail]:
            adj_dict[ctail].append(chead)

    return adj_dict

@torch.no_grad()
def dirichlet_energy(
    feat_matrix: Tensor,
    edge_index: Optional[Tensor] = None,
    adj_dict: Optional[dict] = None,
    p: Optional[int | float] = 2,
) -> float:
    """
    Calculate the Dirichlet energy of a feature matrix given the edge_index or adjacency dictionary.
    
    Args:
        feat_matrix (Tensor): The feature matrix of shape (num_nodes, feature_dim).
        edge_index (Tensor, optional): The edge_index tensor of shape (2, num_edges).
        adj_dict (dict, optional): The adjacency dictionary, where the keys are node IDs and the values are lists of neighboring node IDs.
        p (int | float, optional): The order of the vector norm used in the Dirichlet energy calculation. Default is 2.
        
    Returns:
        The Dirichlet energy of the feature matrix.
    """
    if (edge_index is None) and (adj_dict is None):
        raise ValueError("Neither 'edge_index' nor 'adj_dict' was provided.")
    if (edge_index is not None) and (adj_dict is not None):
        raise ValueError(
            "Both 'edge_index' and 'adj_dict' were provided. Only one should be passed."
        )

    num_nodes: int = feat_matrix.shape[0]
    de: Tensor = 0

    if adj_dict is None:
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index)

    def inner(x_i: Tensor, x_js: Tensor) -> Tensor:
        return TLA.vector_norm(x_i - x_js, ord=p, dim=1).square().sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]

        de += inner(own_feat_vector, nbh_feat_matrix)

    return torch.sqrt(de / num_nodes).item()

class RGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout=0.8, num_layers=2):
        super(RGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.node_types, self.edge_types = metadata
        
        self.linear_projections = nn.ModuleDict()
        
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types), num_bases=num_bases))

    def create_linear_projections(self, x_dict):
        device = next(self.parameters()).device
        for node_type, x in x_dict.items():
            if node_type not in self.linear_projections:
                in_dim = x.size(1)
                self.linear_projections[node_type] = nn.Linear(in_dim, self.convs[0].in_channels).to(device)

    def forward(self, x_dict, data):
        device = next(self.parameters()).device

        # Create linear projections if they don't exist
        self.create_linear_projections(x_dict)

        # Linear projections for each node type
        x_dict = {node_type: self.dropout(self.relu(self.linear_projections[node_type](x.to(device))))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])
        
        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = [x]
        
        # Apply RGCN layers
        for conv in self.convs[:-1]:
            x = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x_latents.append(x)
        x = self.convs[-1](x, edge_index, edge_type)
        return x, x_latents
    
    @torch.no_grad()
    def dirichlet_energy(self, x, edge_index):
        return dirichlet_energy(x, edge_index)
        
    @torch.no_grad()    
    def eval_oversmoothing(self, x_dict, data):
        convert_data = data.to_homogeneous()
        edge_index = convert_data.edge_index
        _, x_latents = self(x_dict, data)
        d_energies = np.array([self.dirichlet_energy(x, edge_index) for x in x_latents], np.float64)
        d_energies_log = np.log10(d_energies)
        return d_energies_log

def load_dataset(dataset_name, root='/tmp/'):
    if dataset_name == 'imdb':
        return IMDB(root=root + 'IMDB')[0]
    elif dataset_name == 'dblp':
        return DBLP(root=root + 'DBLP', transform=T.Constant(node_types='conference'))[0]
    elif dataset_name == 'yelp':
        return Yelp(root=root + 'Yelp')[0]
    elif dataset_name == 'mag':
        transform_mag = T.ToUndirected(merge=True)
        return OGB_MAG(root=root + 'OGB_MAG', preprocess='metapath2vec', transform=transform_mag[0])
    elif dataset_name == 'rcdd':
        return RCDD(root=root + 'RCDD')[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_target_node_type(dataset_name):
    if dataset_name == 'imdb':
        return 'movie'
    elif dataset_name == 'dblp':
        return 'author'
    elif dataset_name == 'yelp':
        return 'business'
    elif dataset_name == 'mag':
        return 'paper'
    elif dataset_name == 'rcdd':
        return 'node-1'
    else:
        raise ValueError(f"Unknown target node type for dataset: {dataset_name}")

def get_num_classes(data, target_node_type):
    return int(data[target_node_type].y.max().item()) + 1

def main(args):
    wandb.login(key='4e8cb6191f0ce50becd1fde96d5c4caf25f6ecfc')
    wandb.init(project=args.dataset, group="rgcn", config=args)

    # Load dataset
    data = load_dataset(args.dataset, root=args.data_root)
    target_node_type = get_target_node_type(args.dataset)
    num_classes = get_num_classes(data, target_node_type)
    
    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Create model
    model = RGCN(data.metadata(), args.hidden_dim, num_classes, 
                 args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def train():
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x_dict, data)
        target_out = out[:data[target_node_type].num_nodes]
        loss = F.cross_entropy(target_out[data[target_node_type].train_mask],
                               data[target_node_type].y[data[target_node_type].train_mask])
        loss.backward()
        optimizer.step()
        return float(loss)

    def val():
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x_dict, data)
            target_out = out[:data[target_node_type].num_nodes]
            pred = target_out.argmax(dim=1)
            correct = pred[data[target_node_type].val_mask] == data[target_node_type].y[data[target_node_type].val_mask]
            acc = int(correct.sum()) / int(data[target_node_type].val_mask.sum())
            f1 = f1_score(data[target_node_type].y[data[target_node_type].val_mask].cpu().numpy(),
                          pred[data[target_node_type].val_mask].cpu().numpy(), average='micro')
        return acc, f1
    
    def test():
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x_dict, data)
            target_out = out[:data[target_node_type].num_nodes]
            pred = target_out.argmax(dim=1)
            correct = pred[data[target_node_type].test_mask] == data[target_node_type].y[data[target_node_type].test_mask]
            acc = int(correct.sum()) / int(data[target_node_type].test_mask.sum())
            f1 = f1_score(data[target_node_type].y[data[target_node_type].test_mask].cpu().numpy(),
                          pred[data[target_node_type].test_mask].cpu().numpy(), average='micro')
        return acc, f1
    
    best_f1 = 0
    best_model = None
    epochs_since_improvement = 0  # Track the number of epochs since last improvement

    for epoch in range(args.num_epochs):
        loss = train()
        acc, f1 = test()

        wandb.log({
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": acc,
            "f1_score": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model = model.state_dict()
            epochs_since_improvement = 0  # Reset the counter when improvement is seen
        else:
            epochs_since_improvement += 1  # Increment the counter if no improvement

        if epochs_since_improvement >= args.patience:
            print(f'Early stopping at epoch {epoch+1:03d} due to no improvement in the last {args.patience} epochs.')
            break

        if (epoch + 1) % args.print_every == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}, Test F1-Score: {f1:.4f}')

    # Load the best model
    model.load_state_dict(best_model)

    # Calculate and log Dirichlet energy only once, after training
    d_energies_log = model.eval_oversmoothing(data.x_dict, data)
    
    # Save the numpy array to a file
    #np.save('de_energies/baseRGCN_d_energies_log.npy', d_energies_log)

    wandb.log({
        "dirichlet_energy": d_energies_log.tolist(),
        "final_accuracy": acc,
        "final_f1_score": best_f1,
    })
    
    print(f'Final Test Acc: {acc:.4f}, F1-Score: {best_f1:.4f}')
    print(f'Dirichlet Energy: {d_energies_log.tolist()}')

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='imdb', choices=['imdb', 'dblp', 'yelp', 'mag', 'rcdd'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_hops', type=int, default=20, help='Number of hops in RGCN')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--print_every', type=int, default=10, help='Print every n epochs')
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to wait for improvement before early stopping')

    args = parser.parse_args()
    main(args)