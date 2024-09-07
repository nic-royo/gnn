# propagate embeddings, can use all nodes or 50, and can also use all node types o only target node, saves results to a csv file
import torch
from torch_geometric.utils import degree
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F
import torch_geometric
import numpy as np
import torch_geometric.transforms as T
from torch_scatter import scatter
from torch_geometric.nn import RGCNConv
from torch_geometric.datasets import DBLP, IMDB, RCDD, AMiner
from sklearn.metrics import f1_score
import argparse
from typing import Optional
import torch.linalg as TLA
from torch import Tensor
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import RGCNConv, norm as geom_norm  # Added import for PairNorm
import pandas as pd
import os
import random

class RGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, num_layers=2):
        super(RGCN, self).__init__()
        self.relu = nn.ReLU()

        self.node_types, self.edge_types = metadata

        self.linear_projections = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=True))
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=True))
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=True))

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
        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])
        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type

        x_latents = [x]

        # Apply RGCN layers
        for conv in self.convs[:-1]:
            x = self.relu(conv(x, edge_index, edge_type))
            x_latents.append(x.detach())
        x = self.convs[-1](x, edge_index, edge_type)
        return x, x_latents
 
class pairnormRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, num_layers=2):
        super(pairnormRGCN, self).__init__()
        self.relu = nn.ReLU()

        self.node_types, self.edge_types = metadata

        self.linear_projections = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.pairnorms = nn.ModuleList()

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
        self.pairnorms.append(geom_norm.PairNorm())  # PairNorm first conv

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
            self.pairnorms.append(geom_norm.PairNorm())  # PairNorm after each middle conv

        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
        self.pairnorms.append(geom_norm.PairNorm())  # PairNorm final conv

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
        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = []

        # Apply RGCN layers with PairNorm
        for conv, pairnorm in zip(self.convs[:-1], self.pairnorms[:-1]):
            x = self.relu(conv(x, edge_index, edge_type))
            x = pairnorm(x)  # do PairNorm after each RGCN layer
            x_latents.append(x)
        x = self.convs[-1](x, edge_index, edge_type)
        x = self.pairnorms[-1](x)  # do PairNorm after the final RGCN layer
        return x, x_latents

class resRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, num_layers=2):
        super(resRGCN, self).__init__()
        self.relu = nn.ReLU()

        self.node_types, self.edge_types = metadata

        self.linear_projections = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), bias=False,num_bases=num_bases))
        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))

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
        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = []

        # Apply RGCN layers
        for i, conv in enumerate(self.convs[:-1]):
            x_ = self.relu(conv(x, edge_index, edge_type))
            x_latents.append(x)

            # Residual connection - no bias
            if i == 0:
                x = x_
            else:
                x = x + x_

        out = self.convs[-1](x, edge_index, edge_type)

        return out, x_latents

class ggRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, num_layers=2, p=2):
        super(ggRGCN, self).__init__()
        self.relu = nn.ReLU()
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.p = p

        self.linear_projections_in = nn.ModuleDict()

        self.rgcn_layers = nn.ModuleList()
        self.gg_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.rgcn_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
            self.gg_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))

        #decoder layer
        self.linear_output=nn.Linear(hidden_dim, out_dim)

    def create_linear_projections(self, x_dict):
        device = next(self.parameters()).device
        for node_type, x in x_dict.items():
            if node_type not in self.linear_projections_in:
                in_dim = x.size(1)
                self.linear_projections_in[node_type] = nn.Linear(in_dim, self.rgcn_layers[0].in_channels).to(device)

    def forward(self, x_dict, data):
        device = next(self.parameters()).device

        # Create linear projections if they don't exist
        self.create_linear_projections(x_dict)

        # Linear projections for each node type
        x_dict = {node_type: self.relu(self.linear_projections_in[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = []

        # GG methods
        for i in range(self.num_layers):
            x_ = torch.relu(self.rgcn_layers[i](x, edge_index, edge_type))

            # Estimate of tau ind order to calculate correct tau
            x_hat = torch.relu(self.gg_layers[i](x, edge_index, edge_type))
            tau = torch.tanh(scatter((torch.abs(x[edge_index[0]] - x[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0, dim_size=x.size(0), reduce='mean'))
            x = (1 - tau) * x + tau * x_
            x_latents.append(x)

        x = self.linear_output(x)
        return x, x_latents

def load_dataset(dataset_name, root='/tmp/'):
    if dataset_name == 'imdb':
        return IMDB(root=root + 'IMDB')[0]
    elif dataset_name == 'dblp':
        return DBLP(root=root + 'DBLP', transform=T.Constant(node_types='conference'))[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_target_node_type(dataset_name):
    if dataset_name == 'imdb':
        return 'movie'
    elif dataset_name == 'dblp':
        return 'author'
    else:
        raise ValueError(f"Unknown target node type for dataset: {dataset_name}")

def get_num_classes(data, target_node_type):
    return int(data[target_node_type].y.max().item()) + 1

def generate_random_features(data):
    torch.manual_seed(2105)
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        feature_dim = 128  
        data[node_type].x = torch.randn((num_nodes, feature_dim))
    return data

def propagate_embeddings(model, data, node_types_option, target_node_type):
    model.eval()
    with torch.no_grad():
        _, x_latents = model(data.x_dict, data)
    
    # Ensure x_latents is a dictionary mapping node types to their embeddings
    x_latents_dict = {node_type: x_latent for node_type, x_latent in zip(data.node_types, x_latents)}
    
    if node_types_option == 'all_node_types':
        return x_latents_dict
    elif node_types_option == 'target_node_type':
        return {target_node_type: x_latents_dict[target_node_type]}
    else:
        raise ValueError(f"Invalid node_types_option: {node_types_option}")
    
def main(args):
    data = load_dataset(args.dataset, root=args.data_root)
    data = generate_random_features(data)

    # get target node type and number of classes
    target_node_type = get_target_node_type(args.dataset)
    num_classes = get_num_classes(data, target_node_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    pyg.seed_everything(seed=42)

    if args.model_type == 'pnrgcn':
        model = pairnormRGCN(data.metadata(), args.hidden_dim, num_classes,
                             args.num_bases, num_layers=args.num_hops).to(device)
    elif args.model_type == 'rgcn':
        model = RGCN(data.metadata(), args.hidden_dim, num_classes,
                     args.num_bases, num_layers=args.num_hops).to(device)
    elif args.model_type == 'resrgcn':
        model = resRGCN(data.metadata(), args.hidden_dim, num_classes,
                     args.num_bases, num_layers=args.num_hops).to(device)
    elif args.model_type == 'ggrgcn':
        model = ggRGCN(data.metadata(), args.hidden_dim, num_classes,
                     args.num_bases, num_layers=args.num_hops).to(device)
    else:
        print("Model not found")
        return  
    
    print(model)

    print("Propagating the embeddings...")
    # Propagate embeddings based on the node type option
    x_latents = propagate_embeddings(model, data, args.node_types_option, target_node_type)

    # Conditionally normalize the embeddings
    if args.embedding_option == 'normalize':
        x_latents = {node_type: F.normalize(x_latent, p=2, dim=1) for node_type, x_latent in x_latents.items()}
        # can normalize along different dimensions, gives the same results
        #x_latents = {node_type: F.normalize(x_latent, p=2, dim=0) for node_type, x_latent in x_latents.items()}

    
    # Convert embeddings to numpy arrays
    x_latents_np = {node_type: x_latent.detach().cpu().numpy() for node_type, x_latent in x_latents.items()}

    all_embeddings = []
    all_node_indices = []

    for node_type, embeddings in x_latents_np.items():
        num_nodes = data[node_type].num_nodes
        if args.use_all_nodes:
            node_indices = torch.arange(num_nodes)
        else:
            node_indices = torch.randperm(num_nodes)[:50]

        final_layer_embeddings = embeddings[node_indices]
        
        # make DataFrame with node type and embeddings
        df = pd.DataFrame(final_layer_embeddings, index=[f'{node_type}_Node_{i}' for i in node_indices])
        df['Node_Type'] = node_type  # Add a column for node type
        all_embeddings.append(df)
        all_node_indices.append(node_indices)
    
    # Combine all DataFrames into one
    combined_df = pd.concat(all_embeddings, axis=0)

    output_dir = f"results/prop_embeddings/all_node_embeddings_norm/{args.dataset}_{args.model_type}"
    os.makedirs(output_dir, exist_ok=True)  

    file_suffix = f"_{args.num_hops}_layer_all_nodes_{'norm' if args.embedding_option == 'normalize' else 'notnorm'}.csv"
    csv_filename = os.path.join(output_dir, f"{args.dataset}_{args.model_type}{file_suffix}")

    combined_df.to_csv(csv_filename)

    print(f'All node embeddings saved to {csv_filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['dblp', 'imdb'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--num_hops', type=int, default=32, help='Number of hops in RGCN')
    parser.add_argument('--embedding_option', type=str, required=True, choices=['normalize', 'no_normalize'], help='Option to either normalize or not normalize the embeddings')
    parser.add_argument('--model_type', type=str, default='rgcn', choices=['rgcn', 'pnrgcn', 'resrgcn', 'ggrgcn'], help='Type of RGCN model to use')
    parser.add_argument('--use_all_nodes', type=lambda x: x.lower() == 'true', default=True, help='Whether to use all nodes or a random sample of 50 nodes')
    parser.add_argument('--node_types_option', type=str, default='all_node_types', choices=['all_node_types', 'target_node_type'], help='Whether to save embeddings for all node types or just the target node type')

    args = parser.parse_args()
    main(args)