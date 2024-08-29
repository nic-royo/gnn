# Code for making water plots, only training, trains on target node types, makes csv files for embeddings
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
import os
import pandas as pd

class RGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout=0.8, num_layers=2):
        super(RGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.node_types, self.edge_types = metadata

        self.linear_projections = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=False))
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
            x = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x_latents.append(x.detach())
        x = self.convs[-1](x, edge_index, edge_type)
        return x, x_latents
 
class pairnormRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout=0.8, num_layers=2):
        super(pairnormRGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.node_types, self.edge_types = metadata

        self.linear_projections = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.pairnorms = nn.ModuleList()

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
        self.pairnorms.append(geom_norm.PairNorm())  # PairNorm first conv

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
            self.pairnorms.append(geom_norm.PairNorm())  # PairNorm after each middle conv

        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types), num_bases=num_bases))
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
        x_dict = {node_type: self.dropout(self.relu(self.linear_projections[node_type](x.to(device))))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = [x]

        # Apply RGCN layers with PairNorm
        for conv, pairnorm in zip(self.convs[:-1], self.pairnorms[:-1]):
            x = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x = pairnorm(x)  # do PairNorm after each RGCN layer
            x_latents.append(x)
        x = self.convs[-1](x, edge_index, edge_type)
        x = self.pairnorms[-1](x)  # do PairNorm after the final RGCN layer
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

def train(model, optimizer, data, target_node_type):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x_dict, data)
    target_out = out[:data[target_node_type].num_nodes]
    loss = F.cross_entropy(target_out[data[target_node_type].train_mask],
                           data[target_node_type].y[data[target_node_type].train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)

def val(model, data, target_node_type):
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

def train_model(model, data, target_node_type, epochs=10, learning_rate=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_f1 = 0
    best_model = None

    for epoch in range(epochs):
        loss = train(model, optimizer, data, target_node_type)
        acc, f1 = val(model, data, target_node_type)

        #print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val Acc: {acc:.4f}, Val F1-Score: {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            best_model = model.state_dict()

    # Load the best model
    model.load_state_dict(best_model)


def main(args):
    data = load_dataset(args.dataset, root=args.data_root)

    target_node_type = get_target_node_type(args.dataset)
    num_classes = get_num_classes(data, target_node_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Create model
    if args.model_type == 'pnrgcn':
        model = pairnormRGCN(data.metadata(), args.hidden_dim, num_classes,
                             args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
    elif args.model_type == 'rgcn':
        model = RGCN(data.metadata(), args.hidden_dim, num_classes,
                     args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
    else:
        print("Model not found")
        return  # Exit the function if the model is not found

    print(model)

    # Training the model
    print("Training the model...")
    train_model(model, data, target_node_type, epochs=args.epochs, learning_rate=args.learning_rate)

    # After training, get the embeddings
    pyg.seed_everything(250)
    _, x_latents = model(data.x_dict, data)

    # Conditionally normalize the embeddings
    if args.embedding_option == 'normalize':
        x_latents = [F.normalize(x_latent, p=2, dim=1) for x_latent in x_latents]

    x_latents_np = [x_latent.detach().cpu().numpy() for x_latent in x_latents]

    # Create the output directory if it doesn't exist
    output_dir = os.path.join('results', 'train_embeddings')
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings only for the last layer (layer 32 when num_hops is 32)
    layer = args.num_hops - 1  # Get the index of the last layer
    start_idx = 0
    all_embeddings = []
    
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        end_idx = start_idx + num_nodes
        
        node_embeddings = x_latents_np[layer][start_idx:end_idx]
        node_types = [node_type] * num_nodes
        
        df = pd.DataFrame(node_embeddings)
        df['Node_Type'] = node_types
        all_embeddings.append(df)
        
        start_idx = end_idx
    
    # Concatenate all embeddings for the last layer
    combined_df = pd.concat(all_embeddings, ignore_index=True)
    
    # Generate the file name
    file_suffix = f"_{args.model_type}_{args.num_hops}_layer_all_nodes_{'norm' if args.embedding_option == 'normalize' else 'notnorm'}.csv"
    csv_filename = os.path.join(output_dir, f"{args.dataset}_layer_{args.num_hops}{file_suffix}")

    # Save the CSV file
    combined_df.to_csv(csv_filename, index=False)
    print(f"Saved embeddings for layer {args.num_hops} to {csv_filename}")

    print('Finished.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['dblp', 'imdb', 'rcdd'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_hops', type=int, default=8, help='Number of hops in RGCN')
    parser.add_argument('--embedding_option', type=str, required=True, choices=['normalize', 'no_normalize'], help='Option to either normalize or not normalize the embeddings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay')
    parser.add_argument('--model_type', type=str, default='rgcn', choices=['rgcn', 'pnrgcn', 'resrgcn', 'ggrgcn'], help='Type of RGCN model to use')

    args = parser.parse_args()
    main(args)