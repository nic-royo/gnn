# code to save the f1 scores
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
import csv
import os


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

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=False))
        self.pairnorms.append(geom_norm.PairNorm())  # PairNorm first conv

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=True))
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

class resRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout=0.8, num_layers=2):
        super(resRGCN, self).__init__()
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
        for i, conv in enumerate(self.convs[:-1]):
            x_ = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x_latents.append(x)
            
            # Residual connection - no bias
            if i == 0:
                x = x_
            else:
                x = x + x_

        out = self.convs[-1](x, edge_index, edge_type)
        
        return out, x_latents

class ggRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout=0.5, num_layers=2, p=2):
        super(ggRGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.p = p
    
        self.linear_projections_in = nn.ModuleDict()
        
        self.rgcn_layers = nn.ModuleList()
        self.gg_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.rgcn_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
            self.gg_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
        
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
        x_dict = {node_type: self.dropout(self.relu(self.linear_projections_in[node_type](x.to(device))))
                for node_type, x in x_dict.items()}

        # Combine all node features
        x = torch.cat([x_dict[node_type] for node_type in self.node_types])
        
        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = [x] 
        
        # GG methods
        for i in range(self.num_layers):
            x_ = torch.relu(self.rgcn_layers[i](x, edge_index, edge_type))
            
            # Estimate of tau ind order to calculate correct tau
            x_hat = torch.relu(self.gg_layers[i](x, edge_index, edge_type))
            tau = torch.tanh(scatter((torch.abs(x[edge_index[0]] - x[edge_index[1]]) ** self.p).squeeze(-1),
                                 edge_index[0], 0, dim_size=x.size(0), reduce='mean'))
            x = (1 - tau) * x + tau * x_
            x_latents.append(x)            
        
        x = self.dropout(x)    
        x = self.linear_output(x)
        return x, x_latents

def load_dataset(dataset_name, root='/tmp/'):
    if dataset_name == 'imdb':
        return IMDB(root=root + 'IMDB')[0]
    elif dataset_name == 'dblp':
        return DBLP(root=root + 'DBLP', transform=T.Constant(node_types='conference'))[0]
    elif dataset_name == 'mag':
        return AMiner(root=root + 'AMiner')[0]
    elif dataset_name == 'rcdd':
        return RCDD(root=root + 'RCDD')[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_target_node_type(dataset_name):
    if dataset_name == 'imdb':
        return 'movie'
    elif dataset_name == 'dblp':
        return 'author'
    elif dataset_name == 'mag':
        return 'paper'
    elif dataset_name == 'rcdd':
        return 'item'
    else:
        raise ValueError(f"Unknown target node type for dataset: {dataset_name}")

def get_num_classes(data, target_node_type):
    return int(data[target_node_type].y.max().item()) + 1

def main(args):
    data = load_dataset(args.dataset, root=args.data_root)
    target_node_type = get_target_node_type(args.dataset)
    num_classes = get_num_classes(data, target_node_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Lists to store the best F1 scores for each run
    best_micro_f1_scores = []
    best_macro_f1_scores = []

    for run in range(3):
        print(f"Starting run {run + 1}")

        # Create model
        if args.model_type == 'pnrgcn':
            model = pairnormRGCN(data.metadata(), args.hidden_dim, num_classes,
                                 args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
        elif args.model_type == 'resrgcn':
            model = resRGCN(data.metadata(), args.hidden_dim, num_classes,
                            args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
        elif args.model_type == 'ggrgcn':
            model = ggRGCN(data.metadata(), args.hidden_dim, num_classes,
                           args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
        elif args.model_type == 'rgcn':
            model = RGCN(data.metadata(), args.hidden_dim, num_classes,
                         args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
        else:
            print("Model not found")
            return

        print(model)

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
                micro_f1 = f1_score(data[target_node_type].y[data[target_node_type].test_mask].cpu().numpy(),
                                    pred[data[target_node_type].test_mask].cpu().numpy(), average='micro')
                macro_f1 = f1_score(data[target_node_type].y[data[target_node_type].test_mask].cpu().numpy(),
                                    pred[data[target_node_type].test_mask].cpu().numpy(), average='macro')
            return micro_f1, macro_f1
        
        best_f1 = 0
        best_model = None
        epochs_since_improvement = 0

        for epoch in range(args.num_epochs):
            loss = train()
            micro_f1, macro_f1 = val()

            print(f'Run: {run + 1}, Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Test Micro F1-Score: {micro_f1:.4f}, Test Macro F1-Score: {macro_f1:.4f}')

            if micro_f1 > best_f1:
                best_f1 = micro_f1
                best_model = model.state_dict()
                best_macro_f1 = macro_f1
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            
            if epochs_since_improvement >= args.patience:
                print(f'Early stopping at epoch {epoch+1:03d} due to no improvement in the last {args.patience} epochs.')
                break

        # Load the best model and get its scores
        model.load_state_dict(best_model)
        best_micro_f1, best_macro_f1 = test()
        best_micro_f1_scores.append(best_micro_f1)
        best_macro_f1_scores.append(best_macro_f1)

        print(f'Run {run + 1} - Final Test Micro F1-Score: {best_micro_f1:.4f}, Macro F1-Score: {best_macro_f1:.4f}')

    # Calculate mean and variance
    micro_f1_mean = np.mean(best_micro_f1_scores)
    micro_f1_var = np.var(best_micro_f1_scores)
    macro_f1_mean = np.mean(best_macro_f1_scores)
    macro_f1_var = np.var(best_macro_f1_scores)

    # Save results to CSV
    csv_filename = f'{args.model_type}_{args.num_hops}_{args.dataset}_layers_results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Model', 'Num Layers', 'Micro F1 Mean', 'Micro F1 Variance', 'Macro F1 Mean', 'Macro F1 Variance', 'Dataset'])
        csv_writer.writerow([args.model_type, args.num_hops, micro_f1_mean, micro_f1_var, macro_f1_mean, macro_f1_var, args.dataset])

    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['dblp', 'imdb', 'rcdd'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_hops', type=int, default=32, help='Number of hops in RGCN')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay')
    parser.add_argument('--model_type', type=str, default='rgcn', choices=['rgcn', 'pnrgcn', 'resrgcn', 'ggrgcn'], help='Type of RGCN model to use')

    args = parser.parse_args()
    main(args)