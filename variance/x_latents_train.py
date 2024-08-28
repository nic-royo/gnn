# 'minecraft water' plots code for doing the training and visualization of the embeddings, only does 50 random nodes, for now
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

        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Val Acc: {acc:.4f}, Val F1-Score: {f1:.4f}')

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
    elif args.model_type == 'resrgcn':
        model = resRGCN(data.metadata(), args.hidden_dim, num_classes,
                        args.num_bases, dropout=args.dropout, num_layers=args.num_hops).to(device)
    else:
        print("Model not found")
        return  # Exit the function if the model is not found
    
    print(model)

    if args.train:
        print("Training the model...")
        train_model(model, data, target_node_type, epochs=args.epochs, learning_rate=args.learning_rate)
    else:
        print("Only propagating the embeddings...")

    pyg.seed_everything(250)
    _, x_latents = model(data.x_dict, data)

    # Conditionally normalize the embeddings
    if args.embedding_option == 'normalize':
        x_latents = [F.normalize(x_latent, p=2, dim=1) for x_latent in x_latents]
    
    x_latents_np = [x_latent.detach().cpu().numpy() for x_latent in x_latents]

    # Select 50 target nodes from the final layer's embeddings OR random sampled 50 nodes
    if args.random_sample is True:
        # Random sample 
        num_target_nodes = data[target_node_type].num_nodes
        target_node_indices = torch.randperm(num_target_nodes)[:50]
    else:
        # Or first 50 nodes
        target_node_indices = torch.arange(50)
    final_layer_embeddings = x_latents_np[-1][target_node_indices]

    plt.figure(figsize=(15, 10))
    sns.heatmap(final_layer_embeddings, cmap='coolwarm', cbar=True, yticklabels=[f'Node {i}' for i in target_node_indices])

    plt.xlabel("Features")
    plt.ylabel("Nodes")
    #title = f"{args.dataset}, {args.num_hops} layers, {'normalized' if args.embedding_option == 'normalize' else 'not normalized'}, {args.model_type} model, {'trained' if args.train else 'propagated'}, {'random sampled' if args.random_sample else 'first 50'}"
    title = f"{args.num_hops} Layers"    
    plt.title(title)
    filename = f"final_plots/minecraft_dblp/{args.model_type}_{'norm' if args.embedding_option == 'normalize' else 'notnorm'}_{'trained' if args.train else 'propagated'}_{args.num_hops}_layer_{args.dataset}_{'random_sample' if args.random_sample else 'first50'}.svg"
    plt.savefig(filename)
    plt.show()

    print('Finished.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['dblp', 'imdb', 'rcdd'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--num_hops', type=int, default=32, help='Number of hops in RGCN')
    parser.add_argument('--embedding_option', type=str, required=True, choices=['normalize', 'no_normalize'], help='Option to either normalize or not normalize the embeddings')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--train', type=bool, required=True, help='Whether to train the model (True or False)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay')
    parser.add_argument('--model_type', type=str, default='rgcn', choices=['rgcn', 'pnrgcn', 'resrgcn', 'ggrgcn'], help='Type of RGCN model to use')
    parser.add_argument('--random_sample', type=lambda x: x.lower() == 'true', default=False, help='Whether to randomly sample 50 nodes for visualization (True or False)')

    args = parser.parse_args()
    main(args)