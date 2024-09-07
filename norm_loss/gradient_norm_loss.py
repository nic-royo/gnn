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
from torch import Tensor
import matplotlib.pyplot as plt
from torch_geometric.nn import RGCNConv, norm as geom_norm
import os
import pandas as pd


class RGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout, num_layers):
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
                self.linear_projections[node_type] = nn.Linear(in_dim, self.convs[0].in_channels, bias=False).to(device)

    def forward(self, x_dict, data):
        device = next(self.parameters()).device


        self.create_linear_projections(x_dict)

        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        x = torch.cat([x_dict[node_type] for node_type in self.node_types])
        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type

        x_latents = [x]

        for conv in self.convs[:-1]:
            x = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x_latents.append(x.detach())
        x = self.convs[-1](x, edge_index, edge_type)
        return x, x_latents

class pairnormRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout, num_layers=2):
        super(pairnormRGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.node_types, self.edge_types = metadata

        self.linear_projections = nn.ModuleDict()

        self.convs = nn.ModuleList()
        self.pairnorms = nn.ModuleList()

        self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=False))
        self.pairnorms.append(geom_norm.PairNorm())

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases, bias=False, root_weight=True))
            self.pairnorms.append(geom_norm.PairNorm())

        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types), num_bases=num_bases))
        self.pairnorms.append(geom_norm.PairNorm())

    def create_linear_projections(self, x_dict):
        device = next(self.parameters()).device
        for node_type, x in x_dict.items():
            if node_type not in self.linear_projections:
                in_dim = x.size(1)
                self.linear_projections[node_type] = nn.Linear(in_dim, self.convs[0].in_channels).to(device)

    def forward(self, x_dict, data):
        device = next(self.parameters()).device


        self.create_linear_projections(x_dict)


        x_dict = {node_type: self.dropout(self.relu(self.linear_projections[node_type](x.to(device))))
                for node_type, x in x_dict.items()}


        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = [x]


        for conv, pairnorm in zip(self.convs[:-1], self.pairnorms[:-1]):
            x = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x = pairnorm(x)
            x_latents.append(x)
        x = self.convs[-1](x, edge_index, edge_type)
        x = self.pairnorms[-1](x)
        return x, x_latents

class resRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout, num_layers=2):
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


        self.create_linear_projections(x_dict)


        x_dict = {node_type: self.dropout(self.relu(self.linear_projections[node_type](x.to(device))))
                for node_type, x in x_dict.items()}


        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = [x]


        for i, conv in enumerate(self.convs[:-1]):
            x_ = self.dropout(self.relu(conv(x, edge_index, edge_type)))
            x_latents.append(x)

            if i == 0:
                x = x_
            else:
                x = x + x_

        out = self.convs[-1](x, edge_index, edge_type)

        return out, x_latents

class ggRGCN(nn.Module):
    def __init__(self, metadata, hidden_dim, out_dim, num_bases, dropout, num_layers=2, p=2):
        super(ggRGCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.node_types, self.edge_types = metadata
        self.num_layers = num_layers
        self.p = p

        self.linear_projections = nn.ModuleDict()

        self.rgcn_layers = nn.ModuleList()
        self.gg_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.rgcn_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))
            self.gg_layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types), num_bases=num_bases))

        self.linear_output=nn.Linear(hidden_dim, out_dim)

    def create_linear_projections(self, x_dict):
        device = next(self.parameters()).device
        for node_type, x in x_dict.items():
            if node_type not in self.linear_projections:
                in_dim = x.size(1)
                self.linear_projections[node_type] = nn.Linear(in_dim, self.rgcn_layers[0].in_channels).to(device)

    def forward(self, x_dict, data):
        device = next(self.parameters()).device

        self.create_linear_projections(x_dict)

        x_dict = {node_type: self.dropout(self.relu(self.linear_projections[node_type](x.to(device))))
                for node_type, x in x_dict.items()}

        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = [x]

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
        return DBLP(root=root + 'dblp', transform=T.Constant(node_types='conference'))[0]
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

def get_grad_norm():
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def main(args):
    data = load_dataset(args.dataset, root=args.data_root)
    data = pyg.transforms.NormalizeFeatures()(data)
    target_node_type = get_target_node_type(args.dataset)

    num_classes = get_num_classes(data, target_node_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
     pyg.seed_everything(3256)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    def train():
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x_dict, data)
        target_out = out[:data[target_node_type].num_nodes]
        loss = F.cross_entropy(target_out[data[target_node_type].train_mask],
                                data[target_node_type].y[data[target_node_type].train_mask])

        # # In order to clip the norm for short experiments
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


        loss.backward()

        grad_norm = torch.sqrt(sum([
                (p.grad.cpu().norm()**2)
                for p in model.parameters()
                if p.grad is not None
            ])).numpy()

        optimizer.step()
        return float(loss), grad_norm

    def train_test():
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x_dict, data)
            target_out = out[:data[target_node_type].num_nodes]
            pred = target_out.argmax(dim=1)
            correct = pred[data[target_node_type].train_mask] == data[target_node_type].y[data[target_node_type].train_mask]
            acc = int(correct.sum()) / int(data[target_node_type].train_mask.sum())
            f1 = f1_score(data[target_node_type].y[data[target_node_type].train_mask].cpu().numpy(),
                          pred[data[target_node_type].train_mask].cpu().numpy(), average='micro')
        return acc, f1

    def val():
        model.eval()
        with torch.no_grad():
            out, _ = model(data.x_dict, data)
            target_out = out[:data[target_node_type].num_nodes]
            pred = target_out.argmax(dim=1)
            loss = F.cross_entropy(target_out[data[target_node_type].val_mask],
                               data[target_node_type].y[data[target_node_type].val_mask])
            correct = pred[data[target_node_type].val_mask] == data[target_node_type].y[data[target_node_type].val_mask]
            acc = int(correct.sum()) / int(data[target_node_type].val_mask.sum())
            f1 = f1_score(data[target_node_type].y[data[target_node_type].val_mask].cpu().numpy(),
                          pred[data[target_node_type].val_mask].cpu().numpy(), average='micro')
        return float(loss), acc, f1

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


    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'grad_norm':[],
    }
    print("iam here")
    for epoch in range(args.epochs):

        train_loss, grad_norm = train()
        train_acc, train_f1 = train_test()
        val_loss, val_acc, val_f1 = val()

        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['train_f1'].append(train_f1)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        metrics['val_f1'].append(val_f1)
        metrics['grad_norm'].append(grad_norm)


        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')


    print(metrics['train_loss'])
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'plot_data/training_metrics_layer_{args.num_hops}_{args.model_type}_{args.dataset}_short.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['imdb', 'dblp'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--dropout', type=float, default=0.42, help='Dropout rate')
    parser.add_argument('--num_hops', type=int, default=8, help='Number of hops in RGCN')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00005, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--print_every', type=int, default=10, help='Print every n epochs')
    parser.add_argument('--model_type', type=str, default='rgcn', choices=['rgcn', 'pnrgcn', 'resrgcn', 'ggrgcn'], help='Type of RGCN model to use')
   # parser.add_argument('--max_norm', type=float, default=1.0, help='norm clipping values')
   # remove to test gradient_clipping

    args = parser.parse_args()
    main(args)