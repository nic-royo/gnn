import torch
from torch_geometric.utils import degree
import torch.nn as nn
import torch_geometric as pyg

import torch
import os
import torch.nn.functional as F
import torch_geometric
import numpy as np
import torch_geometric.transforms as T

from torch_scatter import scatter
from torch_geometric import EdgeIndex
from torch_geometric.nn import RGCNConv, norm as geom_norm
from torch_geometric.datasets import IMDB, DBLP, Yelp, AMiner
from sklearn.metrics import f1_score
import argparse
import wandb

import pandas as pd
from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, remove_self_loops

from typing import Optional

import torch
import torch.linalg as TLA
from torch import Tensor

def dirichlet_energy_roth(x, edge_index, rw=True):
    with torch.no_grad():
        src, dst = edge_index
        deg = degree(src, num_nodes=x.shape[0])
        x = x / torch.norm(x)
        if not rw:
            x = x / torch.sqrt(deg + 0.0).view(-1, 1)
        energy = torch.norm(x[src] - x[dst], dim=1, p=2) ** 2.0
        energy = energy.mean()
        energy *= 0.5
    return float(energy.mean().detach().cpu())

@torch.no_grad
def dirichlet_energy_old(x, edge_index):
    A = torch_geometric.utils.to_dense_adj(edge_index)
    D = torch.diag(torch.sum(A, axis=1))
    L = D - A
    d_e = torch.matmul(torch.matmul(x.T, L), x).squeeze(0)
    energy = torch.trace(d_e)
    return energy

@torch.no_grad
def build_adj_dict(num_nodes, edge_index):
    adj_dict = {nodeid: [] for nodeid in range(num_nodes)}
    for eidx in range(edge_index.shape[1]):
        ctail, chead = edge_index[0, eidx].item(), edge_index[1, eidx].item()
        if not chead in adj_dict[ctail]:
            adj_dict[ctail].append(chead)
    return adj_dict

@torch.no_grad
def dirichlet_energy_adj(
    feat_matrix,
    edge_index=None,
    adj_dict=None,
    p=2):

    num_nodes = feat_matrix.shape[0]
    de = 0

    if adj_dict is None:
        adj_dict = build_adj_dict(num_nodes=num_nodes, edge_index=edge_index)

    def inner(x_i, x_js):
        return TLA.vector_norm(x_i - x_js, ord=p, dim=1).square().sum()

    for node_index in range(num_nodes):
        own_feat_vector = feat_matrix[[node_index], :]
        nbh_feat_matrix = feat_matrix[adj_dict[node_index], :]
        de += inner(own_feat_vector, nbh_feat_matrix)

    return (de / num_nodes).item()


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

        self.create_linear_projections(x_dict)

        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        x = torch.cat([x_dict[node_type] for node_type in self.node_types])
        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type

        x_latents = [x]

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
        self.pairnorms.append(geom_norm.PairNorm())

        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
            self.pairnorms.append(geom_norm.PairNorm())

        self.convs.append(RGCNConv(hidden_dim, out_dim, num_relations=len(self.edge_types),bias=False, num_bases=num_bases))
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

        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = []


        for conv, pairnorm in zip(self.convs[:-1], self.pairnorms[:-1]):
            x = self.relu(conv(x, edge_index, edge_type))
            x = pairnorm(x)
            x_latents.append(x)
        x = self.convs[-1](x, edge_index, edge_type)
        x = self.pairnorms[-1](x)
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


        self.create_linear_projections(x_dict)


        x_dict = {node_type: self.relu(self.linear_projections[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = []

        for i, conv in enumerate(self.convs[:-1]):
            x_ = self.relu(conv(x, edge_index, edge_type))
            x_latents.append(x)

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

        self.linear_output=nn.Linear(hidden_dim, out_dim)

    def create_linear_projections(self, x_dict):
        device = next(self.parameters()).device
        for node_type, x in x_dict.items():
            if node_type not in self.linear_projections_in:
                in_dim = x.size(1)
                self.linear_projections_in[node_type] = nn.Linear(in_dim, self.rgcn_layers[0].in_channels).to(device)

    def forward(self, x_dict, data):
        device = next(self.parameters()).device

        self.create_linear_projections(x_dict)

        x_dict = {node_type: self.relu(self.linear_projections_in[node_type](x.to(device)))
                for node_type, x in x_dict.items()}

        x = torch.cat([x_dict[node_type] for node_type in self.node_types])

        index_data = data.to_homogeneous()
        edge_index = index_data.edge_index
        edge_type = index_data.edge_type
        x_latents = []

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

def generate_random_features(data):
    torch.manual_seed(2105)
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        feature_dim = 128
        data[node_type].x = torch.randn((num_nodes, feature_dim))
    return data

def get_num_classes(data, target_node_type):
    return int(data[target_node_type].y.max().item()) + 1

def main(args):

    data = load_dataset(args.dataset, root=args.data_root)
    data = generate_random_features(data)
    target_node_type = get_target_node_type(args.dataset)
    num_classes = get_num_classes(data, target_node_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    pyg.seed_everything(250)

    if args.model_type == 'pnrgcn':
        model = pairnormRGCN(data.metadata(), args.hidden_dim, num_classes,
                             args.num_bases, num_layers=args.num_hops).to(device)
    elif args.model_type == 'resrgcn':
        model = resRGCN(data.metadata(), args.hidden_dim, num_classes,
                        args.num_bases, num_layers=args.num_hops).to(device)
    elif args.model_type == 'ggrgcn':
        model = ggRGCN(data.metadata(), args.hidden_dim, num_classes,
                       args.num_bases, num_layers=args.num_hops).to(device)
    else:
        model = RGCN(data.metadata(), args.hidden_dim, num_classes,
                     args.num_bases, num_layers=args.num_hops).to(device)


    _, x_latents = model(data.x_dict, data)
    data.to('cpu')
    hom_data =  data.to_homogeneous()
    edge_index = hom_data.edge_index


    def get_2_hop_index(edge_index, num_nodes):
        from torch_geometric import EdgeIndex
        edge_index = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes))
        edge_index = edge_index.sort_by('row')[0]
        edge_index2 = edge_index.matmul(edge_index)[0].as_tensor()
        edge_index2, _ = remove_self_loops(edge_index2)
        new_edge_index = coalesce(edge_index2)
        return new_edge_index

    def filter_edges_by_type_sets(two_hop_index, node_types, source_type, dest_type):

        src_types = node_types[two_hop_index[0]]
        dst_types = node_types[two_hop_index[1]]

        src_mask = torch.tensor([t.item() == source_type for t in src_types])
        dst_mask = torch.tensor([t.item() == dest_type for t in dst_types])

        mask = src_mask & dst_mask

        filtered_edge_index = two_hop_index[:, mask]

        return filtered_edge_index

    two_hop_index = get_2_hop_index(edge_index, hom_data.num_nodes)
    filtered_edge_index = filter_edges_by_type_sets(two_hop_index, hom_data.node_type, 0, 0)

    def eval_oversmooth_adj(method, x_latents, edge_index):
        de = np.array([method(x.to('cpu'), edge_index.to('cpu')) for x in x_latents], np.float64)
        de_log = np.log10(de)
        return de, de_log

    os.makedirs('results/two_hop/', exist_ok=True)


    de_adj, de_log_adj = eval_oversmooth_adj(dirichlet_energy_adj, x_latents, filtered_edge_index)

    df_adj = pd.DataFrame({
        'de': de_adj,
        'de_log': de_log_adj
    })

    filename = f'results/two_hop/oversmooth_adj_{args.model_type}_layers_{args.num_hops}_2hop_{args.dataset}.csv'
    df_adj.to_csv(filename, index=False)
    print(f'Just did {args.model_type}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RGCN for Heterogeneous Datasets')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['imdb', 'dblp'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='/tmp/', help='Root directory for datasets')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num_bases', type=int, default=None, help='Number of bases for RGCN')
    parser.add_argument('--num_hops', type=int, default=16, help='Number of hops in RGCN')
    parser.add_argument('--model_type', type=str, default='rgcn', choices=['rgcn', 'pnrgcn', 'resrgcn', 'ggrgcn'], help='Type of RGCN model to use')

    args = parser.parse_args()
    main(args)