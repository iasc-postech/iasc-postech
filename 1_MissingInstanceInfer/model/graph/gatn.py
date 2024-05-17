import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv

class ResidualGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels=in_channels, out_channels=in_channels, heads=heads, concat=False)
        self.conv2 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=False)
        self.conv_res = GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=False)

    def forward(self, data, edge_index):
        
        intermediate = self.conv1(data, edge_index)
        intermediate = self.conv2(intermediate, edge_index)

        res = self.conv_res(data, edge_index)

        return intermediate + res

class GAT_res(nn.Module):
    def __init__(self, channel_dim=[1028, 128, 256, 1028], heads_num=6, class_numb=210):
        super().__init__()
        self.conv1 = ResidualGAT(in_channels=channel_dim[0], out_channels=channel_dim[1], heads=heads_num)
        self.conv2 = ResidualGAT(in_channels=channel_dim[1], out_channels=channel_dim[2], heads=heads_num)
        self.conv3 = ResidualGAT(in_channels=channel_dim[2], out_channels=channel_dim[3], heads=heads_num)
        self.relu = nn.ReLU()

        self.label_embed = nn.Embedding(num_embeddings=class_numb, embedding_dim=256)

    def forward(self, graph):
        
        data, edge_index = graph.x, graph.edge_index

        residual = data
        intermediate = self.conv1(data, edge_index)
        intermediate = self.conv2(intermediate, edge_index)
        out = self.conv3(intermediate, edge_index)
        final = residual + self.relu(out)

        feature_list = []
        for slice_point in graph.ptr[1:]:
            feature = final[slice_point.item()-1]
            feature_list.append(feature.unsqueeze(0))
        
        feature_batch = torch.cat(feature_list, dim=0)

        return feature_batch

class GAT(nn.Module):
    def __init__(self, channel_dim=[1028, 128, 256, 1028], heads_num=6, class_numb=210):
        super().__init__()
        self.conv1 = GATConv(in_channels=channel_dim[0], out_channels=channel_dim[1], heads=heads_num, concat=False)
        self.conv2 = GATConv(in_channels=channel_dim[1], out_channels=channel_dim[2], heads=heads_num, concat=False)
        self.conv3 = GATConv(in_channels=channel_dim[2], out_channels=channel_dim[3], heads=heads_num, negative_slope=0, concat=False)
        self.label_embed = nn.Embedding(num_embeddings=class_numb, embedding_dim=256)

    def forward(self, graph):
        
        data, edge_index = graph.x, graph.edge_index

        residual = data
        intermediate = self.conv1(data, edge_index)
        intermediate = self.conv2(intermediate, edge_index)
        out = self.conv3(intermediate, edge_index)
        final = out + residual
        
        feature_list = []
        for slice_point in graph.ptr[1:]:
            feature = final[slice_point.item()-1]
            feature_list.append(feature.unsqueeze(0))
        
        feature_batch = torch.cat(feature_list, dim=0)

        return feature_batch