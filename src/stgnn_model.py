import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """Spatial Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj, support)
        return output

class STGNNBlock(nn.Module):
    """Spatio-Temporal Block: Temporal Conv -> Spatial GCN -> Temporal Conv"""
    def __init__(self, in_channels, spatial_channels, out_channels):
        super(STGNNBlock, self).__init__()
        self.temporal1 = nn.Conv2d(in_channels, spatial_channels, kernel_size=(1, 3), padding=(0, 1))
        self.gcn = GraphConvolution(spatial_channels, spatial_channels)
        self.temporal2 = nn.Conv2d(spatial_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        x = self.relu(self.temporal1(x))
        
        b, c, n, t = x.shape
        x_gcn = x.permute(0, 3, 2, 1).reshape(-1, n, c)
        x_gcn = self.relu(self.gcn(x_gcn, adj))
        x = x_gcn.reshape(b, t, n, c).permute(0, 3, 2, 1)
        
        x = self.relu(self.temporal2(x))
        return x

class TaxiSTGNN(nn.Module):
    def __init__(self, num_nodes=263, in_dim=1, hidden_dim=32, out_steps=1):
        super(TaxiSTGNN, self).__init__()
        self.block1 = STGNNBlock(in_dim, hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_steps)

    def forward(self, x, adj):
        x = self.block1(x, adj)
        x = x[:, :, :, -1]
        x = x.permute(0, 2, 1)
        out = self.fc(x)
        return out.squeeze(-1)