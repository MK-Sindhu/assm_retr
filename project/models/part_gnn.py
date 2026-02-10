import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

class PartGNN(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_channels, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
        )

        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, embed_dim),
            )
        )

        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        embed = global_mean_pool(x, batch)
        embed = self.ln(embed)
        embed = self.dropout(embed)

        return embed
