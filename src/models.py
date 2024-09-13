import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


# Define the neural network models for actions
class AddNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AddNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, node1_emb, node2_emb):
        x = torch.cat([node1_emb, node2_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Define the neural network models for actions
class AddBothNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AddBothNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, node1_emb, node2_emb):
        x = torch.cat([node1_emb, node2_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class RemoveNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RemoveNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)        
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, node1_emb, node2_emb):
        x = torch.cat([node1_emb, node2_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 4)  # actions: add, add_both, remove, or stop

    def forward(self, x, edge_index):
        # GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        x = torch.mean(x, dim=0, keepdim=True)  # Pooling node embeddings

        # Fully connected layer to output action probabilities
        x = self.fc(x)
        return F.softmax(x, dim=-1)
    
# Define the critic network
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CriticNetwork, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        x = F.relu(self.gcn3(x, edge_index))
        gcn_output = torch.mean(x, dim=0, keepdim=True)
        x = F.relu(self.fc1(gcn_output))
        x = self.fc2(x)
        return x
