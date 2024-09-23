import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

MASK_VALUE = -1e9

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_num_nodes=1000, temperature=0.5):
        super(PolicyNetwork, self).__init__()
        self.temperature = temperature
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.node_1 = nn.Linear(hidden_dim * 2, max_num_nodes)
        self.node_2 = nn.Linear(hidden_dim * 2, max_num_nodes)
        self.edge_type = nn.Linear(hidden_dim * 2, 3) # 1->2, 2->1, both directions
        self.stop = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index):
        # GCN layers
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        gcn = F.relu(self.gcn3(x, edge_index))
        x1 = torch.mean(gcn, dim=0, keepdim=True)  # Pooling node embeddings
        x2,_ = torch.max(gcn, dim=0, keepdim=True)  # Pooling node embeddings        
        graph_pooling = torch.cat([x1, x2], dim=1)        
        # sample node_id1
        node1_soft = F.gumbel_softmax(self.mask_logits(gcn, edge_index, None, self.node_1(graph_pooling)), tau=self.temperature, hard=False, dim=-1)        
        node1_idx = torch.argmax(node1_soft, dim=-1)
        
        # logits mask for node_id2. Only allow nodes that have no edges to node_id1
        node2_logits = self.mask_logits(gcn, edge_index, node1_idx, self.node_2(graph_pooling))
        
        # Check if all logits are very small (indicating no valid second node)
        if torch.all(node2_logits <= MASK_VALUE):
            return None, None, None, None
        
        node2_soft = F.gumbel_softmax(node2_logits, tau=self.temperature, hard=False, dim=-1)
        node2_idx = torch.argmax(node2_soft, dim=-1)
        
        node1_embedding = gcn[node1_idx]
        node2_embedding = gcn[node2_idx]
        node_embeddings = torch.cat([node1_embedding, node2_embedding], dim=1)
        edge_type_soft = F.gumbel_softmax(self.edge_type(node_embeddings), tau=self.temperature, hard=False, dim=-1)
        # stop signal
        stop_soft = F.gumbel_softmax(self.stop(graph_pooling), tau=self.temperature, hard=False, dim=-1)
        
        return node1_soft, node2_soft, edge_type_soft, stop_soft
    
    def mask_logits(self, x, edge_index, prev_idx, node_logits):
        # mask the logits for non-existing nodes        
        node_logits[0][x.shape[0]:] = MASK_VALUE
        if prev_idx is not None:
            node_logits[0][prev_idx] = MASK_VALUE
            # only allow nodes that have no edges to the previous node
            row, col = edge_index
            mask = (row == prev_idx) | (col == prev_idx)
            connected_nodes = torch.unique(torch.cat([row[mask], col[mask]]))
            node_logits[0][connected_nodes] = MASK_VALUE
        
        return node_logits

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
        x1 = torch.mean(x, dim=0, keepdim=True)
        x2,_ = torch.max(x, dim=0, keepdim=True)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
