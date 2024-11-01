import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

MASK_VALUE = -1e9

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_num_nodes=1000, temperature=0.1):
        super(PolicyNetwork, self).__init__()
        self.temperature = temperature
        self.gcn1 = GATConv(input_dim, hidden_dim, heads=8)
        self.gcn2 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        # self.gcn3 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        # self.gcn4 = GATConv(hidden_dim * 8, hidden_dim, heads=8)        
        self.gcn5 = GATConv(hidden_dim * 8, hidden_dim, heads=1)
        self.node_1_1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.node_1_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.node_1_3 = nn.Linear(hidden_dim, hidden_dim)
        # self.node_1_4 = nn.Linear(hidden_dim, hidden_dim)
        self.node_1_5 = nn.Linear(hidden_dim, hidden_dim)
        self.node_1_logits = nn.Linear(hidden_dim, max_num_nodes)
        self.node_2_1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.node_2_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.node_2_3 = nn.Linear(hidden_dim, hidden_dim)
        # self.node_2_4 = nn.Linear(hidden_dim, hidden_dim)
        self.node_2_5 = nn.Linear(hidden_dim, hidden_dim)
        self.node_2_logits = nn.Linear(hidden_dim, max_num_nodes)
        self.edge_type_1 = nn.Linear(hidden_dim * 5, hidden_dim)
        self.edge_type_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.edge_type_3 = nn.Linear(hidden_dim, hidden_dim)
        # self.edge_type_4 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_type_5 = nn.Linear(hidden_dim, hidden_dim)
        self.edge_type_logits = nn.Linear(hidden_dim, 3)
        self.stop_1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.stop_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.stop_3 = nn.Linear(hidden_dim, hidden_dim)
        # self.stop_4 = nn.Linear(hidden_dim, hidden_dim)
        self.stop_5 = nn.Linear(hidden_dim, hidden_dim)
        self.stop_logits = nn.Linear(hidden_dim, 2)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(self, x, edge_index):
        # GCN layers
        x = F.leaky_relu(self.gcn1(x, edge_index))
        x = F.leaky_relu(self.gcn2(x, edge_index))
        # x = F.leaky_relu(self.gcn3(x, edge_index))
        # x = F.leaky_relu(self.gcn4(x, edge_index))
        gcn = F.leaky_relu(self.gcn5(x, edge_index))
        x1,_ = torch.min(gcn, dim=0, keepdim=True)  # Pooling node embeddings
        x2 = torch.mean(gcn, dim=0, keepdim=True)  # Pooling node embeddings
        x3,_ = torch.max(gcn, dim=0, keepdim=True)  # Pooling node embeddings
        graph_pooling = torch.cat([x1, x2, x3], dim=1)
        # sample node_id1                
        node1 = F.leaky_relu(self.node_1_1(graph_pooling))
        # shortcut = node1
        node1 = F.leaky_relu(self.node_1_2(node1))
        # node1 = F.leaky_relu(self.node_1_3(node1))
        # node1 = F.leaky_relu(self.node_1_4(node1))
        node1 = F.leaky_relu(self.node_1_5(node1))
        node1_logits = F.leaky_relu(self.node_1_logits(node1))
        node1_logits = self.mask_logits(gcn, edge_index, None, node1_logits)

        node1_soft = F.gumbel_softmax(node1_logits, tau=self.temperature, hard=False, dim=-1)        
        node1_idx = torch.argmax(node1_soft, dim=-1)
        
        # logits mask for node_id2. Only allow nodes that have no edges to node_id1
        node2 = F.leaky_relu(self.node_2_1(graph_pooling))
        # shortcut = node2
        node2 = F.leaky_relu(self.node_2_2(node2))
        # node2 = F.leaky_relu(self.node_2_3(node2))
        # node2 = F.leaky_relu(self.node_2_4(node2))
        node2 = F.leaky_relu(self.node_2_5(node2))
        node2_logits = F.leaky_relu(self.node_2_logits(node2))
        node2_logits = self.mask_logits(gcn, edge_index, node1_idx, node2_logits)
        
        # Check if all logits are very small (indicating no valid second node)
        if torch.all(node2_logits <= MASK_VALUE):
            return None, None, None, None
        
        node2_soft = F.gumbel_softmax(node2_logits, tau=self.temperature, hard=False, dim=-1)
        node2_idx = torch.argmax(node2_soft, dim=-1)
        
        node1_embedding = gcn[node1_idx]
        node2_embedding = gcn[node2_idx]
        node_embeddings = torch.cat([x1, x2, x3, node1_embedding, node2_embedding], dim=1)
        edge_type = F.leaky_relu(self.edge_type_1(node_embeddings))
        # shortcut = edge_type    
        edge_type = F.leaky_relu(self.edge_type_2(edge_type))
        # edge_type = F.leaky_relu(self.edge_type_3(edge_type))
        # edge_type = F.leaky_relu(self.edge_type_4(edge_type))
        edge_type = F.leaky_relu(self.edge_type_5(edge_type))
        edge_type_logits = F.leaky_relu(self.edge_type_logits(edge_type))
        edge_type_soft = F.gumbel_softmax(edge_type_logits, tau=self.temperature, hard=False, dim=-1)
        # stop signal
        stop = F.leaky_relu(self.stop_1(graph_pooling))
        shortcut = stop 
        stop = F.leaky_relu(self.stop_2(stop))
        # stop = F.leaky_relu(self.stop_3(stop))
        # stop = F.leaky_relu(self.stop_4(stop))
        stop = F.leaky_relu(self.stop_5(stop))
        stop_logits = F.leaky_relu(self.stop_logits(stop))
        stop_soft = F.gumbel_softmax(stop_logits, tau=self.temperature, hard=False, dim=-1)        
        
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
        self.gcn1 = GATConv(input_dim, hidden_dim, heads=8)
        self.gcn2 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        # self.gcn3 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        # self.gcn4 = GATConv(hidden_dim * 8, hidden_dim, heads=8)
        self.gcn5 = GATConv(hidden_dim * 8, hidden_dim, heads=1)
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.gcn1(x, edge_index))
        x = F.leaky_relu(self.gcn2(x, edge_index))
        # x = F.leaky_relu(self.gcn3(x, edge_index))
        # x = F.leaky_relu(self.gcn4(x, edge_index))
        gcn = F.leaky_relu(self.gcn5(x, edge_index))
        x1,_ = torch.min(gcn, dim=0, keepdim=True)
        x2 = torch.mean(gcn, dim=0, keepdim=True)
        x3,_ = torch.max(gcn, dim=0, keepdim=True)
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.leaky_relu(self.fc1(x))
        # shortcut = x
        x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        # x = F.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x
