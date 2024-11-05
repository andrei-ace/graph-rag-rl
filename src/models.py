import torch
import torch.nn as nn
from torch_geometric.nn import GENConv, DeepGCNLayer
import torch.nn.functional as F

MASK_VALUE = -1e9

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_num_nodes=1000, temperature=0.1, num_layers=7):
        super(PolicyNetwork, self).__init__()
        self.temperature = temperature        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, 
                           aggr='softmax', t=1.0, 
                           learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1)
            self.layers.append(layer)
        
        self.node_1_logits = nn.Linear(hidden_dim * 3, max_num_nodes)
        self.node_2_logits = nn.Linear(hidden_dim * 3, max_num_nodes)
        self.edge_type_logits = nn.Linear(hidden_dim * 5, 3)        
        self.stop_logits = nn.Linear(hidden_dim * 3, 2)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def forward(self, x, edge_index):
        # GCN layers
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        min_gcn,_ = torch.min(x, dim=0, keepdim=True)  # Pooling node embeddings
        mean_gcn = torch.mean(x, dim=0, keepdim=True)  # Pooling node embeddings
        max_gcn,_ = torch.max(x, dim=0, keepdim=True)  # Pooling node embeddings
        graph_pooling = torch.cat([min_gcn, mean_gcn, max_gcn], dim=1)
        
        # sample node_id1 
        node1_logits = self.node_1_logits(graph_pooling)
        node1_logits = self.mask_logits(x, edge_index, None, node1_logits)
        node1_soft = F.gumbel_softmax(node1_logits, tau=self.temperature, hard=False, dim=-1)  
        
        # node1_soft = F.softmax(node1_logits, dim=-1)
        node1_idx = torch.argmax(node1_soft, dim=-1)
        
        # logits mask for node_id2. Only allow nodes that have no edges to node_id1        
        node2_logits = self.node_2_logits(graph_pooling)
        node2_logits = self.mask_logits(x, edge_index, node1_idx, node2_logits)
        
        # Check if all logits are very small (indicating no valid second node)
        if torch.all(node2_logits <= MASK_VALUE):
            return None, None, None, None
        
        node2_soft = F.gumbel_softmax(node2_logits, tau=self.temperature, hard=False, dim=-1)
        # node2_soft = F.softmax(node2_logits, dim=-1)
        node2_idx = torch.argmax(node2_soft, dim=-1)
        
        node1_embedding = x[node1_idx]
        node2_embedding = x[node2_idx]
        node_embeddings = torch.cat([min_gcn, mean_gcn, max_gcn, node1_embedding, node2_embedding], dim=1)        
        edge_type_logits = self.edge_type_logits(node_embeddings)
        edge_type_soft = F.gumbel_softmax(edge_type_logits, tau=self.temperature, hard=False, dim=-1)
        # edge_type_soft = F.softmax(edge_type_logits, dim=-1)
        # stop signal        
        stop_logits = self.stop_logits(graph_pooling)
        stop_soft = F.gumbel_softmax(stop_logits, tau=self.temperature, hard=False, dim=-1)        
        # stop_soft = F.softmax(stop_logits, dim=-1)
        
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
    def __init__(self, input_dim, hidden_dim, num_layers=7):
        super(CriticNetwork, self).__init__()
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, 
                           aggr='softmax', t=1.0, 
                           learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_dim, elementwise_affine=True)
            act = nn.ReLU()
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1)
            self.layers.append(layer)

        self.lin = nn.Linear(hidden_dim * 3, 1)
        self.lin.weight.data.fill_(0)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index):
        # GCN layers
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        
        min_gcn, _ = torch.min(x, dim=0, keepdim=True)
        mean_gcn = torch.mean(x, dim=0, keepdim=True)
        max_gcn, _ = torch.max(x, dim=0, keepdim=True)
        x = torch.cat([min_gcn, mean_gcn, max_gcn], dim=1)        
        return self.lin(x)
