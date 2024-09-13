import torch
import numpy as np
import hashlib
from torch_geometric.data import Data
from collections import defaultdict, deque

from config import POSITIONAL_EMBEDDINGS_DIM
from embeddings import get_nvidia_nim_embeddings

def get_sinusoidal_positional_embeddings(bbox, num_rows, num_cols, d_model):
    """
    Generate sinusoidal positional embeddings for a bounding box.

    Args:
        bbox (tuple): A tuple of four values (x_min, y_min, x_max, y_max) representing the bounding box.
        num_rows (int): The height (number of rows) of the image.
        num_cols (int): The width (number of columns) of the image.
        d_model (int): The dimensionality of the encoding (should be an even number).

    Returns:
        np.array: A numpy array of shape (4, d_model) representing the positional encoding for the bounding box.
    """
    
    assert d_model % 2 == 0, "d_model must be an even number."

    def get_positional_encoding(pos, d_model):
        # Create a positional encoding for a single coordinate
        angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model // 2)) / d_model))
        encoding = np.zeros(d_model)
        encoding[0::2] = np.sin(pos * angle_rates)
        encoding[1::2] = np.cos(pos * angle_rates)
        return encoding

    x_min, y_min, x_max, y_max = bbox

    # Normalize the bbox coordinates
    x_min_norm = x_min / num_cols
    y_min_norm = y_min / num_rows
    x_max_norm = x_max / num_cols
    y_max_norm = y_max / num_rows

    # Get positional encodings for each coordinate
    x_min_encoding = get_positional_encoding(x_min_norm, d_model)
    y_min_encoding = get_positional_encoding(y_min_norm, d_model)
    x_max_encoding = get_positional_encoding(x_max_norm, d_model)
    y_max_encoding = get_positional_encoding(y_max_norm, d_model)

    # Concatenate all 4 encodings into a single array
    bbox_encoding = np.array([x_min_encoding, y_min_encoding, x_max_encoding, y_max_encoding])

    return bbox_encoding

def create_graph(elements, d_model=POSITIONAL_EMBEDDINGS_DIM):
    nodes = []
    edges = []

    # Filter out elements with no text
    elements = [element for element in elements if element[1].strip()]

    texts = [element[1] for element in elements]
    embeddings = get_nvidia_nim_embeddings(texts)

    for (element, embedding) in zip(elements, embeddings):
        box, text, label, label_id = element
        # Convert embedding to tensor if it's not already
        embedding_tensor = torch.tensor(embedding) if not isinstance(embedding, torch.Tensor) else embedding
        pos_embeddings = torch.zeros(d_model*4)  # Initialize pos_embeddings to zeros
        nodes.append({
            "text": text,
            "bbox": box,
            "embedding": embedding_tensor,
            "label": label,
            "label_id": label_id,
            "pos_embeddings": pos_embeddings
        })
    
    # Create one-hot encoded labels
    num_classes = 11 # this is the number of classes in doclaynet
    node_labels = torch.zeros((len(nodes), num_classes))
    for i, node in enumerate(nodes):
        node_labels[i, node["label_id"]] = 1

    # Create the tensors for the nodes. It should contain the embedding, the label as one-hot encoded vector and the pos_embeddings
    node_features = torch.stack([
        torch.cat([
            node["embedding"],
            node_labels[i],  # Use the one-hot encoded label
            node["pos_embeddings"]
        ]) for i, node in enumerate(nodes)
    ])

    # Create edge_index tensor if there are edges
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

    # Create a directed graph by using the original Data class
    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    return data, nodes, edges


def get_edge_attributes(elem_i, elem_j):
    box_i = elem_i["bbox"]
    box_j = elem_j["bbox"]
    return [abs(box_i[0] - box_j[0]), abs(box_i[1] - box_j[1])]


def update_coordinates_and_merge_graphs(graphs_nodes_edges, images, d_model=POSITIONAL_EMBEDDINGS_DIM):
    # Initialize empty lists for combined node features, edges, and edge attributes
    combined_x = []
    combined_edge_index = []
    combined_edge_attr = []
    updated_nodes = []
    updated_edges = []
    total_height = 0

    node_offset = 0
    for (graph, nodes, edges), img in zip(graphs_nodes_edges, images):
        x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        # Update node coordinates and add to combined list
        for i, node in enumerate(nodes):
            bbox = node["bbox"]
            updated_bbox = [
                bbox[0],
                bbox[1] + total_height,
                bbox[2],
                bbox[3] + total_height,
            ]
            node["bbox"] = updated_bbox
            updated_nodes.append(node)

        # Add node features and edges to combined lists
        combined_x.append(x)
        combined_edge_index.append(edge_index + node_offset)
        if edge_attr is not None:
            combined_edge_attr.append(edge_attr)

        # Update edge indices to account for node offset
        for edge in edges:
            updated_edges.append((edge[0] + node_offset, edge[1] + node_offset))

        node_offset += x.size(0)
        total_height += img.size[1]    

    # Concatenate all the node features, edges, and edge attributes
    combined_x = torch.cat(combined_x, dim=0)
    combined_edge_index = torch.cat(combined_edge_index, dim=1)
    combined_edge_attr = torch.cat(combined_edge_attr, dim=0) if combined_edge_attr else None

    merged_graph = Data(x=combined_x, edge_index=combined_edge_index, edge_attr=combined_edge_attr)

    # Calculate the sinusoidal positional embeddings for the updated bounding boxes
    for node, node_x in zip(updated_nodes, combined_x):
        bbox = node["bbox"]
        # pos_embeddings is a np array 4 x d_model
        pos_embeddings = get_sinusoidal_positional_embeddings(bbox, total_height, img.size[0], d_model)
        node["pos_embeddings"] = torch.tensor(pos_embeddings.flatten())
        # Update the positional embeddings
        node_x[-d_model*4:] = node["pos_embeddings"]

    return merged_graph, updated_nodes, updated_edges


def tensor_to_tuple(tensor):
    if isinstance(tensor, torch.Tensor):
        return tuple(tensor.tolist())
    return tensor


# def compute_graph_hash(graph, nodes, edges):
#     # Convert tensor attributes in nodes to tuples for consistent sorting
#     sorted_nodes = sorted(nodes, key=lambda x: tuple((k, tensor_to_tuple(v)) for k, v in sorted(x.items())))

#     # Convert edges to tuples if they are tensors
#     sorted_edges = sorted((tuple(edge) if isinstance(edge, torch.Tensor) else edge) for edge in edges)

#     # Create a string representation of the sorted nodes and edges
#     nodes_str = "".join([str(node) for node in sorted_nodes])
#     edges_str = "".join([str(edge) for edge in sorted_edges])

#     # Combine the graph features, sorted nodes, and sorted edges into a single string
#     combined_str = str(graph.x.tolist()) + nodes_str + edges_str

#     # Compute the hash value using SHA-256
#     hash_value = hashlib.sha256(combined_str.encode()).hexdigest()

#     return hash_value


def split_graph(graph, nodes, edges):
    # Find strongly connected components in the directed graph
    components = find_strongly_connected_components(graph.edge_index, graph.num_nodes)

    subgraphs = []
    for component in components:
        # Create subgraphs for each component
        subgraph_nodes = component
        subgraph_node_idx = {node: i for i, node in enumerate(subgraph_nodes)}
        subgraph_edges = [
            (subgraph_node_idx[u], subgraph_node_idx[v])
            for u, v in edges
            if u in subgraph_node_idx and v in subgraph_node_idx
        ]

        # Create the subgraph Data object
        subgraph_x = graph.x[subgraph_nodes]
        subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()
        subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index)

        # Create the subgraph's nodes and edges list
        subgraph_nodes_list = [nodes[node] for node in subgraph_nodes]
        subgraph_edges_list = subgraph_edges

        subgraphs.append((subgraph, subgraph_nodes_list, subgraph_edges_list))

    return subgraphs

def find_strongly_connected_components(edge_index, num_nodes):
    # Implement Kosaraju's algorithm for finding strongly connected components
    def dfs(v, adj, visited, stack):
        visited[v] = True
        for u in adj[v]:
            if not visited[u]:
                dfs(u, adj, visited, stack)
        stack.append(v)

    def reverse_dfs(v, adj, visited, component):
        visited[v] = True
        component.append(v)
        for u in adj[v]:
            if not visited[u]:
                reverse_dfs(u, adj, visited, component)

    # Create adjacency lists
    adj = [[] for _ in range(num_nodes)]
    rev_adj = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        u, v = edge_index[:, i].tolist()
        adj[u].append(v)
        rev_adj[v].append(u)

    # First DFS
    visited = [False] * num_nodes
    stack = []
    for i in range(num_nodes):
        if not visited[i]:
            dfs(i, adj, visited, stack)

    # Second DFS
    visited = [False] * num_nodes
    components = []
    while stack:
        v = stack.pop()
        if not visited[v]:
            component = []
            reverse_dfs(v, rev_adj, visited, component)
            components.append(component)

    return components

def extract_text_from_graph(graph, nodes, edges):    
    node_texts = [node["text"] for node in nodes if "text" in node]
    # Concatenate all text into a single string
    concatenated_text = " ".join(node_texts)
    return concatenated_text