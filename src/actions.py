import numpy as np
import torch


def get_possible_node_pairs(x, edge_index):
    node_count = x.size(0)
    existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
    
    possible_add_pairs = []
    for i in range(node_count):
        for j in range(node_count):
            if i != j and (i, j) not in existing_edges:
                possible_add_pairs.append((i, j))

    possible_remove_pairs = list(existing_edges)
    
    return possible_add_pairs, possible_remove_pairs


def get_possible_node_pairs_no_cycles(x, edge_index):
    node_count = x.size(0)
    existing_edges = set(tuple(edge) for edge in edge_index.t().tolist())
    
    # Create a dictionary to represent the graph
    graph = {i: set() for i in range(node_count)}
    for edge in existing_edges:
        graph[edge[0]].add(edge[1])
    
    def has_path(start, end, visited=None):
        if visited is None:
            visited = set()
        if start == end:
            return True
        visited.add(start)
        for neighbor in graph[start]:
            if neighbor not in visited and has_path(neighbor, end, visited):
                return True
        return False

    possible_add_pairs = []
    for i in range(node_count):
        for j in range(node_count):
            if i != j and (i, j) not in existing_edges and not has_path(j, i):
                possible_add_pairs.append((i, j))

    possible_remove_pairs = list(existing_edges)
    
    return possible_add_pairs, possible_remove_pairs


def apply_action(graph, edges, action_name, node_pair):
    if action_name == "add":
        # Add edge to graph (directed)
        node1, node2 = node_pair
        graph.edge_index = torch.cat(
            [graph.edge_index, torch.tensor([[node1], [node2]], device=graph.x.device)], dim=1
        )
        edges.append((node1, node2))
    elif action_name == "add_both":
        # Check if node1 -> node2 is already added, if not add it
        node1, node2 = node_pair
        if not any((graph.edge_index[0] == node1) & (graph.edge_index[1] == node2)):
            graph.edge_index = torch.cat(
                [graph.edge_index, torch.tensor([[node1], [node2]], device=graph.x.device)], dim=1
            )
            edges.append((node1, node2))
        # Add edge from node2 -> node1 if it doesn't exist
        if not any((graph.edge_index[0] == node2) & (graph.edge_index[1] == node1)):
            graph.edge_index = torch.cat(
                [graph.edge_index, torch.tensor([[node2], [node1]], device=graph.x.device)], dim=1
            )
            edges.append((node2, node1))
    elif action_name == "remove":
        # Remove edge from graph (directed)
        node1, node2 = node_pair
        mask = ~(
            torch.logical_and(graph.edge_index[0] == node1, graph.edge_index[1] == node2)
        )
        graph.edge_index = graph.edge_index[:, mask]
        edges.remove((node1, node2))

    return graph, edges


def revert_action(graph, edges, action_name, node_pair):
    if action_name == "add":
        # Remove the previously added edge (directed)
        node1, node2 = node_pair
        mask = ~(
            torch.logical_and(graph.edge_index[0] == node1, graph.edge_index[1] == node2)
        )
        graph.edge_index = graph.edge_index[:, mask]
        edges.remove((node1, node2))
    elif action_name == "add_both":
        # Remove both previously added edges (bidirectional)
        node1, node2 = node_pair
        mask = ~(
            (torch.logical_and(graph.edge_index[0] == node1, graph.edge_index[1] == node2) |
             torch.logical_and(graph.edge_index[0] == node2, graph.edge_index[1] == node1))
        )
        graph.edge_index = graph.edge_index[:, mask]
        edges = [edge for edge in edges if edge not in [(node1, node2), (node2, node1)]]
    elif action_name == "remove":
        # Add the previously removed edge (directed)
        node1, node2 = node_pair
        graph.edge_index = torch.cat(
            [graph.edge_index, torch.tensor([[node1], [node2]], device=graph.x.device)], dim=1
        )
        edges.append((node1, node2))

    return graph, edges


# Example usage
def sample_action(graph, add_net, add_both_net, remove_net, critic_net, action_probs):
    
    # Sample an action based on policy output 
    actions = ["add", "add_both", "remove", "stop"]
    chosen_action_index = torch.multinomial(action_probs, 1).item()
    chosen_action_name = actions[chosen_action_index]

    possible_add_pairs, possible_remove_pairs = get_possible_node_pairs(graph.x, graph.edge_index)
    
    if chosen_action_name == "add" and possible_add_pairs != []:
        add_probs = []
        for pair in possible_add_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            add_prob = add_net(node1_emb, node2_emb)
            add_probs.append(add_prob.item())
        add_probs = torch.tensor(add_probs, device=graph.x.device)
        add_probs = torch.nn.functional.softmax(add_probs, dim=0)
        chosen_idx = torch.multinomial(add_probs, 1).item()
        chosen_action_pair = possible_add_pairs[chosen_idx]
        chosen_prob = action_probs[:, 0]
        # Compute and print critic value
        node1_emb = graph.x[chosen_action_pair[0]].unsqueeze(0)
        node2_emb = graph.x[chosen_action_pair[1]].unsqueeze(0)
        one_hot = torch.tensor([1, 0, 0, 0], device=graph.x.device).unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            one_hot,
            node1_emb,
            node2_emb,
            chosen_prob.unsqueeze(0),
        )
    elif chosen_action_name == "add_both" and possible_add_pairs != []:
        add_both_probs = []
        for pair in possible_add_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            add_both_prob = add_both_net(node1_emb, node2_emb)
            add_both_probs.append(add_both_prob.item())
        add_both_probs = torch.tensor(add_both_probs, device=graph.x.device)
        add_both_probs = torch.nn.functional.softmax(add_both_probs, dim=0)
        chosen_idx = torch.multinomial(add_both_probs, 1).item()
        chosen_action_pair = possible_add_pairs[chosen_idx]
        chosen_prob = action_probs[:, 1]
        # Compute and print critic value
        node1_emb = graph.x[chosen_action_pair[0]].unsqueeze(0)
        node2_emb = graph.x[chosen_action_pair[1]].unsqueeze(0)
        one_hot = torch.tensor([0, 1, 0, 0], device=graph.x.device).unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            one_hot,
            node1_emb,
            node2_emb,
            chosen_prob.unsqueeze(0),
        )
    elif chosen_action_name == "remove" and possible_remove_pairs != []:
        remove_probs = []
        for pair in possible_remove_pairs:
            node1_emb = graph.x[pair[0]].unsqueeze(0)
            node2_emb = graph.x[pair[1]].unsqueeze(0)
            remove_prob = remove_net(node1_emb, node2_emb)
            remove_probs.append(remove_prob.item())
        remove_probs = torch.tensor(remove_probs, device=graph.x.device)
        remove_probs = torch.nn.functional.softmax(remove_probs, dim=0)
        chosen_idx = torch.multinomial(remove_probs, 1).item()
        chosen_action_pair = possible_remove_pairs[chosen_idx]
        chosen_prob = action_probs[:, 2]
        # Compute and print critic value
        node1_emb = graph.x[chosen_action_pair[0]].unsqueeze(0)
        node2_emb = graph.x[chosen_action_pair[1]].unsqueeze(0)
        one_hot = torch.tensor([0, 0, 1, 0], device=graph.x.device).unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            one_hot,
            node1_emb,
            node2_emb,
            chosen_prob.unsqueeze(0),
        )
    else:
        chosen_action_name = "stop"
        chosen_action_pair = None
        chosen_prob = action_probs[:, 3]
        # Compute and print critic value
        placeholder = graph.x.mean(dim=0).unsqueeze(0)
        one_hot = torch.tensor([0, 0, 0, 1], device=graph.x.device).unsqueeze(0)
        critic_value = critic_net(
            graph.x,
            graph.edge_index,
            one_hot,
            placeholder,
            placeholder,
            chosen_prob.unsqueeze(0),
        )

    return chosen_action_name, chosen_action_pair, chosen_prob, critic_value