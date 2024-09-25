from math import sqrt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models import CriticNetwork, PolicyNetwork
from rag import rag
from graphs import find_strongly_connected_components
import concurrent.futures
import os
from statistics import variance

MAX_STEPS = 64

class PPO:
    def __init__(self, input_dim, shaped_reward_coef=0.1, device="cpu", episodes=10):
        hidden_dim = 64
        # Initialize networks
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.critic_net = CriticNetwork(input_dim, hidden_dim)
        self.device = torch.device(device)
        self.shaped_reward_coef = shaped_reward_coef
        self.policy_net.to(self.device)
        self.critic_net.to(self.device)
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters())
            + list(self.critic_net.parameters()),
            lr=1e-3,
        )
        # Initialize loss functions for PPO
        self.value_loss_fn = torch.nn.MSELoss()        
        self.episodes = episodes

    def calculate_shaped_reward(self, graph):
        num_nodes = graph.num_nodes
        # Get the strongly connected components
        sccs = find_strongly_connected_components(graph.edge_index, num_nodes)
        scc_len = len(sccs)
        target_scc = sqrt(num_nodes)

        # Reward for how close the number of SCCs is to the square root of the number of nodes
        num_scc_reward = 1.0 - abs(scc_len - target_scc) / target_scc

        # Reward for low variance in SCC sizes
        scc_sizes = [len(scc) for scc in sccs]
        size_variance = variance(scc_sizes) if len(scc_sizes) > 1 else 0
        max_possible_variance = ((num_nodes - scc_len + 1) ** 2) * (scc_len - 1) / (4 * scc_len)
        size_variance_reward = 1.0 - (size_variance / max_possible_variance if max_possible_variance > 0 else 0)

        # Penalty for large SCCs
        avg_scc_size = num_nodes / scc_len
        large_scc_penalty = sum([len(scc) for scc in sccs if len(scc) > 2 * avg_scc_size]) / num_nodes

        # Combine the rewards and penalties (you can adjust the weights as needed)
        reward = 0.4 * num_scc_reward + 0.4 * size_variance_reward - 0.2 * large_scc_penalty
        return reward

    def evaluate_graph_value(self, graph, nodes, edges, questions_answers, episode_num, split=0.5):
        if questions_answers is None or episode_num < self.episodes * split:
            score = self.calculate_shaped_reward(graph) * self.shaped_reward_coef
        else:
            results = rag(graph, nodes, edges, questions_answers)
            score = sum([score for _, _, _, score in results]) / len(results)
    
        return score


    def modify_graph(self, graph, nodes, edges, node1_idx, node2_idx, edge_type_idx):        
        match (edge_type_idx):
            case 0:
                graph.edge_index = torch.cat((graph.edge_index, torch.tensor([[node1_idx], [node2_idx]], device=self.device)), dim=1)
                edges.append((node1_idx, node2_idx))
            case 1:
                graph.edge_index = torch.cat((graph.edge_index, torch.tensor([[node2_idx], [node1_idx]], device=self.device)), dim=1)
                edges.append((node2_idx, node1_idx))
            case 2:
                graph.edge_index = torch.cat((graph.edge_index, torch.tensor([[node1_idx, node2_idx], [node2_idx, node1_idx]], device=self.device)), dim=1)
                edges.append((node1_idx, node2_idx))
                edges.append((node2_idx, node1_idx))
        return graph, nodes, edges
    
    def generate_trajectory(self, graph, nodes, edges, questions_answers, episode_num, max_steps=MAX_STEPS):
        trajectory = []
        graph = graph.to(self.device)
        self.policy_net.eval()
        self.critic_net.eval()
        with torch.inference_mode():
            stop_idx = 0
            current_value = self.evaluate_graph_value(graph, nodes, edges, questions_answers, episode_num)
            for _ in range(max_steps):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(graph.x, graph.edge_index)            
                node1_idx = node1_soft.argmax().item()
                node2_idx = node2_soft.argmax().item()
                edge_type_idx = edge_type_soft.argmax().item()
                done = stop_soft.argmax().item()
                action_prob = (node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx]).item()                                
                # apply the action
                graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)
                value = self.evaluate_graph_value(graph, nodes, edges, questions_answers, episode_num)
                reward = value - current_value
                current_value = value
                trajectory.append(((node1_idx, node2_idx), edge_type_idx, action_prob, reward, done))
                if stop_idx == 1:
                    break
                                
        return trajectory, graph, nodes, edges

    def compute_advantages_and_returns(self, trajectory, graph, nodes, edges, gamma=0.99, lambda_=0.95):
        # Move to device
        graph = graph.to(self.device)
        advantages = []
        returns = []
        values = []

        # First pass: compute values for each state        
        for t in range(len(trajectory)):
            (node1_idx, node2_idx), edge_type_idx, action_prob, reward, done = trajectory[t]
            
            # Compute value for the current state
            current_value = self.critic_net(graph.x, graph.edge_index).squeeze()
            values.append(current_value.item())
            
            # Apply the action to advance the graph state
            graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)
        
        # Compute value for the final state
        final_value = self.critic_net(graph.x, graph.edge_index).squeeze().item()

        # Second pass: compute advantages and returns in reverse order
        gae = 0
        for t in reversed(range(len(trajectory))):
            reward = trajectory[t][3]
            done = trajectory[t][4]
            
            if t == len(trajectory) - 1:
                next_value = final_value
            else:
                next_value = values[t + 1]
            
            delta = reward + gamma * next_value * (1 - done) - values[t]
            gae = delta + lambda_ * gae * (1 - done)
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)

        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        return advantages, returns

    def run_episode(self, episode_num, graph, nodes, edges, questions_answers, num_trajectories=4):
        # Set networks to evaluation mode for the episode
        self.policy_net.eval()
        self.critic_net.eval()        
        trajectories = []        
        advantages_all = []
        returns_all = []
        with torch.inference_mode():
            # starting_fc_score = self.evaluate_value(graph, nodes, edges, questions_answers)
            for _ in range(num_trajectories):                        
                new_graph = Data(x=graph.x, edge_index=graph.edge_index.clone())                
                new_edges = edges.copy()                
                new_nodes = nodes.copy()
                trajectory, new_graph, new_nodes, new_edges = self.generate_trajectory(new_graph, new_nodes, new_edges, questions_answers, episode_num)                
                trajectories.append(trajectory)                
            
                # total_reward = sum([step[3] for step in trajectory])
                # print(f"Total reward: {total_reward:.3f}")
                # ending_fc_score = self.evaluate_value(new_graph, new_nodes, new_edges, questions_answers)
                # print(f"Starting FC score: {starting_fc_score:.3f}, Ending FC score: {ending_fc_score:.3f}")

            for trajectory in trajectories:
                new_graph = Data(x=graph.x, edge_index=graph.edge_index.clone())                
                new_edges = edges.copy()
                new_nodes = nodes.copy()
                advantages, returns = self.compute_advantages_and_returns(trajectory, new_graph, new_nodes, new_edges)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)                
                advantages = advantages.tolist()
                returns = returns.tolist()
                advantages_all.append(advantages)
                returns_all.append(returns)

        for trajectory, advantages, returns in zip(trajectories, advantages_all, returns_all):
            self.update_policy(trajectory, advantages, returns, graph, nodes, edges)
        
    def update_policy(self, trajectory, advantages, returns, graph, nodes, edges, epsilon=0.2, epochs=10):
        graph = graph.to(self.device)
        self.policy_net.train()
        self.critic_net.train()

        old_log_probs = torch.stack([torch.log(torch.tensor(action_prob, device=self.device)) for _, _, action_prob, _, _ in trajectory])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            new_log_probs = []
            new_values = []

            current_graph = Data(x=graph.x, edge_index=graph.edge_index.clone()).to(self.device)
            current_edges = edges.copy()

            for i, ((node1_idx, node2_idx), edge_type_idx, action_prob, _, _) in enumerate(trajectory):
                node1_soft, node2_soft, edge_type_soft, _ = self.policy_net(current_graph.x, current_graph.edge_index)
                log_action_prob = torch.log(node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx] + 1e-10)
                new_log_probs.append(log_action_prob)

                value_new = self.critic_net(current_graph.x, current_graph.edge_index)
                new_values.append(value_new)

                current_graph, current_edges, _ = self.modify_graph(current_graph, current_edges, nodes, node1_idx, node2_idx, edge_type_idx)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.value_loss_fn(new_values, returns)

            total_loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return graph, nodes, edges
    

    def infer_trajectory(self, graph, nodes, edges, max_steps=MAX_STEPS):
        self.policy_net.eval()        
        self.critic_net.eval()
        trajectory = []
        graph = graph.to(self.device)
        with torch.inference_mode():
            for _ in range(max_steps):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(graph.x, graph.edge_index)
                node1_idx = node1_soft.argmax().item()
                node2_idx = node2_soft.argmax().item()
                edge_type_idx = edge_type_soft.argmax().item()
                done = stop_soft.argmax().item()
                action_prob = (node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx]).item()
                
                # Apply the action
                graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)                
                
                trajectory.append(((node1_idx, node2_idx), edge_type_idx, action_prob, 0, done))
                
                if done == 1:
                    break

        return trajectory, graph, nodes, edges


