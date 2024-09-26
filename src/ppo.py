from math import sqrt
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from config import EMBEDDINGS_TOKEN_LIMIT
from models import CriticNetwork, PolicyNetwork
from rag import rag
from graphs import find_strongly_connected_components
from concurrent.futures import ThreadPoolExecutor
import os

MIN_STEPS = 4
MAX_STEPS = 256

class PPO:
    def __init__(self, input_dim, 
                 device="cpu", 
                 episodes=100,
                 shaped_reward_coef=0.1, 
                 split=0.5, 
                 start_temp=5.0,
                 end_temp=0.1, 
                 decay_rate=0.1, 
                 num_trajectories=8,
                 epochs=4):
        hidden_dim = 128
        # Initialize networks
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.critic_net = CriticNetwork(input_dim, hidden_dim)
        self.device = torch.device(device)        
        self.policy_net.to(self.device)
        self.critic_net.to(self.device)
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters())
            + list(self.critic_net.parameters()),
            lr=1e-4,
        )
        # Initialize loss functions for PPO
        self.value_loss_fn = torch.nn.MSELoss()        
        self.episodes = episodes
        self.shaped_reward_coef = shaped_reward_coef
        self.split = split
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.decay_rate = decay_rate
        self.num_trajectories = num_trajectories
        self.max_workers = min(num_trajectories, os.cpu_count() or 1)
        self.epochs = epochs

    def calculate_shaped_reward(self, graph, nodes, edges):
        num_nodes = graph.num_nodes
        sccs = find_strongly_connected_components(graph.edge_index, num_nodes)
        scc_len = len(sccs)
        target_scc = num_nodes // 2
        max_tokens_per_scc = EMBEDDINGS_TOKEN_LIMIT

        def normalize_reward(reward):
            return 2.0 * (reward - 0.5)  # Normalize to [-1, 1]

        def calculate_num_scc_reward():
            reward = 1.0 - abs(scc_len - target_scc) / target_scc
            return normalize_reward(reward)

        def calculate_size_variance_reward():
            scc_sizes = torch.tensor([len(scc) for scc in sccs], dtype=torch.float32, device=self.device)
            size_variance = torch.var(scc_sizes).item() if len(scc_sizes) > 1 else 0
            max_possible_variance = ((num_nodes - scc_len + 1) ** 2) * (scc_len - 1) / (4 * scc_len)
            reward = 1.0 - (size_variance / max_possible_variance if max_possible_variance > 0 else 0)
            return normalize_reward(reward)

        def calculate_large_scc_penalty():
            avg_scc_size = num_nodes / scc_len
            penalty = sum([len(scc) for scc in sccs if len(scc) > 2 * avg_scc_size]) / num_nodes
            return normalize_reward(penalty)

        def calculate_degree_balance_reward():
            degrees = torch.sum(graph.edge_index, dim=1).float()
            degree_variance = torch.var(degrees).item()
            max_possible_degree_variance = (num_nodes - 1) * (num_nodes - 1) / 4
            reward = 1.0 - (degree_variance / max_possible_degree_variance)
            return normalize_reward(reward)

        # def calculate_token_distribution_reward():
        #     penalty = 0.0
        #     scc_token_counts = [sum(nodes[node]["num_tokens"] for node in scc) for scc in sccs]
        #     for total_tokens in scc_token_counts:
        #         if total_tokens > max_tokens_per_scc:
        #             penalty += (total_tokens - max_tokens_per_scc) / max_tokens_per_scc
            
        #     # Calculate variance of token counts
        #     token_variance = torch.var(torch.tensor(scc_token_counts, dtype=torch.float32, device=self.device)).item()
        #     max_possible_variance = (max_tokens_per_scc ** 2) * (len(sccs) - 1) / len(sccs)
        #     reward = 1.0 - (token_variance / max_possible_variance if max_possible_variance > 0 else 0)
            
        #     net_reward = reward - penalty
        #     return normalize_reward(net_reward)

        num_scc_reward = calculate_num_scc_reward()
        size_variance_reward = calculate_size_variance_reward()
        degree_balance_reward = calculate_degree_balance_reward()
        # token_limit_penalty = calculate_token_distribution_reward()

        reward = (
            0.33 * num_scc_reward +
            0.33 * size_variance_reward +
            0.33 * degree_balance_reward
            # 0.50 * token_limit_penalty +
        )
        return reward

    def evaluate_graph_value(self, graph, nodes, edges, done, 
                             questions_answers=None, 
                             initial_value=0.0, 
                             episode_num=0):
        score = 0
        if episode_num < self.episodes * self.split:
            score = self.calculate_shaped_reward(graph, nodes, edges) * self.shaped_reward_coef
            # add the score of the RAG queries to the score
            if done and questions_answers is not None:
                results = rag(graph, nodes, edges, questions_answers)
                score += sum([score for _, _, _, score in results]) / len(results) - initial_value
        else:
            # Only evaluate the score if the episode is done
            if done and questions_answers is not None:
                results = rag(graph, nodes, edges, questions_answers)
                score = sum([score for _, _, _, score in results]) / len(results)
            else:
                score = 0
    
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
    
    def generate_trajectory(self, graph, nodes, edges, questions_answers, episode_num):
        trajectory = []
        graph = graph.to(self.device)
        self.policy_net.eval()
        self.critic_net.eval()
        with torch.inference_mode():
            starting_value = self.evaluate_graph_value(graph, nodes, edges, False, 
                                                       questions_answers=questions_answers, 
                                                       initial_value=0.0, 
                                                       episode_num=episode_num)
            current_value = starting_value
            for i in range(MAX_STEPS):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(graph.x, graph.edge_index)
                # if node1 or node 2 are None then we stop
                if node1_soft is None or node2_soft is None or edge_type_soft is None or stop_soft is None:
                    break
                node1_idx = node1_soft.argmax().item()
                node2_idx = node2_soft.argmax().item()
                edge_type_idx = edge_type_soft.argmax().item()
                done = stop_soft.argmax().item()
                last_step = i == MAX_STEPS - 1 or done == 1
                action_prob = (node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx] * stop_soft[0][0]).item()                                
                # apply the action
                graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)
                value = self.evaluate_graph_value(graph, nodes, edges, last_step, 
                                                  questions_answers=questions_answers, 
                                                  initial_value=starting_value, 
                                                  episode_num=episode_num)
                reward = value - current_value
                current_value = value
                trajectory.append(((node1_idx, node2_idx), edge_type_idx, action_prob, reward, done))
                # ignore the stop if steps is less than min_steps
                if last_step:
                    break
                                
        return trajectory, graph, nodes, edges

    def compute_advantages_and_returns(self, trajectory, graph, nodes, edges, gamma=0.995, lambda_=0.95):
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

    def run_episode(self, episode_num, graph, nodes, edges, questions_answers, temperature):
        self.policy_net.set_temperature(temperature)
        
        # Set networks to evaluation mode for the episode
        self.policy_net.eval()
        self.critic_net.eval()        
        
        def process_trajectory(_):            
            with torch.inference_mode():
                while True:  # Keep generating trajectories until we get one with at least min_steps
                    new_graph = Data(x=graph.x, edge_index=graph.edge_index.clone())                
                    new_edges = edges.copy()                
                    new_nodes = nodes.copy()
                    trajectory, new_graph, new_nodes, new_edges = self.generate_trajectory(new_graph, new_nodes, new_edges, questions_answers, episode_num)                
                    
                    if len(trajectory) >= MIN_STEPS:
                        break  # Exit the loop if the trajectory has at least min_steps
                
                new_graph = Data(x=graph.x, edge_index=graph.edge_index.clone())                
                new_edges = edges.copy()
                new_nodes = nodes.copy()
                
                advantages, returns = self.compute_advantages_and_returns(trajectory, new_graph, new_nodes, new_edges)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)                
                advantages = advantages.tolist()
                
                return trajectory, advantages, returns.tolist()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_trajectory, range(self.num_trajectories)))

        for trajectory, advantages, returns in results:
            self.update_policy(trajectory, advantages, returns, graph, nodes, edges)

    def update_policy(self, trajectory, advantages, returns, graph, nodes, edges, epsilon=0.2):
        graph = graph.to(self.device)
        self.policy_net.train()
        self.critic_net.train()

        old_log_probs = torch.stack([torch.log(torch.tensor(action_prob, device=self.device)) for _, _, action_prob, _, _ in trajectory])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            new_log_probs = []
            new_values = []            

            current_graph = Data(x=graph.x, edge_index=graph.edge_index.clone()).to(self.device)
            current_nodes = nodes.copy()
            current_edges = edges.copy()

            for i, ((node1_idx, node2_idx), edge_type_idx, _, _, done) in enumerate(trajectory):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(current_graph.x, current_graph.edge_index)
                if node1_soft is None or node2_soft is None or edge_type_soft is None or stop_soft is None:
                    break
                log_action_prob = torch.log(node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * 
                                            edge_type_soft[0][edge_type_idx] * stop_soft[0][0] + 1e-10)
                new_log_probs.append(log_action_prob)

                value_new = self.critic_net(current_graph.x, current_graph.edge_index)
                new_values.append(value_new)

                current_graph, current_nodes, current_edges = self.modify_graph(
                    current_graph, current_nodes, current_edges, node1_idx, node2_idx, edge_type_idx)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()

            # Combine action and done probabilities
            combined_log_probs = new_log_probs
            ratio = torch.exp(combined_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.value_loss_fn(new_values, returns)

            total_loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return graph, nodes, edges
    

    def infer_trajectory(self, graph, nodes, edges):
        # Set temperature to 0.1 to make the inference more deterministic
        self.policy_net.set_temperature(0.1)
        self.policy_net.eval()        
        self.critic_net.eval()
        trajectory = []
        graph = graph.to(self.device)
        with torch.inference_mode():
            for _ in range(MAX_STEPS):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(graph.x, graph.edge_index)
                if node1_soft is None or node2_soft is None or edge_type_soft is None or stop_soft is None:
                    break
                node1_idx = node1_soft.argmax().item()
                node2_idx = node2_soft.argmax().item()
                edge_type_idx = edge_type_soft.argmax().item()
                done = stop_soft.argmax().item()            
                
                # Apply the action
                graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)                
                
                # we don't need the action probability or reward, so we set them to 0
                trajectory.append(((node1_idx, node2_idx), edge_type_idx, 0, 0, done))
                
                if done == 1:
                    break

        return trajectory, graph, nodes, edges


