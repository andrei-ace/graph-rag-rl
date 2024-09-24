from math import sqrt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models import CriticNetwork, PolicyNetwork
from rag import rag
from graphs import find_strongly_connected_components
import concurrent.futures
import os

MAX_STEPS = 64

class PPO:
    def __init__(self, input_dim, shaped_reward_coef=0.1, device="cpu"):
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
        self.policy_loss_fn = torch.nn.CrossEntropyLoss()        

    def shaped_reward_value(self, graph, nodes, edges, questions_answers):
        # Reward for how close the number of strongly connected components (SCC) is to the square root of the number of nodes
        # The reward is 1 if the number of SCC is equal to the square root of the number of nodes, and decreases as the number of SCC increases or decreases
        scc_len = len(find_strongly_connected_components(graph.edge_index, len(nodes)))
        target_scc = sqrt(len(nodes))
        reward = 1.0 - abs(scc_len - target_scc) / target_scc
        return reward

    def evaluate_value(self, graph, nodes, edges, questions_answers):
        # results = rag(graph, nodes, edges, questions_answers)
        # score = sum([score for _, _, _, score in results]) / len(results)
        score = self.shaped_reward_value(graph, nodes, edges, questions_answers) * self.shaped_reward_coef
        return score


    def apply_action(self, graph, edges, nodes, node1_idx, node2_idx, edge_type_idx):        
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
        return graph, edges, nodes
    
    def collect_trajectory(self, graph, edges, nodes, questions_answers, max_steps=MAX_STEPS):
        trajectory = []
        graph = graph.to(self.device)
        self.policy_net.eval()
        self.critic_net.eval()
        with torch.inference_mode():
            stop_idx = 0
            current_value = self.evaluate_value(graph, nodes, edges, questions_answers)
            for _ in range(max_steps):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(graph.x, graph.edge_index)            
                node1_idx = node1_soft.argmax().item()
                node2_idx = node2_soft.argmax().item()
                edge_type_idx = edge_type_soft.argmax().item()
                done = stop_soft.argmax().item()
                action_prob = (node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx]).item()                                
                # apply the action
                graph, edges, nodes = self.apply_action(graph, edges, nodes, node1_idx, node2_idx, edge_type_idx)
                value = self.evaluate_value(graph, nodes, edges, questions_answers)
                reward = value - current_value
                current_value = value
                trajectory.append(((node1_idx, node2_idx), edge_type_idx, action_prob, reward, done))
                if stop_idx == 1:
                    break
                                
        return trajectory, graph, edges, nodes


    def episode(self, graph, nodes, edges, questions_answers, num_trajectories=1, max_steps=MAX_STEPS):
        # Set networks to evaluation mode for the episode
        self.policy_net.eval()
        self.critic_net.eval()        
        trajectories = []        
        with torch.inference_mode():
            starting_fc_score = self.evaluate_value(graph, nodes, edges, questions_answers)
            for _ in range(num_trajectories):                        
                new_graph = Data(x=graph.x, edge_index=graph.edge_index.clone())            
                new_edges = edges.copy()
                new_nodes = nodes.copy()
                
                trajectory, new_graph, new_edges, new_nodes = self.collect_trajectory(new_graph, new_edges, new_nodes, questions_answers)                
                trajectories.append(trajectory)
                print(trajectory)
                total_reward = sum([step[3] for step in trajectory])
                print(f"Total reward: {total_reward:.3f}")
                ending_fc_score = self.evaluate_value(new_graph, new_nodes, new_edges, questions_answers)
                print(f"Starting FC score: {starting_fc_score:.3f}, Ending FC score: {ending_fc_score:.3f}")
            
        #     def collect_single_trajectory():
        #         new_graph = Data(x=graph.x, edge_index=graph.edge_index.clone())
        #         new_nodes = nodes.copy()
        #         new_edges = edges.copy()
        #         trajectory, new_graph, new_nodes, new_edges = self.collect_trajectory(
        #             new_graph, new_nodes, new_edges
        #         )
        #         # Compute real values and rewards
        #         real_values, rewards, new_graph, new_nodes, new_edges = self.compute_real_values(
        #             trajectory, new_graph, new_nodes, new_edges, questions_answers
        #         )
        #         # Compute advantages and returns
        #         advantages, returns = self.compute_advantages(trajectory)
        #         return trajectory, advantages, returns, real_values

        #     # Determine the optimal number of threads
        #     max_workers = min(num_trajectories, os.cpu_count() or 1)
            
        #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #         futures = [executor.submit(collect_single_trajectory) for _ in range(num_trajectories)]
        #         trajectories = [future.result() for future in concurrent.futures.as_completed(futures)]


        # for trajectory, advantages, returns, real_values in trajectories:            
        #     # Reject trajectories with less steps than min_steps
        #     if len(trajectory) >= MIN_STEPS:
        #         # PPO update
        #         graph, nodes, edges = self.ppo_update(trajectory, advantages, returns, graph, nodes, edges)
        #     else:
        #         print(f"Trajectory rejected: {len(trajectory)} steps < {MIN_STEPS} min steps")


    # def revert_graph(self, graph, nodes, edges, trajectory):
    #     # trajectory: (action_name, node_pair, action_prob, pair_prob, value, total_log_prob, node1_emb, node2_emb, reward)
    #     # Apply the trajectory in reverse to revert the graph to its initial state
    #     for action_name, node_pair, _, _, _, _, _, _, _ in reversed(trajectory):
    #         graph, edges = revert_action(graph, edges, action_name, node_pair)
    #     return graph, nodes, edges
    
    def compute_policy_loss(self, edge_prob, advantage):
        target = (
            torch.tensor(1.0, dtype=torch.float32, device=self.device)
            if advantage > 0
            else torch.tensor(0.0, device=self.device)
        )
        bce_loss = F.binary_cross_entropy(edge_prob, target)
        return bce_loss * advantage.abs()


    def ppo_update(self, trajectory, advantages, returns, graph, nodes, edges, epsilon=0.2, epochs=10):
        # Move graph to the same device as the model
        graph = graph.to(self.device)
        # Set networks to training mode for the update phase
        self.policy_net.train()
        self.critic_net.train()        

        # Extract old log probabilities from trajectory
        old_log_probs = torch.stack([total_log_prob for _, _, _, _, _, total_log_prob, _, _, _ in trajectory])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            new_log_probs = []
            new_values = []
            action_losses = []

            # Use a fresh copy of the initial graph for each epoch
            current_graph = Data(x=graph.x, edge_index=graph.edge_index.clone()).to(self.device)
            current_edges = edges.copy()

            # Recompute states and action probabilities by reusing the actions from the trajectory
            for i, (action_name, node_pair, action_prob, pair_prob, value, _, node1_emb, node2_emb, _) in enumerate(trajectory):
                # Compute action probabilities from the policy network
                action_probs = self.policy_net(current_graph.x, current_graph.edge_index).squeeze(0)
                action_index = self.action_mapping[action_name]
                action_prob_new = action_probs[action_index]
                log_action_prob = torch.log(action_prob_new + 1e-10)

                total_log_prob = log_action_prob

                if action_name != "stop" and node_pair is not None:
                    # Recompute edge probability from the appropriate action network
                    if action_name == "add":
                        edge_prob = self.add_net(node1_emb.unsqueeze(0), node2_emb.unsqueeze(0)).squeeze()
                    elif action_name == "add_both":
                        edge_prob = self.add_both_net(node1_emb.unsqueeze(0), node2_emb.unsqueeze(0)).squeeze()
                    elif action_name == "remove":
                        edge_prob = self.remove_net(node1_emb.unsqueeze(0), node2_emb.unsqueeze(0)).squeeze()
                    else:
                        raise ValueError(f"Unknown action: {action_name}")

                    log_edge_prob = torch.log(edge_prob + 1e-10)
                    total_log_prob += log_edge_prob

                    # Compute action network loss using BCE
                    action_loss = self.compute_policy_loss(edge_prob, advantages[i])
                    action_losses.append(action_loss)
                else:
                    # No action network loss for 'stop' action
                    edge_prob = torch.tensor(1.0, device=self.device)
                    action_loss = torch.tensor(0.0, device=self.device)
                    action_losses.append(action_loss)

                new_log_probs.append(total_log_prob)

                # Compute value estimate
                value_new = self.critic_net(current_graph.x, current_graph.edge_index)
                new_values.append(value_new)

                # Apply the action to the current graph
                current_graph, current_edges = apply_action(current_graph, current_edges, action_name, node_pair)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()
            action_losses = torch.stack(action_losses)

            # Compute the ratio (pi_theta / pi_theta_old)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = self.value_loss_fn(new_values.squeeze(), returns)

            # Total loss
            # total_loss = policy_loss # + 0.5 * value_loss + action_losses.mean()
            total_loss = value_loss

            # Update networks
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return graph, nodes, edges




    def infer_trajectory(self, graph, nodes, edges, max_steps=MAX_STEPS):
        self.policy_net.eval()        
        self.critic_net.eval()
        trajectory = []        

        # with torch.no_grad():
        #     trajectory = []
        #     step_count = 0
        #     attempt_count = 0
        #     max_attempts = max_steps * 3  # Prevent infinite loop

        #     while step_count < max_steps:
        #         if attempt_count >= max_attempts:
        #             print("Warning: Max attempts reached. Terminating trajectory collection.")
        #             break

        #         # Move graph data to the correct device
        #         graph = graph.to(self.device)

        #         # Get action probabilities from the policy network
        #         action_probs = self.policy_net(graph.x, graph.edge_index).squeeze(0)  # Shape: [4]

        #         # Call sample_action function to get the action
        #         action_name, node_pair, action_prob, value, pair_prob = self.sample_action_inference(
        #             graph,
        #             self.add_net,
        #             self.add_both_net,
        #             self.remove_net,
        #             self.critic_net,
        #             action_probs
        #         )

        #         if action_name != "stop" and node_pair is not None:
        #             # Apply the action to the graph
        #             graph, edges = apply_action(graph, edges, action_name, node_pair)
        #             trajectory.append((action_name, node_pair))
        #             step_count += 1
        #             attempt_count += 1
        #         else:
        #             if len(trajectory) >= min_steps:
        #                 trajectory.append((action_name, node_pair))
        #                 break
        #             else:
        #                 attempt_count += 1
        #                 continue

        return trajectory, graph, nodes, edges


