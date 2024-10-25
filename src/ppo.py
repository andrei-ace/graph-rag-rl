import random
import torch
from torch_geometric.data import Data
from config import EMBEDDINGS_TOKEN_LIMIT
from models import CriticNetwork, PolicyNetwork
from rag import rag
from graphs import find_strongly_connected_components
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np

MIN_STEPS = 2
MAX_STEPS = 256
TRAIN_EPOCHS = 4
HIDDEN_DIM = 128
RAG_SKIP_STEPS = 8
NUM_TRAJECTORIES = 16
SAMPLE_SIZE = 8

class PPO:
    def __init__(self,
                 input_dim,
                 episodes=100,
                 hidden_dim=HIDDEN_DIM,
                 split=0.5,
                 start_temp=2.0,
                 end_temp=0.1,
                 decay_rate=0.01,
                 num_trajectories=NUM_TRAJECTORIES,
                 epochs=TRAIN_EPOCHS,
                 device="cpu"):
        # Initialize networks
        self.policy_net = PolicyNetwork(input_dim, hidden_dim)
        self.critic_net = CriticNetwork(input_dim, hidden_dim)        
        self.device = torch.device(device)        
        self.policy_net.to(self.device)
        self.critic_net.to(self.device)
        
        # Initialize weights and biases of the critic network's linear layers to 0
        # for module in self.critic_net.modules():
        #     if isinstance(module, torch.nn.Linear):
        #         torch.nn.init.zeros_(module.weight)
        #         torch.nn.init.zeros_(module.bias)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters())
            + list(self.critic_net.parameters()),
            lr=1e-5,
        )
        # Initialize loss functions for PPO
        self.value_loss_fn = torch.nn.MSELoss()        
        self.episodes = episodes
        self.split = split
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.decay_rate = decay_rate
        self.num_trajectories = num_trajectories
        self.max_workers = min(num_trajectories, os.cpu_count() or 1)
        self.epochs = epochs

    def max_steps_for_episode(self, episode_num):
        if episode_num < self.episodes * self.split:
            # Calculate the fraction of the split that has been completed
            fraction_completed = episode_num / (self.episodes * self.split)
            # Calculate the max steps based on the fraction completed
            return int((0.1 + random.uniform(0, 0.4) + 0.5 * fraction_completed) * MAX_STEPS)
        else:
            return MAX_STEPS

    def calculate_shaped_reward(self, graph, nodes, edges):
        num_nodes = graph.num_nodes
        strongly_connected_components = find_strongly_connected_components(graph.edge_index, num_nodes)
        num_components = len(strongly_connected_components)
        target_num_components = num_nodes // 2
        target_num_edges = MAX_STEPS // 2
        max_tokens_per_component = EMBEDDINGS_TOKEN_LIMIT

        # def normalize_distribution(values):
        #     values = np.array(values)
        #     return (values - np.mean(values)) / (np.std(values) + 1e-8)

        # def kl_divergence_to_normal(values):
        #     if len(values) < 2:
        #         return 0  # Return 0 similarity for single-value distributions

        #     normalized_values = normalize_distribution(values)
            
        #     # Check if all values are the same (resulting in zero standard deviation)
        #     if np.all(normalized_values == normalized_values[0]):
        #         return 0  # Return 0 similarity for uniform distributions
            
        #     # Fit a normal distribution to the data
        #     mu, std = stats.norm.fit(normalized_values)
            
        #     # Calculate KL divergence
        #     hist, bin_edges = np.histogram(normalized_values, bins='auto', density=True)
        #     bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
            
        #     pdf_observed = hist
        #     pdf_normal = stats.norm.pdf(bin_midpoints, mu, std)
            
        #     # Add small epsilon to avoid division by zero
        #     epsilon = 1e-10
        #     pdf_observed = pdf_observed + epsilon
        #     pdf_normal = pdf_normal + epsilon
            
        #     kl_div = np.sum(pdf_observed * np.log(pdf_observed / pdf_normal))
            
        #     # Convert KL divergence to a similarity score (lower is better, so we invert)
        #     similarity = 1 / (1 + kl_div)
        #     return similarity

        def evaluate_num_edges():
            return 1.0 - abs(len(edges) - target_num_edges) / target_num_edges

        def evaluate_component_count():
            return 1.0 - abs(num_components - target_num_components) / target_num_components

        # def evaluate_size_distribution():
        #     component_sizes = [len(component) for component in strongly_connected_components]
        #     return kl_divergence_to_normal(component_sizes)

        # def evaluate_node_degree_balance():
        #     node_degrees = torch.sum(graph.edge_index, dim=1).float().cpu().numpy()
        #     return kl_divergence_to_normal(node_degrees)            

        def evaluate_token_distribution():
            # Calculate the number of tokens in each strongly connected component
            component_token_counts = [sum(nodes[node]["num_tokens"] for node in component) 
                                      for component in strongly_connected_components]
            
            # Calculate a penalty for components exceeding the max token limit
            return -np.mean(
                np.clip(
                    np.array(component_token_counts) - max_tokens_per_component, 
                    0, 
                    None
                ) / max_tokens_per_component
            )

        overall_score = (
            evaluate_component_count() * 0.4 \
            + evaluate_num_edges() * 0.1 \
            # + evaluate_size_distribution() \
            # + evaluate_node_degree_balance() \
            + evaluate_token_distribution() * 0.5
        )
        return overall_score * 0.1


    def calculate_rag_score(self, graph, nodes, edges, questions_answers):
        results = rag(graph, nodes, edges, questions_answers)
        return sum([score for _, _, _, score in results]) / len(results)        


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
        improvement = None
        graph = Data(x=graph.x, edge_index=graph.edge_index.clone())
        graph = graph.to(self.device)
        edges = edges.copy()
        nodes = nodes.copy()
        self.policy_net.eval()
        self.critic_net.eval()
        with torch.inference_mode():
                        
            starting_value = self.calculate_shaped_reward(graph, nodes, edges) # if episode_num < self.episodes * self.split else 0.0
            current_value = starting_value
            starting_value_rag = self.calculate_rag_score(graph, nodes, edges, questions_answers)
            current_rag_score = starting_value_rag
            rag_score = None

            max_steps = self.max_steps_for_episode(episode_num)
            for i in range(max_steps):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(graph.x, graph.edge_index)
                # if node1 or node 2 are None then we stop
                if node1_soft is None or node2_soft is None or edge_type_soft is None or stop_soft is None:
                    # get last step of trajectory and set done to 1
                    node1_idx = trajectory[-1][0][0]
                    node2_idx = trajectory[-1][0][1]
                    edge_type_idx = trajectory[-1][1]
                    action_prob = (1.0 - trajectory[-1][2])
                    last_step = True
                    done = 1
                    # remove last step from trajectory
                    # the modification on the graph was already applied in the last step
                    trajectory = trajectory[:-1]
                else:
                    node1_idx = node1_soft.argmax().item()
                    node2_idx = node2_soft.argmax().item()
                    edge_type_idx = edge_type_soft.argmax().item() # 0 node1 -> node2, 1 node2 -> node1, 2 node1 <-> node2
                    done = stop_soft.argmax().item() # 0 if we are not done, 1 if we are done
                    last_step = (i == max_steps - 1) or (done == 1)
                    if last_step:
                        action_prob = (node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx] * stop_soft[0][1]).item()
                        done = 1
                    else:
                        action_prob = (node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * edge_type_soft[0][edge_type_idx] * stop_soft[0][0]).item()
                        done = 0
                
                    # apply the action
                    graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)                
                
                value = self.calculate_shaped_reward(graph, nodes, edges) # if episode_num < self.episodes * self.split else 0.0
                if i>0 and (episode_num >= self.episodes * self.split) and (last_step or i%RAG_SKIP_STEPS==0):
                    rag_score = self.calculate_rag_score(graph, nodes, edges, questions_answers)
                    value += rag_score - current_rag_score
                    current_rag_score = rag_score
                                
                reward = value - current_value
                current_value = value
                trajectory.append(((node1_idx, node2_idx), edge_type_idx, action_prob, reward, done))
                if i > 0 and last_step:
                    if rag_score is None:
                        rag_score = self.calculate_rag_score(graph, nodes, edges, questions_answers)
                    # if abs(rag_score - starting_value_rag) > 1e-7:
                    improvement = rag_score - starting_value_rag
                    total_reward = sum(reward for _, _, _, reward, _ in trajectory)
                    
                    num_nodes = graph.num_nodes
                    strongly_connected_components = find_strongly_connected_components(graph.edge_index, num_nodes)
                    num_components = len(strongly_connected_components)
                    print(f"Length: {i+1:3d}, SCC: {num_components:3d} of {num_nodes:3d}, " +
                        f"Starting RAG score: {starting_value_rag:.7f}, Current RAG score: {rag_score:.7f}, " +
                        f"Improvement RAG: {improvement:+.7f}, Total reward: {total_reward:+.7f}")
                    # print(f"Length: {i+1:3d}, SCC: {num_components:3d} of {num_nodes:3d}, " +
                    #     f"Starting score: {starting_value:.7f}, Current score: {current_value:.7f}, " +
                    #     f"Total reward: {total_reward:+.7f}")
                    break
                                
        return improvement, trajectory

    # this must be run inside torch.inference_mode()
    def compute_advantages_and_returns(self, trajectory, graph, nodes, edges, gamma=0.995, lambda_=0.95):
        graph = Data(x=graph.x, edge_index=graph.edge_index.clone())
        graph = graph.to(self.device)
        edges = edges.copy()
        nodes = nodes.copy()
        advantages = []
        returns = []
        values = []

        # First pass: compute values for each state
        for t in range(len(trajectory)):
            (node1_idx, node2_idx), edge_type_idx, _, _, _ = trajectory[t]
            
            # Compute value for the current state
            current_value = self.critic_net(graph.x, graph.edge_index).squeeze().item()
            values.append(current_value)
            
            # Apply the action to advance the graph state
            graph, nodes, edges = self.modify_graph(graph, nodes, edges, node1_idx, node2_idx, edge_type_idx)
        
        # Compute value for the final state
        final_value = self.critic_net(graph.x, graph.edge_index).squeeze().item()
        # Second pass: compute advantages and returns in reverse order
        gae = 0
        for t in reversed(range(len(trajectory))):
            reward = trajectory[t][3]
            done = 1 if t == len(trajectory) - 1 else 0
            
            next_value = final_value if t == len(trajectory) - 1 else values[t + 1]
            
            delta = reward + gamma * next_value * (1 - done) - values[t]
            gae = delta + gamma * lambda_ * gae * (1 - done)
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)        

        advantages = np.array(advantages)
        returns = np.array(returns)
        return advantages, returns

    def run_episode(self, episode_num, graph, nodes, edges, questions_answers, temperature):
        self.policy_net.set_temperature(temperature)
        
        # Set networks to evaluation mode for the episode
        self.policy_net.eval()
        self.critic_net.eval()        
        
        def process_trajectory(_):            
            with torch.inference_mode():
                while True:  # Keep generating trajectories until we get one with at least min_steps                    
                    improvement, trajectory = self.generate_trajectory(
                        graph, nodes, edges, questions_answers, episode_num)                
                    
                    if len(trajectory) >= MIN_STEPS:
                        break  # Exit the loop if the trajectory has at least min_steps                
                
                advantages, returns = self.compute_advantages_and_returns(trajectory, graph, nodes, edges)                
                
                return improvement, trajectory, advantages, returns

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_trajectory, range(self.num_trajectories)))

        # sample SAMPLE_SIZE trajectories with the highest improvement and lowest improvement
        results.sort(key=lambda x: x[0])
        results = results[:SAMPLE_SIZE] + results[-SAMPLE_SIZE:]        

        results = normalize_advantages(results)
        improvement_sum = 0.0
        for improvement, trajectory, advantages, returns in results:                
            improvement_sum += improvement            
                
        print(f"Starting training for {len(results)} trajectories, average RAG improvement: {improvement_sum / len(results):.7f}...")        
        for _ in range(TRAIN_EPOCHS):
            random.shuffle(results)
            for improvement, trajectory, advantages, returns in results:                
                self.update_policy(trajectory, advantages, returns, graph, nodes, edges)
        print(f"Training complete")

    def update_policy(self, trajectory, advantages, returns, graph, nodes, edges, epsilon=0.2):
        # print(f"Advantages: {advantages}")
        self.policy_net.train()
        self.critic_net.train()

        # Calculate old_log_probs once, outside the loop
        old_log_probs = torch.stack([torch.log(torch.tensor(action_prob, device=self.device)) for _, _, action_prob, _, _ in trajectory])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            new_log_probs = []
            new_values = []            

            current_graph = Data(x=graph.x, edge_index=graph.edge_index.clone()).to(self.device)
            current_graph = current_graph.to(self.device)
            current_nodes = nodes.copy()
            current_edges = edges.copy()

            for i, ((node1_idx, node2_idx), edge_type_idx, _, _, done) in enumerate(trajectory):
                node1_soft, node2_soft, edge_type_soft, stop_soft = self.policy_net(current_graph.x, current_graph.edge_index)
                if node1_soft is None or node2_soft is None or edge_type_soft is None or stop_soft is None:
                    break
                    
                if done == 1:
                    log_action_prob = torch.log(node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * 
                                            edge_type_soft[0][edge_type_idx] * stop_soft[0][1] + 1e-10)
                else:
                    log_action_prob = torch.log(node1_soft[0][node1_idx] * node2_soft[0][node2_idx] * 
                                            edge_type_soft[0][edge_type_idx] * stop_soft[0][0] + 1e-10)
                new_log_probs.append(log_action_prob)

                value_new = self.critic_net(current_graph.x, current_graph.edge_index)
                new_values.append(value_new)

                current_graph, current_nodes, current_edges = self.modify_graph(
                    current_graph, current_nodes, current_edges, node1_idx, node2_idx, edge_type_idx)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()

            # Create a temporary truncated version of old_log_probs
            truncated_old_log_probs = old_log_probs[:len(new_log_probs)]
            
            ratio = torch.exp(new_log_probs - truncated_old_log_probs)
            
            surr1 = ratio * advantages[:len(new_log_probs)]
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages[:len(new_log_probs)]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = self.value_loss_fn(new_values, returns[:len(new_log_probs)])

            total_loss = policy_loss + 0.1 * value_loss
            
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
            # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_norm=0.5)
            self.optimizer.step()

            # print(f"Epoch {epoch+1:2d}/{self.epochs:2d}, Policy loss: {policy_loss.item():.9f}, Value loss: {value_loss.item():.9f}, Total loss: {total_loss.item():.9f}")

        return graph, nodes, edges
    

    def infer_trajectory(self, graph, nodes, edges):
        graph = Data(x=graph.x, edge_index=graph.edge_index.clone())
        graph = graph.to(self.device)
        edges = edges.copy()
        nodes = nodes.copy()
        # Set temperature to 0.1 to make the inference more deterministic
        self.policy_net.set_temperature(0.1)
        self.policy_net.eval()        
        self.critic_net.eval()
        trajectory = []
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


def normalize_advantages(results):
    all_advantages = np.concatenate([adv for _, _, adv, _ in results])
    std = np.std(all_advantages)
    
    normalized_results = []
    for improvement, trajectory, advantages, returns in results:
        # Normalize by dividing by std, preserving sign and keeping 0.0 at 0.0
        normalized_advantages = advantages / (std + 1e-8)
        normalized_results.append((improvement, trajectory, normalized_advantages, returns))
    
    return normalized_results


