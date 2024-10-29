import random
import torch
from torch_geometric.data import Data
from config import EMBEDDINGS_TOKEN_LIMIT, HIDDEN_DIM, SPLIT, START_TEMP, END_TEMP, DECAY_RATE, EPOCHS
from models import CriticNetwork, PolicyNetwork
from rag import rag
from graphs import find_strongly_connected_components
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
from transformers import PretrainedConfig
import shutil

MIN_STEPS = 2
MAX_STEPS = 256
REPEAT_TRAJECTORIES = 8
TRAJECTORY_TRAIN_EPOCHS = 8
RAG_SKIP_STEPS = 32
NUM_TRAJECTORIES = 16
SAMPLE_SIZE = 8

class PPOConfig(PretrainedConfig):
    def __init__(self, input_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim    
        self.gamma = 0.995
        self.lambda_ = 0.95

class PPO:
    def __init__(self,
                 config: PPOConfig,                 
                 device="cpu"):        
        self.config = config
        self.device = torch.device(device)

        # Initialize networks
        self.policy_net = PolicyNetwork(config.input_dim, config.hidden_dim)
        self.critic_net = CriticNetwork(config.input_dim, config.hidden_dim)        
        self.policy_net.to(self.device)
        self.critic_net.to(self.device)                

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters())
            + list(self.critic_net.parameters()),
            lr=1e-6,
        )
        # Initialize loss functions for PPO
        self.value_loss_fn = torch.nn.MSELoss()        
        self.gamma = config.gamma
        self.lambda_ = config.lambda_
        
        self.episodes = EPOCHS
        self.episode_num = 0
        self.split = SPLIT
        self.start_temp = START_TEMP
        self.end_temp = END_TEMP
        self.decay_rate = DECAY_RATE
        self.num_trajectories = NUM_TRAJECTORIES
        self.max_workers = min(self.num_trajectories, os.cpu_count() or 1)        

    @classmethod
    def from_pretrained(cls, path, config: PPOConfig, device="cpu"):
        """Initialize PPO with pre-trained weights."""
        instance = cls(config, device)
        instance.policy_net = PolicyNetwork.from_pretrained(path)
        instance.critic_net = CriticNetwork.from_pretrained(path)
        instance.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location=device))
        print(f"Model loaded from {path}")
        return instance
    
    @classmethod
    def for_training(cls, config: PPOConfig, device="cpu"):
        """Initialize PPO for training with new weights."""
        return cls(config, device)
    
    def save_model(self, path):
        """Save the model parameters and configuration to the specified path using Hugging Face."""
        os.makedirs(path, exist_ok=True)
        
        # Save the policy network state_dict
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy_net.bin'))
        
        # Save the critic network state_dict
        torch.save(self.critic_net.state_dict(), os.path.join(path, 'critic_net.bin'))
        
        # Save the optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        
        # Save the configuration
        self.config.save_pretrained(path)
        
        print(f"Model and configuration saved to {path}")

    @classmethod
    def load_model(cls, path, device="cpu"):
        # Load the configuration
        config = PPOConfig.from_pretrained(path)
        
        # Create an instance of the class
        instance = cls(config, device)
        
        # Load the policy network state_dict
        instance.policy_net.load_state_dict(torch.load(os.path.join(path, 'policy_net.bin'), map_location=instance.device))
        instance.policy_net.to(instance.device)  # Move to device
        
        # Load the critic network state_dict
        instance.critic_net.load_state_dict(torch.load(os.path.join(path, 'critic_net.bin'), map_location=instance.device))
        instance.critic_net.to(instance.device)  # Move to device
        
        # Load the optimizer state
        instance.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location=instance.device))
        
        print(f"Model and configuration loaded from {path}")
        return instance

    def save_model_checkpoint(self, path_to_checkpoint_dir, additional_params=None, keep_last_k=3):        
                
        cleanup_old_checkpoints(path_to_checkpoint_dir, keep_last_k)
        
        path = os.path.join(path_to_checkpoint_dir, f'checkpoint_{self.episode_num}')
        """Save the model parameters, configuration, and additional parameters to the specified path."""
        os.makedirs(path, exist_ok=True)
        
        # Save the policy network state_dict
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'policy_net.bin'))
        
        # Save the critic network state_dict
        torch.save(self.critic_net.state_dict(), os.path.join(path, 'critic_net.bin'))
        
        # Save the optimizer state
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
        
        # Save the configuration
        self.config.save_pretrained(path)
        
        # Save additional parameters
        checkpoint = {            
            'episodes': self.episodes,
            'episode_num': self.episode_num,
            'split': self.split,
            'temperature': self.start_temp,
            'start_temp': self.start_temp,
            'end_temp': self.end_temp,
            'decay_rate': self.decay_rate,
            # Add any other parameters you want to save
        }
        if additional_params:
            checkpoint.update(additional_params)
        
        torch.save(checkpoint, os.path.join(path, 'additional_params.pt'))        

    @classmethod
    def load_model_checkpoint(cls, path, device="cpu"):
        if not os.path.exists(path):
            print(f"Checkpoint directory {path} does not exist")
            return
        # find the latest checkpoint
        checkpoint_files = [
            f for f in os.listdir(path)
            if f.startswith('checkpoint_') and f.split('_')[-1].isdigit() and os.path.isdir(os.path.join(path, f))
        ]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        path = os.path.join(path, latest_checkpoint)
        print(f"Loading checkpoint from {path}")
        """Load the model parameters, configuration, and additional parameters from the specified path."""
        # Load the configuration
        config = PPOConfig.from_pretrained(path)
        
        # Initialize networks with the loaded configuration
        instance = cls(config, device=device)
        instance.policy_net = PolicyNetwork(config.input_dim, config.hidden_dim)
        instance.critic_net = CriticNetwork(config.input_dim, config.hidden_dim)
        
        # Load the policy network state_dict
        instance.policy_net.load_state_dict(torch.load(os.path.join(path, 'policy_net.bin'), map_location=device))
        instance.policy_net.to(device)  # Move to device
        
        # Load the critic network state_dict
        instance.critic_net.load_state_dict(torch.load(os.path.join(path, 'critic_net.bin'), map_location=device))
        instance.critic_net.to(device)  # Move to device
        
        # Load the optimizer state
        instance.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location=device))
        
        # Load additional parameters
        checkpoint = torch.load(os.path.join(path, 'additional_params.pt'), map_location=device)
        instance.start_temp = checkpoint.get('temperature', instance.start_temp)
        instance.episodes = checkpoint.get('episodes', instance.episodes)
        instance.episode_num = checkpoint.get('episode_num', instance.episode_num)
        instance.split = checkpoint.get('split', instance.split)
        instance.start_temp = checkpoint.get('start_temp', instance.start_temp)
        instance.end_temp = checkpoint.get('end_temp', instance.end_temp)
        instance.decay_rate = checkpoint.get('decay_rate', instance.decay_rate)        
        return instance

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
        def evaluate_num_edges():
            return 1.0 - abs(len(edges) - target_num_edges) / target_num_edges

        def evaluate_component_count():
            return 1.0 - abs(num_components - target_num_components) / target_num_components        

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
            evaluate_component_count() * 0.1 +
            evaluate_num_edges() * 0.1 +              
            evaluate_token_distribution() * 0.8
        )
        return overall_score


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
    

    def generate_trajectory(self, graph, nodes, edges, questions_answers, episode_num, starting_value_rag=None):
        trajectory = []
        improvement = None
        graph = Data(x=graph.x, edge_index=graph.edge_index.clone())
        graph = graph.to(self.device)
        edges = edges.copy()
        nodes = nodes.copy()
        self.policy_net.eval()
        self.critic_net.eval()
        with torch.inference_mode():
                        
            starting_value = self.calculate_shaped_reward(graph, nodes, edges) if episode_num < self.episodes * self.split else 0.0
            current_value = starting_value
            if starting_value_rag is None:
                starting_value_rag = self.calculate_rag_score(graph, nodes, edges, questions_answers)
            current_rag_score = starting_value_rag
            rag_score = None
            value = None

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
                
                if value is None:
                    value = starting_value
                else:
                    value = self.calculate_shaped_reward(graph, nodes, edges) if episode_num < self.episodes * self.split else 0.0
                if i>0 and (last_step or i%RAG_SKIP_STEPS==0): # and (episode_num >= self.episodes * self.split):
                    if rag_score is None:
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
    def compute_advantages_and_returns(self, trajectory, graph, nodes, edges):
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
            
            delta = reward + self.gamma * next_value * (1 - done) - values[t]
            gae = delta + self.gamma * self.lambda_ * gae * (1 - done)
            
            returns.insert(0, gae + values[t])
            advantages.insert(0, gae)        

        advantages = np.array(advantages)
        returns = np.array(returns)
        return advantages, returns

    def run_episode(self, episode_num, graph, nodes, edges, questions_answers, temperature):
        self.episode_num = episode_num
        self.policy_net.set_temperature(temperature)
        
        # Set networks to evaluation mode for the episode
        self.policy_net.eval()
        self.critic_net.eval()        

        with torch.inference_mode():
            starting_value_rag = self.calculate_rag_score(graph, nodes, edges, questions_answers)
        
        def process_trajectory(_):                        
            with torch.inference_mode():    
                while True:  # Keep generating trajectories until we get one with at least min_steps                    
                    improvement, trajectory = self.generate_trajectory(
                        graph, nodes, edges, questions_answers, episode_num, starting_value_rag)                
                    
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
        for _ in range(REPEAT_TRAJECTORIES):
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
                
        for epoch in range(TRAJECTORY_TRAIN_EPOCHS):
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
            self.optimizer.step()
            
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


def cleanup_old_checkpoints(path_to_checkpoint_dir, keep_last_k):
    if not os.path.exists(path_to_checkpoint_dir):
        return    
    # List all files in the directory that match the checkpoint pattern
    checkpoint_files = [
        f for f in os.listdir(path_to_checkpoint_dir)
        if f.startswith('checkpoint_') and f.split('_')[-1].isdigit() and os.path.isdir(os.path.join(path_to_checkpoint_dir, f))
    ]    
    # Sort the files based on the numeric suffix
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1]))        
    # Remove old checkpoints if there are more than keep_last_k
    while len(checkpoint_files) >= keep_last_k:
        shutil.rmtree(os.path.join(path_to_checkpoint_dir, checkpoint_files.pop(0)))