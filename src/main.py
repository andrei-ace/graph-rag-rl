import math
import os
import shutil
import warnings
import torch
from tqdm import tqdm
import joblib  # type: ignore
import argparse
from torch.utils.tensorboard import SummaryWriter

from config import EMBEDDINGS_SIZE, EPOCHS, HIDDEN_DIM, START_TEMP, END_TEMP, SPLIT, DECAY_RATE, TOP_K, POSITIONAL_EMBEDDINGS_DIM
from images import convert_pdf_to_images, vertically_append_images
from detect_layout import CLASS_NAMES, detect_layout_elements
from ocr import ocr_elements
from graphs import create_graph, find_strongly_connected_components, update_coordinates_and_merge_graphs
from ppo import PPO, PPOConfig
from visuals import visualize_graph
from questions import PDFS
from rag import rag

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Attempting to run cuBLAS")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# Define a cache directory
CACHE_DIR = "__cache__"

# Ensure the cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)
if os.path.exists("docs/output"):
    shutil.rmtree("docs/output")
os.makedirs("docs/output", exist_ok=True)

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

def cache_results(cache_key, func, *args, **kwargs):
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if os.path.exists(cache_file):
        return joblib.load(cache_file)
    else:
        results = func(*args, **kwargs)
        joblib.dump(results, cache_file)
        return results

# Function to process the PDF and return the results
def process_pdf(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    layout_items = [detect_layout_elements(image) for image in images]
    ocr_results = [ocr_elements(image, items) for image, items in zip(images, layout_items)]
    graphs_nodes_edges = [create_graph(elements) for elements in ocr_results]
    merged_graph, merged_nodes, merged_edges = update_coordinates_and_merge_graphs(graphs_nodes_edges, images)
    merged_image = vertically_append_images(images)
    return merged_graph, merged_nodes, merged_edges, merged_image

def infer_pdf(pdf_entry, ppo, device=device) -> tuple[float, int, int]:
    scc_count = 0
    path_length = 0
    (pdf_path, questions_answers) = pdf_entry
    cache_key = os.path.basename(pdf_path)  # or generate a unique key based on pdf_path
    merged_graph, merged_nodes, merged_edges, merged_image = cache_results(cache_key, process_pdf, pdf_path)
    merged_graph = merged_graph.to(device)

    save_path = "docs/output/no_trainig.png"
    if ppo is not None:
        # This will change the graph in place
        trajectory, merged_graph, merged_nodes, merged_edges = ppo.infer_trajectory(
            merged_graph, merged_nodes, merged_edges
        )
        strongly_connected_components = find_strongly_connected_components(merged_graph.edge_index, merged_graph.num_nodes)
        scc_count = len(strongly_connected_components)
        path_length = len(trajectory)
        print(f"length of trajectory: {len(trajectory)} num_components: {len(strongly_connected_components)}")
        save_path = "docs/output/with_trainig.png"
    # visualize_graph(merged_image, merged_nodes, merged_edges, save_path=save_path)
    results = rag(merged_graph, merged_nodes, merged_edges, questions_answers)
    if ppo is None:
        for question, answer, generated_answer, score in results:
            print(f"Question: {question}\nProvided Answer:{answer}\nGenerated Answer: {generated_answer}\nScore: {score:.4f}")
            print("-" * 100)
    mean_score = sum([score for _, _, _, score in results]) / len(results)
    return mean_score, scc_count, path_length


def train_pdf(pdf_entry, ppo, device=device):
    (pdf_path, questions_answers) = pdf_entry
    cache_key = os.path.basename(pdf_path)
    merged_graph, merged_nodes, merged_edges, _ = cache_results(cache_key, process_pdf, pdf_path)
    merged_graph = merged_graph.to(device)
    ppo.run_episode(merged_graph, merged_nodes, merged_edges, questions_answers)


def determine_temperature(episode_num):
    # Adjusted phase lengths
    phase1_length = EPOCHS * SPLIT / 2
    phase2_length = EPOCHS * SPLIT / 2
    phase3_length = EPOCHS * (1 - SPLIT) / 2
    phase4_length = EPOCHS * (1 - SPLIT) / 2

    # Define the intermediate stop point as 1/4 of the distance between start_temp and end_temp
    intermediate_temp = END_TEMP + (START_TEMP - END_TEMP) * (3 / 4)

    if episode_num < phase1_length:
        # Phase 1: Explore with shaped reward, stopping at 2/3 point
        return START_TEMP + (intermediate_temp - START_TEMP) * (episode_num / phase1_length)
    elif episode_num < phase1_length + phase2_length:
        # Phase 2: Exploit with shaped reward (exponential decay starting from intermediate point)
        adjusted_episode_num = episode_num - phase1_length
        return intermediate_temp * math.exp(-DECAY_RATE * adjusted_episode_num)
    elif episode_num < phase1_length + phase2_length + phase3_length:
        # Phase 3: Explore with real reward, stopping at 2/3 point
        adjusted_episode_num = episode_num - (phase1_length + phase2_length)
        return START_TEMP + (intermediate_temp - START_TEMP) * (adjusted_episode_num / phase3_length)
    else:
        # Phase 4: Exploit with real reward (exponential decay starting from intermediate point)
        adjusted_episode_num = episode_num - (phase1_length + phase2_length + phase3_length)
        return intermediate_temp * math.exp(-DECAY_RATE * adjusted_episode_num)


if __name__ == "__main__":
    PDFS = PDFS[:-1]
    parser = argparse.ArgumentParser(description="Process PDF with optional caching.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable caching of results")
    parser.add_argument("--continue-from-last-checkpoint", action="store_true", help="Continue training from the last checkpoint")
    args = parser.parse_args()
    
    if args.disable_cache:
        # delete the cache directory
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Device type: {device.type}")
    print(f"Device capabilities:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")    

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="logs")

    # Load last checkpoint if --continue-from-checkpoint is set
    if args.continue_from_last_checkpoint:
        checkpoint_dir = "models/checkpoints"
        ppo = PPO.load_model_checkpoint(checkpoint_dir, device=device)
        start_episode = ppo.episode_num + 1
        print(f"Continuing from episode {start_episode-1}")
    else:
        # Initialize PPO
        input_dim = EMBEDDINGS_SIZE + len(CLASS_NAMES) + 1 + 4 * POSITIONAL_EMBEDDINGS_DIM    
        ppo = PPO.for_training(PPOConfig(input_dim=input_dim, hidden_dim=HIDDEN_DIM), device=device)
        shutil.rmtree("models/checkpoints", ignore_errors=True)
        start_episode = 0

    mean_score_notrain = sum([score for score, _, _ in [infer_pdf(pdf, None) for pdf in PDFS]]) / len(PDFS)
    # print(f"Mean scores no training: {mean_score_notrain:.7f}")

    pbar = tqdm(total=EPOCHS, desc="Training PPO")
    
    for episode_num in range(start_episode, EPOCHS):
        pbar.n = episode_num        
        pbar.refresh()        
        for pdf_entry in PDFS:        
            train_pdf(pdf_entry, ppo)
        temperature = determine_temperature(episode_num)
        next_temperature = determine_temperature(episode_num+1)
        ppo.step(episode_num, next_temperature)
        
        inferred = [infer_pdf(pdf, ppo) for pdf in PDFS]
        mean_score_withtrain = sum(score for score, _, _ in inferred) / len(PDFS)
        mean_scc_count = sum(scc_count for _, scc_count, _ in inferred) / len(PDFS)
        mean_path_length = sum(path_length for _, _, path_length in inferred) / len(PDFS)

        # Log metrics to TensorBoard
        writer.add_scalar('LR', ppo.scheduler.get_last_lr()[0], episode_num)
        writer.add_scalar('Temperature', temperature, episode_num)
        writer.add_scalar('Mean Score No Train', mean_score_notrain, episode_num)
        writer.add_scalar('Mean Score With Train', mean_score_withtrain, episode_num)
        writer.add_scalar('Improvement', mean_score_withtrain - mean_score_notrain, episode_num)
        writer.add_scalar('Mean SCC Count', mean_scc_count, episode_num)
        writer.add_scalar('Mean Path Length', mean_path_length, episode_num)

        pbar.set_postfix({
            'Temperature': f'{temperature:.7f}',
            'LR': f'{ppo.scheduler.get_last_lr()[0]:.7f}',
            'No Train': f'{mean_score_notrain:.7f}',            
            'Improvement': f'{mean_score_withtrain - mean_score_notrain:.7f}'            
        })
        # save the model
        ppo.save_model_checkpoint(f"models/checkpoints")
    
    ppo.save_model(f"models/final")
    # Close the TensorBoard writer
    writer.close()
