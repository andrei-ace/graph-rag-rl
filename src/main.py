import os
import shutil
import warnings
import torch
from tqdm import tqdm
import joblib  # type: ignore
import argparse

from config import EMBEDDINGS_SIZE, POSITIONAL_EMBEDDINGS_DIM
from images import convert_pdf_to_images, vertically_append_images
from detect_layout import CLASS_NAMES, detect_layout_elements
from ocr import ocr_elements
from graphs import create_graph, update_coordinates_and_merge_graphs
from ppo import PPO
from visuals import visualize_graph
from questions import PDFS
from rag import rag

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

EPOCHS = 2
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

def infer_pdf(pdf_entry, ppo, device=device):
    (pdf_path, questions_answers) = pdf_entry
    cache_key = os.path.basename(pdf_path)  # or generate a unique key based on pdf_path
    merged_graph, merged_nodes, merged_edges, merged_image = cache_results(cache_key, process_pdf, pdf_path)
    merged_graph = merged_graph.to(device)

    save_path = "docs/output/no_trainig.png"
    if ppo is not None:
        # This will change the graph in place r
        trajectory, merged_graph, merged_nodes, merged_edges = ppo.infer_trajectory(
            merged_graph, merged_nodes, merged_edges
        )
        print(f"length of trajectory: {len(trajectory)}")
        save_path = "docs/output/with_trainig.png"
    visualize_graph(merged_image, merged_nodes, merged_edges, save_path=save_path)
    results = rag(merged_graph, merged_nodes, merged_edges, questions_answers)
    # for question, answer, generated_answer, score in results:
    #     print(f"Question: {question}\nProvided Answer:{answer}\nGenerated Answer: {generated_answer}\nScore: {score:.4f}")
    #     print("-" * 100)
    mean_score = sum([score for _, _, _, score in results]) / len(results)
    return mean_score


def train_pdf(pdf_entry, ppo, device=device):
    (pdf_path, questions_answers) = pdf_entry
    cache_key = os.path.basename(pdf_path)
    merged_graph, merged_nodes, merged_edges, _ = cache_results(cache_key, process_pdf, pdf_path)
    merged_graph = merged_graph.to(device)
    ppo.episode(merged_graph, merged_nodes, merged_edges, questions_answers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF with optional caching.")
    parser.add_argument("--disable-cache", action="store_true", help="Disable caching of results")
    args = parser.parse_args()
    if args.disable_cache:
        # delete the cache directory
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Device type: {device.type}")
    print(f"Device capabilities:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if device.type == 'cuda':
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    mean_score_notrain = infer_pdf(PDFS[-1], None)
    print(f"Mean scores no training: {mean_score_notrain:.4f}")
    input_dim = EMBEDDINGS_SIZE + len(CLASS_NAMES) + 4*POSITIONAL_EMBEDDINGS_DIM    
    ppo = PPO(input_dim=input_dim, device=device)
    pbar = tqdm(range(EPOCHS), desc="Training PPO")
    for _ in pbar:
        for pdf_entry in PDFS[:-1]:        
            train_pdf(pdf_entry, ppo)        
        mean_score_withtrain = infer_pdf(PDFS[-1], ppo)
        pbar.set_postfix({
            'No Train': f'{mean_score_notrain:.4f}',
            'With Train': f'{mean_score_withtrain:.4f}'
        })
        print("-" * 100)
        print(f"Mean scores no training: {mean_score_notrain:.4f} vs with trainig: {mean_score_withtrain:.4f}")
        print("-" * 100)