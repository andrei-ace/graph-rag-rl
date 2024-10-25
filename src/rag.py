from torch.nn.functional import cosine_similarity
from graphs import extract_text_from_graph, split_graph
from embeddings import get_nvidia_nim_embeddings
import requests
import json
import torch
import math

from config import RANK_MODEL, RANK_URL, TOP_K

def rank_answers(question, retrieved_text_list, 
                 model: str = RANK_MODEL,
                 url: str = RANK_URL) -> tuple[float, str]:
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": model,
        "query": {"text": question},
        "passages": [{"text": text} for text in retrieved_text_list],
        "truncate": "END"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        results = response.json()
        # Extract logits for each retrieved text
        logits = [result['logit'] for result in results['rankings']]
        
        # Apply softmax to normalize logits
        exp_logits = [math.exp(logit) for logit in logits]
        total = sum(exp_logits)
        probabilities = [exp_logit / total for exp_logit in exp_logits]
        
        # Find the best score and its corresponding text
        best_score = max(probabilities)
        best_index = probabilities.index(best_score)
        best_relevant_text = retrieved_text_list[best_index]
        
        return best_score, best_relevant_text  # Return best score and text
    else:
        print(f"Error: {response.status_code}")
        return 0, ""  # Return zero score and empty text in case of error


def evaluate_answer(question, retrieved_text, correct_answer, 
                    model: str = RANK_MODEL,
                    url: str = RANK_URL) -> float:
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": model,
        "query": {"text": question},
        "passages": [
            {"text": retrieved_text},
            {"text": correct_answer}
        ],
        "truncate": "END"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        results = response.json()
        retrieved_logit = results['rankings'][0]['logit']
        correct_logit = results['rankings'][1]['logit']
        
        # Apply softmax to normalize logits
        exp_retrieved = math.exp(retrieved_logit)
        exp_correct = math.exp(correct_logit)
        total = exp_retrieved + exp_correct
        
        retrieved_prob = exp_retrieved / total
        correct_prob = exp_correct / total
        
        # Calculate similarity based on normalized probabilities
        similarity = 1 - abs(retrieved_prob - correct_prob)
        
        return similarity  # Already in [0, 1] range
    else:
        print(f"Error: {response.status_code}")
        return 0  # Return 0 similarity in case of error


def retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=1) -> list[str]:
    # Convert question_embedding to a tensor if it's not already
    question_embedding = torch.tensor(question_embedding).unsqueeze(0)
    
    # Ensure text_embeddings is also a tensor
    text_embeddings = torch.tensor(text_embeddings)
    
    similarities = cosine_similarity(question_embedding, text_embeddings, dim=1)
    # Sort indices of similarities in descending order
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    
    return [texts[idx] for idx in top_k_indices]


def rag(graph, nodes, edges, questions_answers) -> list[tuple[str, str, str, float]]:
    subgraphs = split_graph(graph, nodes, edges)
    # print(f"Found {len(subgraphs)} subgraphs")
    texts = []
    for subgraph, nodes, edges in subgraphs:
        txt = extract_text_from_graph(subgraph, nodes, edges)
        texts.append(txt)

    text_embeddings = [embedding['embedding'] for embedding in get_nvidia_nim_embeddings(texts)]
    questions, answers = zip(*questions_answers)
    question_embeddings = [embedding['embedding'] for embedding in get_nvidia_nim_embeddings(questions)]

    results = []
    for question, provided_answer, question_embedding in zip(questions, answers, question_embeddings):
        relevant_text_list = retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=TOP_K)
        # Get the best score and relevant text
        best_score, best_relevant_text = rank_answers(question, relevant_text_list)
        
        score = evaluate_answer(question, best_relevant_text, provided_answer)
        results.append((question, provided_answer, best_relevant_text, score))
        

    return results
