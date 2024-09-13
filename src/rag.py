from torch.nn.functional import cosine_similarity
from graphs import extract_text_from_graph, split_graph
from embeddings import get_nvidia_nim_embeddings
import requests
import json
import torch
import math

from config import RANK_MODEL, RANK_URL

def evaluate_answer(question, retrieved_text, correct_answer, 
                    model: str = RANK_MODEL,
                    url: str = RANK_URL):
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


def retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=1):
    # Convert question_embedding to a tensor if it's not already
    question_embedding = torch.tensor(question_embedding).unsqueeze(0)
    
    # Ensure text_embeddings is also a tensor
    text_embeddings = torch.tensor(text_embeddings)
    
    similarities = cosine_similarity(question_embedding, text_embeddings, dim=1)
    # Sort indices of similarities in descending order
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    
    return [texts[idx] for idx in top_k_indices]


def rag(graph, nodes, edges, questions_answers):
    subgraphs = split_graph(graph, nodes, edges)
    print(f"Found {len(subgraphs)} subgraphs")
    texts = []
    for subgraph, nodes, edges in subgraphs:
        txt = extract_text_from_graph(subgraph, nodes, edges)
        texts.append(txt)

    text_embeddings = get_nvidia_nim_embeddings(texts)
    questions, answers = zip(*questions_answers)
    question_embeddings = get_nvidia_nim_embeddings(questions)

    results = []
    for question, provided_answer, question_embedding in zip(questions, answers, question_embeddings):
        relevant_text_list = retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=10)
        scores = []
        for relevant_text in relevant_text_list:
            score = evaluate_answer(question, relevant_text, provided_answer)
            scores.append(score)
        # Choose the best score
        best_score = max(scores)
        best_index = scores.index(best_score)
        best_relevant_text = relevant_text_list[best_index]
        results.append((question, provided_answer, best_relevant_text, best_score))

    return results