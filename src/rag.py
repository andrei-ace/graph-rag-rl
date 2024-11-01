from torch.nn.functional import cosine_similarity, sigmoid
from graphs import extract_text_from_graph, split_graph
from embeddings import get_nvidia_nim_embeddings
import requests
import json
import torch
import math

from config import RANK_MODEL, RANK_URL, TOP_K

def rank_answers(question, correct_answer, retrieved_text_list, 
                 model: str = RANK_MODEL,
                 url: str = RANK_URL) -> tuple[float, str]:
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    payload = {
        "model": model,
        "query": {"text": question},
        "passages": [{"text": correct_answer}] + [{"text": text} for text in retrieved_text_list],
        "truncate": "END"
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        results = response.json()
        rankings = results['rankings']
        # order by index as the list index is not guaranteed to be in order
        rankings.sort(key=lambda x: x['index'])
        # Extract logits for each retrieved text
        logits = [result['logit'] for result in rankings]
        
        # scale logits so the correct answer logit
        logits_tensor = torch.tensor(logits)
        logits_tensor = logits_tensor / logits_tensor[0] * 4

        # convert to probabilities
        probabilities = sigmoid(logits_tensor).tolist()
        
        # Find the best score and its corresponding text, ignore the correct answer (index 0)
        probabilities = probabilities[1:]
        best_score = max(probabilities)        
        best_index = probabilities.index(best_score)
        best_relevant_text = retrieved_text_list[best_index]        
        
        return best_score, best_relevant_text  # Return best score and text
    else:
        print(f"Error: {response.status_code}")
        return 0, ""  # Return zero score and empty text in case of error


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
        best_score, best_relevant_text = rank_answers(question, provided_answer, relevant_text_list)
        results.append((question, provided_answer, best_relevant_text, best_score))
        

    return results
