from torch.nn.functional import cosine_similarity
from graphs import extract_text_from_graph, split_graph
from embeddings import get_nvidia_nim_embeddings
import requests
import json

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
        # Assuming the API returns scores in the same order as the input passages
        retrieved_score = results['scores'][0]
        correct_score = results['scores'][1]
        # Calculate similarity as the ratio of retrieved_score to correct_score
        similarity = retrieved_score / correct_score if correct_score != 0 else 0
        return similarity
    else:
        print(f"Error: {response.status_code}")
        return 0  # Return 0 similarity in case of error


def retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=1):
    # Compute cosine similarities
    similarities = cosine_similarity(question_embedding.unsqueeze(0), text_embeddings)
    # Sort indices of similarities in descending order
    top_k_indices = similarities.argsort(descending=True)[:top_k]
    return " ".join([texts[idx] for idx in top_k_indices])


def rag(graph, nodes, edges, questions_answers):
    subgraphs = split_graph(graph, nodes, edges)
    texts = []
    for subgraph, nodes, edges in subgraphs:
        txt = extract_text_from_graph(subgraph, nodes, edges)
        texts.append(txt)

    text_embeddings = get_nvidia_nim_embeddings(texts)
    questions, answers = zip(*questions_answers)
    question_embeddings = get_nvidia_nim_embeddings(questions)

    results = []
    for question, provided_answer, question_embedding in zip(questions, answers, question_embeddings):
        relevant_text = retrieve_relevant_text(question_embedding, text_embeddings, texts, top_k=1)
        score = evaluate_answer(question, relevant_text, provided_answer)
        results.append((question, provided_answer, relevant_text, score))

    return results