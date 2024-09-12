import requests
import json
from typing import List

from config import EMBEDDINGS_URL, EMBEDDINGS_MODEL

def get_nvidia_nim_embeddings(
    texts: List[str],
    model: str = EMBEDDINGS_MODEL,
    embeddings_url: str = EMBEDDINGS_URL,
    batch_size: int = 16
) -> List[List[float]]:
    if not embeddings_url:
        raise ValueError("embeddings_url must be provided")
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        payload = {
            "input": batch,
            "model": model,
            "input_type": "query"
        }
        
        response = requests.post(embeddings_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            batch_embeddings = [item['embedding'] for item in result['data']]
            all_embeddings.extend(batch_embeddings)
        else:
            raise Exception(f"Error getting embeddings: {response.status_code} - {response.text}")
    
    return all_embeddings