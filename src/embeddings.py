import requests
import json
from typing import List, Dict

from config import EMBEDDINGS_URL, EMBEDDINGS_MODEL

def get_nvidia_nim_embeddings(
    texts: List[str],
    model: str = EMBEDDINGS_MODEL,
    embeddings_url: str = EMBEDDINGS_URL
) -> List[Dict[str, any]]:
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
    
    for text in texts:
        payload = {
            "input": [text],
            "model": model,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END"
        }
        
        response = requests.post(embeddings_url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            item = result['data'][0]
            all_embeddings.append({
                "embedding": item['embedding'],
                "num_tokens": result['usage']['total_tokens']
            })
        else:
            raise Exception(f"Error getting embeddings: {response.status_code} - {response.text}")
    
    return all_embeddings