import os

EMBEDDINGS_URL = os.environ.get('EMBEDDINGS_URL', 'http://localhost:8001/v1/embeddings')
EMBEDDINGS_SIZE = os.environ.get('EMBEDDINGS_SIZE', 1024)
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL', 'nvidia/nv-embedqa-e5-v5')

RANK_MODEL = os.environ.get('RANK_MODEL', 'nvidia/nv-rerankqa-mistral-4b-v3')
RANK_URL = os.environ.get('RANK_URL', 'http://localhost:8002/v1/ranking')

POSITIONAL_EMBEDDINGS_DIM = 64