import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity

# Load dataset and create embeddings (global for testing)
df = pd.read_csv("movies.csv")

# Load the Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert the 'plot of the movies into an embedding
plot_embeddings = model.encode(df["plot"].tolist(), convert_to_tensor=True)



def search_movies(query, top_n=5):
    
    query_emb = model.encode(query, convert_to_tensor=True)

    scores = cosine_similarity(query_emb.unsqueeze(0), plot_embeddings)

    top_indices = scores.topk(top_n).indices.tolist()

    results = df.iloc[top_indices].copy()

    results['similarity'] = [scores[i].item() for i in top_indices]
    
    return results