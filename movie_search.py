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
    # Encode the user's query into an embedding tensor
    query_emb = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query embedding and all plot embeddings
    scores = cosine_similarity(query_emb.unsqueeze(0), plot_embeddings)

    # Get indices of the top_n most similar movies
    top_indices = scores.topk(top_n).indices.tolist()

    # Select the corresponding rows from the dataframe
    results = df.iloc[top_indices].copy()

    # Add a 'similarity' column with the similarity scores for the top movies
    results['similarity'] = [scores[i].item() for i in top_indices]
    
    # Return the resulting dataframe with top matches and their similarity scores