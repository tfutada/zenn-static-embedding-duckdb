import tracemalloc

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

tracemalloc.start()

# Load document texts
df = pd.read_json("livedoor.json", lines=True)
df["doc_index"] = df.index

# Load saved vectors (make sure they align with df rows)
vectors = np.load("vectors-livedoor-static.npy")

# Compute cosine similarity matrix
sim_matrix = cosine_similarity(vectors)


# Get top-k most similar document pairs (excluding self-pairs)
def get_top_similar_pairs(sim_matrix, top_k=5):
    pairs = []
    num_docs = sim_matrix.shape[0]

    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            pairs.append(((i, j), sim_matrix[i, j]))

    return sorted(pairs, key=lambda x: x[1], reverse=True)[:top_k]

print(tracemalloc.get_traced_memory())  # (current, peak)

top_pairs = get_top_similar_pairs(sim_matrix, top_k=50)

# Display results
for (i, j), score in top_pairs:
    print(f"\n🔗 Similarity: {score:.4f}")
    print(f"[{i}] ({df.loc[i, 'publisher']}): {df.loc[i, 'body'][:200]}...")
    print(f"[{j}] ({df.loc[j, 'publisher']}): {df.loc[j, 'body'][:200]}...")

print(tracemalloc.get_traced_memory())  # (current, peak)

tracemalloc.stop()
