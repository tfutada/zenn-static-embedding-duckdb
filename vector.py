import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time
import consts
from hello import similarities

# Load the static embedding model
model_name = "hotchpotch/static-embedding-japanese"
model = SentenceTransformer(model_name, device="cpu", truncate_dim=1024)

# Load your JSONL file
df = pd.read_json(consts.LIVEDOOR_JSON, lines=True)
print(df.head())

# Encode all texts
texts = df["body"].tolist()

start = time.time()
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
end = time.time()
print(f"Embedding time: {end - start:.4f} seconds")

# Save embeddings as a NumPy array
np.save(consts.VECTORS_NPY, embeddings, allow_pickle=False)

similarities_all = model.similarity(embeddings[0], embeddings[1:])
print("Similarities for the first document:")
