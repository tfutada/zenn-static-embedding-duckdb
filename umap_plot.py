import json
import pandas as pd
import plotly.express as px
import umap
from sentence_transformers import SentenceTransformer
from typing import List
from utils import fold_text
import consts

# Load documents from the JSONL file
with open(consts.LIVEDOOR_JSON, 'r', encoding='utf-8') as fd:
    docs = [json.loads(line) for line in fd]

# Extract texts and publishers
texts: List[str] = [doc.get("body", "") for doc in docs]
publishers: List[str] = [doc.get("publisher", "Unknown") for doc in docs]

# Load the embedding model
model_name = "hotchpotch/static-embedding-japanese"
model = SentenceTransformer(model_name, device="cpu")

# Generate embeddings
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

# Reduce dimensions using UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

# Prepare data for plotting
df_plot = pd.DataFrame({
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1],
    "publisher": publishers,
    "text": [fold_text(text) for text in texts]
})

# Create the scatter plot
fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="publisher",
    hover_data={"text": True},
    title="UMAP Projection of Livedoor Articles",
    labels={"x": "UMAP Dimension 1", "y": "UMAP Dimension 2"}
)

# Adjust marker size for better visibility
fig.update_traces(marker=dict(size=5))

# Display the plot
fig.show()

fig.write_image(consts.UMAP_PNG, width=1000, height=800)
