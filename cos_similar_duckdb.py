import textwrap
import time

import numpy as np
import pandas as pd
import polars as pl
import duckdb

import consts

# Load precomputed vectors and metadata
vectors = np.load(consts.VECTORS_NPY)  # Shape: (N, D)
assert vectors.shape[1] == 1024, "Expected 1024-dimensional vectors"

df = pd.read_json(consts.LIVEDOOR_JSON, lines=True)
df["doc_index"] = df.index

# Convert to Polars DataFrame
pl_df = pl.DataFrame({
    "doc_index": df["doc_index"],
    "feature": vectors.tolist()
})

# Connect to DuckDB
con = duckdb.connect("livedoor_vectors.duckdb")

# Install and load the VSS extension
con.execute("INSTALL vss;")
con.execute("LOAD vss;")

# Optional: Enable experimental persistence for HNSW index
con.execute("SET hnsw_enable_experimental_persistence = true;")

# Register the Polars DataFrame as a DuckDB table
con.register("pl_df", pl_df)

# Create or replace the 'docs' table with the vector data, casting to FLOAT[1024]
# populate livedoor documents with their vectors into DuckDB from Polars DataFrame
con.execute("""
    CREATE OR REPLACE TABLE docs AS
    SELECT
        doc_index,
        feature::FLOAT[1024] AS feature
    FROM pl_df
""")

# Create an HNSW index on the 'feature' column using cosine distance
con.execute("""
            CREATE INDEX my_docs_hnsw
                ON docs
                USING HNSW (feature)
                WITH (metric = 'cosine');
            """)

# Query to find top similar pairs using cosine similarity
print("Executing query to find top similar pairs...")

TARGET_DOC_INDEX = 1234  # Example: using the first document as the target
print(f"{df.loc[TARGET_DOC_INDEX, 'publisher']}): {df.loc[TARGET_DOC_INDEX, 'body'][:200]}...")
query_vector = vectors[TARGET_DOC_INDEX].astype(np.float32).tolist()  # ndarray to list conversion

start = time.time()
top_k_docs = con.execute(
    "SELECT doc_index, array_cosine_similarity(?::FLOAT[1024], feature) AS sim FROM docs ORDER BY sim DESC LIMIT 5",
    [query_vector]
).fetchall()
end = time.time()

print(f"Embedding time: {end - start:.4f} seconds")
for i, score in top_k_docs:
    print(f"\n🔗 Similarity: {score:.4f}")
    print(f"[{i}] ({df.loc[i, 'publisher']}): {df.loc[i, 'body'][:200]}...")

#
# Alternative query using HSNW index
# EXPLAIN sql shows that HNSW index is used

sql = textwrap.dedent(f"""
    SELECT
        doc_index,
        array_cosine_distance(feature, ?::FLOAT[1024]) AS score
    FROM   docs
    ORDER  BY score
    LIMIT  5;
""")

start = time.time()
top_k_docs_2 = con.execute(sql, [query_vector]).fetchall()
end = time.time()
print(f"Embedding time: {end - start:.4f} seconds")

for i, score in top_k_docs_2:
    print(f"\n🔗 Similarity: {1 - score:.4f}")
    print(f"[{i}] ({df.loc[i, 'publisher']}): {df.loc[i, 'body'][:200]}...")

con.close()
