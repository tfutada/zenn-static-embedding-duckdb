import duckdb

# Connect to in-memory DuckDB
con = duckdb.connect()

# Install and load the VSS extension
con.execute("INSTALL vss;")
con.execute("LOAD vss;")

# Create the table and insert data
con.execute("CREATE TABLE embeddings (id INTEGER, vec FLOAT[3]);")
vectors = [
    (1, [0.1, 0.2, 0.3]),
    (2, [0.4, 0.5, 0.6]),
    (3, [0.7, 0.8, 0.9])
]
con.executemany("INSERT INTO embeddings VALUES (?, ?);", vectors)

# Create the HNSW index
con.execute("CREATE INDEX idx ON embeddings USING HNSW (vec) WITH (metric = 'cosine');")

# Perform the nearest neighbor search
query_vector = [0.2, 0.3, 0.4]
result = con.execute("""
    SELECT id, vec, array_cosine_distance(vec, ?::FLOAT[3]) AS distance
    FROM embeddings
    ORDER BY array_cosine_distance(vec, ?::FLOAT[3])
    LIMIT 2;
""", [query_vector, query_vector]).fetchall()

# Output the results
for row in result:
    print(f"ID: {row[0]}, Vector: {row[1]}, Distance: {row[2]:.4f}")
