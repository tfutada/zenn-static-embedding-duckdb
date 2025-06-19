import os

# ライブドアコーパスをここにおく
# 例) CORPUS_DIR/it-life-hack/it-life-hack-6292880.txt
CORPUS_DIR = os.getenv("CORPUS_DIR", "/Users/tafu/live-door/corpus")
LIVEDOOR_JSON = os.getenv("LIVEDOOR_JSON", "livedoor.json")
UMAP_PNG = os.getenv("UMAP_PNG", "umap_livedoor_plot.png")
VECTORS_NPY = os.getenv("VECTORS_NPY", "vectors-livedoor-static.npy")
