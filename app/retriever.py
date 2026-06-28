import json
import logging
import faiss
from sentence_transformers import SentenceTransformer

# =====================================================
# Configuration
# =====================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = "vector_store/medical_knowledge.index"

METADATA_PATH = "vector_store/medical_metadata.json"

TOP_K = 20

# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =====================================================
# Load Embedding Model
# =====================================================

logger.info("Loading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

logger.info("Embedding model loaded.")

# =====================================================
# Load FAISS Index
# =====================================================

logger.info("Loading FAISS index...")

index = faiss.read_index(INDEX_PATH)

logger.info(f"Total vectors loaded : {index.ntotal}")

# =====================================================
# Load Metadata
# =====================================================

logger.info("Loading metadata...")

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

logger.info(f"Metadata records : {len(metadata)}")


# =====================================================
# Search Function
# =====================================================

def search(query: str, top_k: int = TOP_K):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        doc = metadata[idx].copy()

        doc["similarity"] = float(score)

        results.append(doc)

    return results


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    while True:

        query = input("\nAsk Question : ")

        if query.lower() == "exit":
            break

        docs = search(query)

        print("\nRetrieved Documents\n")

        for i, doc in enumerate(docs, start=1):

            print("=" * 80)

            print(f"Rank : {i}")

            print(f"Similarity : {doc['similarity']:.4f}")

            print(f"Topic : {doc['focus']}")

            print(f"Question : {doc['question']}")

            print(f"Source : {doc['source']}")

            print()