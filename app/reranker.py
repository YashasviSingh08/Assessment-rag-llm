import logging
from sentence_transformers import CrossEncoder

# =====================================================
# Configuration
# =====================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K = 5

# =====================================================
# Logging
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =====================================================
# Load CrossEncoder
# =====================================================

logger.info("Loading CrossEncoder...")

cross_encoder = CrossEncoder(MODEL_NAME)

logger.info("CrossEncoder loaded successfully.")


# =====================================================
# Rerank Function
# =====================================================

def rerank(query: str, retrieved_docs: list, top_k: int = TOP_K):

    if not retrieved_docs:
        return []

    pairs = []

    for doc in retrieved_docs:

        pairs.append(
            (
                query,
                doc["content"]
            )
        )

    scores = cross_encoder.predict(pairs)

    for doc, score in zip(retrieved_docs, scores):

        doc["rerank_score"] = float(score)

    retrieved_docs.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return retrieved_docs[:top_k]


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    from retriever import search

    while True:

        query = input("\nAsk Question : ")

        if query.lower() == "exit":
            break

        docs = search(query)

        docs = rerank(query, docs)

        print("\nBest Documents\n")

        for i, doc in enumerate(docs, start=1):

            print("=" * 80)

            print(f"Rank : {i}")

            print(f"Similarity     : {doc['similarity']:.4f}")

            print(f"Rerank Score   : {doc['rerank_score']:.4f}")

            print(f"Topic          : {doc['focus']}")

            print(f"Question       : {doc['question']}")

            print(f"Source         : {doc['source']}")

            print()