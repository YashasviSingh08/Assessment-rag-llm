from retriever import search
from reranker import rerank

query = "cholestasis of pregnancy"

docs = search(query)

best_docs = rerank(query, docs)

for doc in best_docs:
    print("=" * 60)
    print(doc["focus"])
    print(doc["rerank_score"])