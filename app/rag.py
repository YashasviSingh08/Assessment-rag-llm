import logging

from app.retriever import search
from app.reranker import rerank
from app.prompt import build_prompt
from app.llm import generate
from app.logger import logger

from app.agent import detect_intent
from app.appointment import book_appointment

# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Confidence


def calculate_confidence(similarity, rerank_score):

    if similarity >= 0.75 and rerank_score >= 8:
        return "High"

    elif similarity >= 0.60 and rerank_score >= 5:
        return "Medium"

    return "Low"


# RAG Pipeline


def ask(question):

    logger.info(f"Question: {question}")

    # Step 1 : Detect Intent


    intent = detect_intent(question)

    logger.info(f"Detected Intent : {intent}")

    # Appointment Tool


    if intent == "appointment":

        return book_appointment(question)

    # Medical RAG


    retrieved_docs = search(question)

    if not retrieved_docs:

        return {

            "question": question,

            "answer": "I could not find this information in the provided documents.",

            "confidence": "Low",

            "sources": []

        }

    # Rerank

    best_docs = rerank(question, retrieved_docs)


    # Prompt


    prompt = build_prompt(question, best_docs)


    # LLM


    llm_response = generate(prompt)


    # Sources


    sources = []

    for doc in best_docs:

        sources.append({

            "topic": doc["focus"],

            "question": doc["question"],

            "source": doc["source"],

            "similarity": round(doc["similarity"], 4),

            "rerank_score": round(doc["rerank_score"], 4)

        })

    confidence = calculate_confidence(

        best_docs[0]["similarity"],

        best_docs[0]["rerank_score"]

    )

    return {

        "question": question,

        "answer": llm_response["answer"],

        "confidence": confidence,

        "model": llm_response["model"],

        "response_time": llm_response["response_time"],

        "sources": sources

    }


# Testing

if __name__ == "__main__":

    while True:

        query = input("\nAsk Question : ")

        if query.lower() == "exit":
            break

        result = ask(query)

        print("\n")

        print("=" * 100)

        print("RESULT")

        print("=" * 100)

        if "answer" in result:

            print("\nQuestion:\n")

            print(result["question"])

            print("\nAnswer:\n")

            print(result["answer"])

            if "confidence" in result:

                print("\nConfidence:", result["confidence"])

            if "model" in result:

                print("\nModel:", result["model"])

            if "response_time" in result:

                print("\nResponse Time:", result["response_time"])

            if "sources" in result:

                print("\nSources:\n")

                for source in result["sources"]:

                    print("-" * 50)

                    print("Topic:", source["topic"])

                    print("Question:", source["question"])

                    print("Source:", source["source"])

                    print("Similarity:", source["similarity"])

                    print("Rerank Score:", source["rerank_score"])

        else:

            print(result)

        print("=" * 100)