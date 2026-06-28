import time
import logging
import ollama


# Configuration


MODEL_NAME = "llama3.2"


# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# Generate Response


def generate(prompt: str):

    try:

        start = time.time()

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        end = time.time()

        logger.info("LLM response generated successfully.")

        return {

            "answer": response["message"]["content"],

            "model": MODEL_NAME,

            "response_time": round(end - start, 2),

            "status": "success"

        }

    except Exception as e:

        logger.error(str(e))

        return {

            "answer": "An error occurred while generating the response.",

            "model": MODEL_NAME,

            "response_time": 0,

            "status": "error"

        }



# Testing


if __name__ == "__main__":

    prompt = """
Context

Diabetes is a chronic disease.

Question

What is diabetes?
"""

    result = generate(prompt)

    print("\nAnswer\n")

    print(result["answer"])

    print("\nModel :", result["model"])

    print("Response Time :", result["response_time"])