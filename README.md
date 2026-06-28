# Healthcare AI Assistant (RAG)

## Overview

This project is a Healthcare AI Assistant built using a Retrieval-Augmented Generation (RAG) pipeline. It answers medical questions using the MedQuAD dataset instead of relying only on the language model's knowledge.

The application retrieves relevant medical information using FAISS, reranks the results with a CrossEncoder model, and generates the final answer using a locally running Llama 3.2 model through Ollama.

In addition to answering medical questions, the application can identify appointment booking requests and route them to a separate appointment workflow.

---

## Features

* Medical question answering using RAG
* Local Llama 3.2 model with Ollama
* Semantic search using Sentence Transformers
* FAISS vector database
* CrossEncoder reranking
* FastAPI REST API
* Appointment request routing
* Rebuild vector database using `/ingest`
* Source citations with every answer

---

## Tech Stack

* Python
* FastAPI
* FAISS
* Sentence Transformers
* CrossEncoder
* Ollama
* Llama 3.2
* Pandas
* NumPy

---

## Project Structure

```text
Healthcare-AI-Assistant/

app/
│── main.py
│── rag.py
│── retriever.py
│── reranker.py
│── prompt.py
│── llm.py
│── embeddings.py
│── appointment.py
│── agent.py

Scripts/
│── processed/
│     processed_medquad.csv

vector_store/
│── medical_knowledge.index
│── medical_metadata.json

requirements.txt
Dockerfile
README.md
```

---

## How It Works

1. The user sends a question.
2. The system checks whether it is a medical question or an appointment request.
3. For medical questions:

   * The question is converted into an embedding.
   * FAISS retrieves the most relevant documents.
   * A CrossEncoder reranks those documents.
   * A prompt is built using the best results.
   * Llama 3.2 generates the final answer.
4. The response is returned along with the source documents.

---

## API Endpoints

### GET /

Returns a simple welcome message.

---

### GET /health

Returns the API health status.

---

### POST /ask

Ask a medical question.

Example:

```json
{
    "question": "What are the symptoms of diabetes?"
}
```

---

### POST /ingest

Rebuilds the vector database from the processed MedQuAD dataset.

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
```

Make sure the model is available:

```bash
ollama pull llama3.2
```

### 3. Run FastAPI

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## Example Questions

* What are the symptoms of diabetes?
* How is childhood acute lymphoblastic leukemia diagnosed?
* What causes asthma?
* What are the treatments for leukemia?

Appointment example:

* Book an appointment with a cardiologist tomorrow.

---

## Future Improvements

If I continue working on this project, I would like to add:

* Hybrid search (BM25 + FAISS)
* Conversation memory
* Better intent detection using an LLM
* Authentication for the API
* Background document ingestion
* Support for additional medical datasets

---

## Notes

This project was built to demonstrate the complete workflow of a RAG application, including document preprocessing, embedding generation, retrieval, reranking, prompt engineering, API development, and local LLM integration.

The focus was on building a modular and easy-to-understand architecture rather than only producing answers.

---

## Author

**Yashasvi Singh**
