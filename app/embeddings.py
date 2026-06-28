import os
import json
import logging

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from app.logger import logger


# Configuration

DATA_PATH = "scripts/processed/processed_medquad.csv"

VECTOR_STORE = "vector_store"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 256

INDEX_FILE = "medical_knowledge.index"

METADATA_FILE = "medical_metadata.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

os.makedirs(VECTOR_STORE, exist_ok=True)


def build_vector_store():

    logger.info("Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    logger.info(f"Total Records : {len(df)}")

    logger.info(f"Loading Embedding Model : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    logger.info("Embedding model loaded successfully.")

    texts = df["content"].tolist()

    embeddings = []

    logger.info("Generating embeddings...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):

        batch = texts[i:i+BATCH_SIZE]

        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    logger.info(f"Embedding Shape : {embeddings.shape}")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    logger.info(f"Vectors Indexed : {index.ntotal}")

    index_path = os.path.join(VECTOR_STORE, INDEX_FILE)

    faiss.write_index(index, index_path)

    metadata = df[
        [
            "document_id",
            "focus",
            "qtype",
            "question",
            "answer",
            "source",
            "url",
            "content"
        ]
    ].to_dict(orient="records")

    metadata_path = os.path.join(
        VECTOR_STORE,
        METADATA_FILE
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )

    logger.info("Embedding pipeline completed successfully.")

    return {

        "status": "success",

        "documents": len(df),

        "vectors": index.ntotal,

        "index": index_path,

        "metadata": metadata_path

    }


if __name__ == "__main__":

    result = build_vector_store()

    print(result)