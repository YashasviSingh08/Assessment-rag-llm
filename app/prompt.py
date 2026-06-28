# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM_PROMPT = """
You are an expert Healthcare AI Assistant.

Your job is to answer ONLY using the retrieved medical documents.

STRICT RULES

1. Use ONLY the retrieved medical documents.

2. Never use outside knowledge.

3. Never guess.

4. Never make up treatments or diagnoses.

5. If the answer is not available in the retrieved documents, reply exactly:

"I could not find this information in the provided documents."

6. If multiple retrieved documents contain useful information, combine them into one coherent answer.

7. Keep the answer concise, factual and professional.

8. Do NOT mention:
- Document numbers
- Similarity scores
- Rerank scores
- Internal system details

9. Do NOT mention sources in the answer.
The application will provide citations separately.
"""


# =====================================================
# Build Prompt
# =====================================================

def build_prompt(question: str, documents: list):

    context = ""

    for i, doc in enumerate(documents, start=1):

        context += f"""
==================================================
Medical Document {i}

Medical Topic:
{doc['focus']}

Question:
{doc['question']}

Answer:
{doc['answer']}

Source:
{doc['source']}
==================================================

"""

    prompt = f"""
{SYSTEM_PROMPT}

Retrieved Medical Documents

{context}

User Question

{question}

Answer:
"""

    return prompt


# =====================================================
# Debug
# =====================================================

if __name__ == "__main__":

    docs = [
        {
            "focus": "Diabetes",
            "question": "What is diabetes?",
            "answer": "Diabetes is a chronic disease in which blood sugar levels remain high.",
            "source": "MedlinePlus"
        }
    ]

    prompt = build_prompt(
        "Explain diabetes",
        docs
    )

    print(prompt)