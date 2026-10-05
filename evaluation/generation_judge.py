import json

from langchain_openai import ChatOpenAI

from src.rag_pipeline import RAGPipeline


PDF_PATH = "data/documents/Rahul.pdf"


# ==========================================
# Judge LLM
# ==========================================

judge = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# ==========================================
# Evaluation Dataset
# ==========================================

with open(
    "evaluation/generation_questions.json",
    "r"
) as f:

    questions = json.load(f)


# ==========================================
# RAG Pipeline
# ==========================================

rag = RAGPipeline(
    PDF_PATH
)


# ==========================================
# Evaluation
# ==========================================

for item in questions:

    question = item["question"]

    ground_truth = (
        item["ground_truth"]
    )

    # ======================================
    # Run RAG
    # ======================================

    response = rag.ask(
        question
    )

    answer = response["answer"]

    results = response["results"]

    # ======================================
    # Build Context
    # ======================================

    context = "\n\n".join(
        result["chunk"]["text"]
        for result in results
    )

    # ======================================
    # Judge Prompt
    # ======================================

    prompt = f"""
You are evaluating a Retrieval-Augmented
Generation system.

Evaluate the generated answer using the
question, retrieved context and ground-truth
answer.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Ground Truth:
{ground_truth}

Evaluate the answer on three dimensions.

1. Faithfulness:
Is every important claim in the generated
answer supported by the retrieved context?

2. Relevance:
Does the answer directly answer the question?

3. Correctness:
Does the answer agree with the ground-truth
answer?

Return ONLY valid JSON in this format:

{{
    "faithfulness": 1,
    "relevance": 1,
    "correctness": 1,
    "reason": "short explanation"
}}

Use a score from 0 to 1.
"""


    # ======================================
    # Call Judge
    # ======================================

    response = judge.invoke(
        prompt
    )

    print("\n")
    print("=" * 70)

    print(
        f"Question: {question}"
    )

    print(
        f"Generated Answer: {answer}"
    )

    print(
        "\nJudge Evaluation:"
    )

    print(
        response.content
    )

    print("=" * 70)