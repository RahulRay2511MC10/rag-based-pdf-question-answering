import os
import sys
import json
import re
import statistics

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.rag_pipeline import RAGPipeline


# ============================================================
# 2. CONFIGURATION
# ============================================================

load_dotenv()

PDF_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "documents",
    "Rahul.pdf"
)

QUESTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "questions.json"
)


# ============================================================
# 3. MODELS
# ============================================================

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

judge_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# ============================================================
# 4. LOAD QUESTIONS
# ============================================================

def load_questions(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        data = data["questions"]

    return data


# ============================================================
# 5. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 6. SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(
    expected_answer,
    generated_answer
):

    embeddings = embedding_model.embed_documents(
        [
            expected_answer,
            generated_answer
        ]
    )

    a = embeddings[0]
    b = embeddings[1]

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = (
        sum(x * x for x in a)
    ) ** 0.5

    norm_b = (
        sum(x * x for x in b)
    ) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


# ============================================================
# 7. LLM-AS-A-JUDGE
# ============================================================

def llm_judge(
    question,
    expected_answer,
    generated_answer,
    context
):

    prompt = f"""
You are evaluating the quality of an answer produced by
a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer against the reference answer
and the supplied context.

Do NOT use outside knowledge.

QUESTION:
{question}

REFERENCE ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

RETRIEVED CONTEXT:
{context}

Evaluate three dimensions.

1. ANSWER CORRECTNESS
Does the generated answer convey the same factual information
as the reference answer?

2. ANSWER RELEVANCE
Does the generated answer directly answer the question without
unnecessary information?

3. FAITHFULNESS
Is the generated answer fully supported by the retrieved context?

Give each score from 1 to 5.

Return ONLY valid JSON in exactly this format:

{{
    "correctness": 1,
    "relevance": 1,
    "faithfulness": 1,
    "reason": "brief explanation"
}}
"""

    response = judge_llm.invoke(
        prompt
    )

    content = response.content.strip()

    # Remove markdown fences if the model returns them
    content = re.sub(
        r"```json|```",
        "",
        content
    ).strip()

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError:

        return {
            "correctness": 0,
            "relevance": 0,
            "faithfulness": 0,
            "reason": (
                "Judge returned invalid JSON."
            )
        }

    return result


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL ANSWER QUALITY EVALUATION")
    print("=" * 70)

    questions = load_questions(
        QUESTIONS_PATH
    )

    print(
        f"\nNumber of questions: {len(questions)}"
    )

    print("\nInitializing RAG pipeline...")

    # --------------------------------------------------------
    # IMPORTANT:
    # Load and chunk the PDF exactly as your existing
    # pipeline expects.
    # --------------------------------------------------------

    from src.loader import load_pdf
    from src.chunker import chunk_documents

    documents = load_pdf(
        PDF_PATH
    )

    chunks = chunk_documents(
        documents
    )

    rag = RAGPipeline(
        chunks
    )

    semantic_scores = []

    correctness_scores = []
    relevance_scores = []
    faithfulness_scores = []

    results = []


    # ========================================================
    # EVALUATE EACH QUESTION
    # ========================================================

    for i, item in enumerate(
        questions,
        start=1
    ):

        question = item.get(
            "question"
        )

        expected_answer = item.get(
            "expected_answer",
            item.get(
                "answer",
                ""
            )
        )

        print("\n")
        print("-" * 70)
        print(
            f"QUESTION {i}/{len(questions)}"
        )
        print("-" * 70)

        print(
            "Question:",
            question
        )

        print(
            "Expected:",
            expected_answer
        )


        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        result = rag.ask(
            question=question,
            chat_history=[]
        )

        generated_answer = result.get(
            "answer",
            ""
        )

        print(
            "Generated:",
            generated_answer
        )


        # ----------------------------------------------------
        # SEMANTIC SIMILARITY
        # ----------------------------------------------------

        similarity = semantic_similarity(
            expected_answer,
            generated_answer
        )

        semantic_scores.append(
            similarity
        )

        print(
            f"Semantic Similarity: "
            f"{similarity:.4f}"
        )


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for retrieval_result in result.get(
            "results",
            []
        ):

            chunk = retrieval_result.get(
                "chunk",
                {}
            )

            text = chunk.get(
                "text",
                ""
            )

            if text:
                context_parts.append(
                    text
                )

        context = "\n\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # LLM JUDGE
        # ----------------------------------------------------

        judge_result = llm_judge(
            question,
            expected_answer,
            generated_answer,
            context
        )

        correctness = float(
            judge_result.get(
                "correctness",
                0
            )
        )

        relevance = float(
            judge_result.get(
                "relevance",
                0
            )
        )

        faithfulness = float(
            judge_result.get(
                "faithfulness",
                0
            )
        )

        correctness_scores.append(
            correctness
        )

        relevance_scores.append(
            relevance
        )

        faithfulness_scores.append(
            faithfulness
        )

        print(
            f"LLM Correctness: "
            f"{correctness:.2f}/5"
        )

        print(
            f"LLM Relevance: "
            f"{relevance:.2f}/5"
        )

        print(
            f"LLM Faithfulness: "
            f"{faithfulness:.2f}/5"
        )

        print(
            "Judge Reason:",
            judge_result.get(
                "reason",
                ""
            )
        )


        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "semantic_similarity": similarity,
                "judge": judge_result
            }
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    avg_similarity = (
        statistics.mean(
            semantic_scores
        )
        if semantic_scores
        else 0
    )

    avg_correctness = (
        statistics.mean(
            correctness_scores
        )
        if correctness_scores
        else 0
    )

    avg_relevance = (
        statistics.mean(
            relevance_scores
        )
        if relevance_scores
        else 0
    )

    avg_faithfulness = (
        statistics.mean(
            faithfulness_scores
        )
        if faithfulness_scores
        else 0
    )


    print("\n")
    print("=" * 70)
    print("FINAL ANSWER QUALITY EVALUATION")
    print("=" * 70)

    print(
        f"\nNumber of questions: "
        f"{len(questions)}"
    )

    print("\nSEMANTIC SIMILARITY")
    print("-" * 70)

    print(
        f"Average Semantic Similarity: "
        f"{avg_similarity:.4f}"
    )

    print(
        f"Percentage: "
        f"{avg_similarity * 100:.2f}%"
    )


    print("\nLLM-AS-A-JUDGE")
    print("-" * 70)

    print(
        f"Answer Correctness: "
        f"{avg_correctness:.2f} / 5"
    )

    print(
        f"Answer Relevance: "
        f"{avg_relevance:.2f} / 5"
    )

    print(
        f"Faithfulness: "
        f"{avg_faithfulness:.2f} / 5"
    )


    print("\nNORMALIZED LLM SCORES")
    print("-" * 70)

    print(
        f"Answer Correctness: "
        f"{avg_correctness / 5 * 100:.2f}%"
    )

    print(
        f"Answer Relevance: "
        f"{avg_relevance / 5 * 100:.2f}%"
    )

    print(
        f"Faithfulness: "
        f"{avg_faithfulness / 5 * 100:.2f}%"
    )


    print("\n")
    print("=" * 70)
    print("Evaluation completed.")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()