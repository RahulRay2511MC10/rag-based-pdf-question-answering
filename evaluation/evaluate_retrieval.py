import os
import sys
import json
import statistics


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# IMPORTANT:
# Add project root BEFORE importing src
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 2. IMPORT PROJECT MODULES
# ============================================================

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.rag_pipeline import RAGPipeline


# ============================================================
# 3. CONFIGURATION
# ============================================================

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

TOP_K = 10
# ============================================================
# 4. LOAD QUESTIONS
# ============================================================

def load_questions(path):
    """
    Load evaluation questions from questions.json.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Questions file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # Handle either:
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # or:
    #
    # {
    #   "questions": [...]
    # }

    if isinstance(data, dict):

        if "questions" in data:
            data = data["questions"]

        else:
            raise ValueError(
                "questions.json contains a dictionary "
                "but no 'questions' key was found."
            )

    if not isinstance(data, list):
        raise ValueError(
            "questions.json must contain a list of questions."
        )

    return data


# ============================================================
# 5. EXTRACT QUESTION
# ============================================================

def get_question(item):
    """
    Supports common question field names.
    """

    question = (
        item.get("question")
        or item.get("query")
        or item.get("q")
    )

    if not question:
        raise ValueError(
            f"Could not find question in:\n{item}"
        )

    return question


# ============================================================
# 6. EXTRACT RELEVANT PAGES
# ============================================================

def get_relevant_pages(item):
    """
    Extract ground-truth relevant pages.
    """

    pages = (
        item.get("relevant_pages")
        or item.get("relevant_page")
        or item.get("pages")
        or item.get("page")
    )

    if pages is None:
        return []

    if isinstance(pages, int):
        return [pages]

    if isinstance(pages, str):

        try:
            return [int(pages)]

        except ValueError:
            return []

    return [
        int(page)
        for page in pages
        if str(page).isdigit()
    ]


# ============================================================
# 7. EXTRACT PAGE FROM RETRIEVAL RESULT
# ============================================================

def get_page_from_result(result):
    """
    Safely extract page number from the current
    retrieval result structure.

    Expected structure:

    {
        "chunk": {
            "text": "...",
            "metadata": {
                "page": 1
            }
        },
        "distance": 1.23
    }
    """

    chunk = result.get(
        "chunk",
        {}
    )

    metadata = chunk.get(
        "metadata",
        {}
    )

    return metadata.get(
        "page"
    )


# ============================================================
# 8. EXTRACT SOURCE FROM RETRIEVAL RESULT
# ============================================================

def get_source_from_result(result):

    chunk = result.get(
        "chunk",
        {}
    )

    metadata = chunk.get(
        "metadata",
        {}
    )

    return metadata.get(
        "source"
    )


# ============================================================
# 9. CALCULATE HIT@K
# ============================================================

def calculate_hit_at_k(
    retrieved_pages,
    relevant_pages
):

    if not relevant_pages:
        return 0

    for page in retrieved_pages:

        if page in relevant_pages:
            return 1

    return 0


# ============================================================
# 10. CALCULATE RECIPROCAL RANK
# ============================================================

def calculate_reciprocal_rank(
    retrieved_pages,
    relevant_pages
):

    if not relevant_pages:
        return 0.0

    for rank, page in enumerate(
        retrieved_pages,
        start=1
    ):

        if page in relevant_pages:

            return 1.0 / rank

    return 0.0


# ============================================================
# 11. CALCULATE PRECISION@K
# ============================================================

def calculate_precision_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    if k == 0:
        return 0.0

    relevant_count = 0

    for page in retrieved_pages[:k]:

        if page in relevant_pages:
            relevant_count += 1

    return relevant_count / k


# ============================================================
# 12. CALCULATE RECALL@K
# ============================================================

def calculate_recall_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    if not relevant_pages:
        return 0.0

    retrieved_relevant = set(
        retrieved_pages[:k]
    ).intersection(
        set(relevant_pages)
    )

    return (
        len(retrieved_relevant)
        /
        len(set(relevant_pages))
    )


# ============================================================
# 13. PRINT RETRIEVAL RESULTS
# ============================================================

def print_results(
    results,
    relevant_pages
):

    retrieved_pages = []

    print()
    print(
        "-" * 70
    )

    print(
        "RETRIEVED RESULTS"
    )

    print(
        "-" * 70
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        page = get_page_from_result(
            result
        )

        source = get_source_from_result(
            result
        )

        distance = result.get(
            "distance"
        )

        chunk = result.get(
            "chunk",
            {}
        )

        text = chunk.get(
            "text",
            ""
        )

        if page is not None:
            retrieved_pages.append(
                page
            )

        print(
            f"\nRank {rank}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Distance: {distance}"
        )

        print(
            f"Source: {source}"
        )

        print(
            "Text:"
        )

        print(
            text[:500]
        )

    return retrieved_pages


# ============================================================
# 14. MAIN EVALUATION
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "RAG RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPDF: {PDF_PATH}"
    )

    print(
        f"Questions: {QUESTIONS_PATH}"
    )

    print(
        f"Top-K: {TOP_K}"
    )


    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_questions(
        QUESTIONS_PATH
    )

    print(
        f"\nNumber of evaluation questions: "
        f"{len(questions)}"
    )


    # --------------------------------------------------------
    # Initialize RAG
    # --------------------------------------------------------

    print()
    print("\nInitializing RAG pipeline...")

# ------------------------------------------------------------
# LOAD PDF
# ------------------------------------------------------------

    documents = load_pdf(PDF_PATH)

    print(
    f"Loaded {len(documents)} document pages."
    )

# ------------------------------------------------------------
# CHUNK DOCUMENTS
# ------------------------------------------------------------

    chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=100
        )

    print(
            f"Created {len(chunks)} chunks."
        )

        # ------------------------------------------------------------
        # INITIALIZE RAG PIPELINE
        # ------------------------------------------------------------

    rag = RAGPipeline(
            chunks=chunks
        )


    # --------------------------------------------------------
    # Evaluation storage
    # --------------------------------------------------------

    hit_scores = []

    reciprocal_ranks = []

    precision_scores = []

    recall_scores = []


    # ========================================================
    # EVALUATE EACH QUESTION
    # ========================================================

    for question_number, item in enumerate(
        questions,
        start=1
    ):

        question = get_question(
            item
        )

        relevant_pages = get_relevant_pages(
            item
        )


        print()
        print()
        print(
            "=" * 70
        )

        print(
            f"Question {question_number}"
        )

        print(
            "=" * 70
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Relevant pages: {relevant_pages}"
        )


        # ----------------------------------------------------
        # FAISS RETRIEVAL
        # ----------------------------------------------------

        print()
        print(
            "Running FAISS retrieval..."
        )

        try:

            results = rag.retriever.retrieve(
                question,
                k=TOP_K
            )

        except TypeError:

            # Fallback in case the current Retriever
            # does not accept k as a keyword.

            results = rag.retriever.retrieve(
                question,
                TOP_K
            )


        # ----------------------------------------------------
        # Print retrieved chunks
        # ----------------------------------------------------

        retrieved_pages = print_results(
            results,
            relevant_pages
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        hit = calculate_hit_at_k(
            retrieved_pages,
            relevant_pages
        )

        reciprocal_rank = calculate_reciprocal_rank(
            retrieved_pages,
            relevant_pages
        )

        precision = calculate_precision_at_k(
            retrieved_pages,
            relevant_pages,
            TOP_K
        )

        recall = calculate_recall_at_k(
            retrieved_pages,
            relevant_pages,
            TOP_K
        )


        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        hit_scores.append(
            hit
        )

        reciprocal_ranks.append(
            reciprocal_rank
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )


        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print()
        print(
            "-" * 70
        )

        print(
            "QUESTION METRICS"
        )

        print(
            "-" * 70
        )

        print(
            f"Retrieved pages: {retrieved_pages}"
        )

        print(
            f"Hit@{TOP_K}: {hit}"
        )

        print(
            f"Reciprocal Rank: "
            f"{reciprocal_rank:.4f}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{precision:.4f}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{recall:.4f}"
        )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    print()
    print()
    print(
        "=" * 70
    )

    print(
        "FINAL RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )


    if not hit_scores:

        print(
            "\nNo evaluation results available."
        )

        return


    hit_rate = (
        sum(hit_scores)
        /
        len(hit_scores)
    )

    mrr = statistics.mean(
        reciprocal_ranks
    )

    mean_precision = statistics.mean(
        precision_scores
    )

    mean_recall = statistics.mean(
        recall_scores
    )


    print()

    print(
        f"Number of questions: "
        f"{len(hit_scores)}"
    )

    print(
        f"Hit@{TOP_K}: "
        f"{hit_rate:.4f}"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    print(
        f"Precision@{TOP_K}: "
        f"{mean_precision:.4f}"
    )

    print(
        f"Recall@{TOP_K}: "
        f"{mean_recall:.4f}"
    )


    print()
    print(
        "=" * 70
    )

    print(
        "Evaluation completed."
    )

    print(
        "=" * 70
    )


# ============================================================
# 15. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()