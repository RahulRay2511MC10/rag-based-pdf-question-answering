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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 2. IMPORT RAG PIPELINE
# ============================================================

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

RETRIEVAL_K = 10
EVALUATION_K = 5


# ============================================================
# 4. LOAD QUESTIONS
# ============================================================

def load_questions(path):

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

    if isinstance(data, dict):

        if "questions" in data:

            data = data["questions"]

        else:

            raise ValueError(
                "questions.json must contain "
                "'questions' key."
            )

    if not isinstance(data, list):

        raise ValueError(
            "questions.json must contain a list."
        )

    return data


# ============================================================
# 5. PAGE EXTRACTION
# ============================================================

def get_page_from_result(result):

    chunk = result.get(
        "chunk",
        {}
    )

    if not isinstance(chunk, dict):
        return None

    metadata = chunk.get(
        "metadata",
        {}
    )

    if not isinstance(metadata, dict):
        return None

    return metadata.get("page")


# ============================================================
# 6. SOURCE EXTRACTION
# ============================================================

def get_source_from_result(result):

    chunk = result.get(
        "chunk",
        {}
    )

    if not isinstance(chunk, dict):
        return None

    metadata = chunk.get(
        "metadata",
        {}
    )

    if not isinstance(metadata, dict):
        return None

    return metadata.get("source")


# ============================================================
# 7. FIRST RELEVANT RANK
# ============================================================

def first_relevant_rank(
    retrieved_pages,
    relevant_pages
):

    relevant_pages = set(
        relevant_pages
    )

    for rank, page in enumerate(
        retrieved_pages,
        start=1
    ):

        if page in relevant_pages:

            return rank

    return None


# ============================================================
# 8. HIT@K
# ============================================================

def hit_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    retrieved_pages = retrieved_pages[:k]

    relevant_pages = set(
        relevant_pages
    )

    for page in retrieved_pages:

        if page in relevant_pages:

            return 1

    return 0


# ============================================================
# 9. MRR
# ============================================================

def reciprocal_rank(
    retrieved_pages,
    relevant_pages
):

    rank = first_relevant_rank(
        retrieved_pages,
        relevant_pages
    )

    if rank is None:

        return 0.0

    return 1.0 / rank


# ============================================================
# 10. PRECISION@K
# ============================================================

def precision_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    retrieved_pages = retrieved_pages[:k]

    if not retrieved_pages:

        return 0.0

    relevant_pages = set(
        relevant_pages
    )

    relevant_count = sum(
        1
        for page in retrieved_pages
        if page in relevant_pages
    )

    return relevant_count / len(
        retrieved_pages
    )


# ============================================================
# 11. RECALL@K
# ============================================================

def recall_at_k(
    retrieved_pages,
    relevant_pages,
    k
):

    if not relevant_pages:

        return 0.0

    retrieved_pages = set(
        retrieved_pages[:k]
    )

    relevant_pages = set(
        relevant_pages
    )

    matched = len(
        retrieved_pages.intersection(
            relevant_pages
        )
    )

    return matched / len(
        relevant_pages
    )


# ============================================================
# 12. CALCULATE ALL METRICS
# ============================================================

def calculate_metrics(
    retrieved_pages,
    relevant_pages,
    k
):

    return {

        "hit": hit_at_k(
            retrieved_pages,
            relevant_pages,
            k
        ),

        "mrr": reciprocal_rank(
            retrieved_pages,
            relevant_pages
        ),

        "precision": precision_at_k(
            retrieved_pages,
            relevant_pages,
            k
        ),

        "recall": recall_at_k(
            retrieved_pages,
            relevant_pages,
            k
        )
    }


# ============================================================
# 13. PRINT RESULT
# ============================================================

def print_question_result(
    question_number,
    question,
    relevant_pages,
    faiss_pages,
    reranked_pages,
    faiss_metrics,
    reranked_metrics
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"QUESTION {question_number}"
    )

    print(
        "-" * 70
    )

    print(
        "Question:",
        question
    )

    print(
        "Relevant pages:",
        relevant_pages
    )

    print(
        "\nFAISS Top-5:"
    )

    print(
        faiss_pages
    )

    print(
        "\nReranked Top-5:"
    )

    print(
        reranked_pages
    )

    # --------------------------------------------------------
    # First relevant rank
    # --------------------------------------------------------

    faiss_rank = first_relevant_rank(
        faiss_pages,
        relevant_pages
    )

    reranked_rank = first_relevant_rank(
        reranked_pages,
        relevant_pages
    )

    print(
        "\nFAISS first relevant rank:",
        faiss_rank
    )

    print(
        "Reranked first relevant rank:",
        reranked_rank
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    print(
        "\nFAISS metrics:"
    )

    print(
        "  Hit@5:",
        faiss_metrics["hit"]
    )

    print(
        "  MRR:",
        round(
            faiss_metrics["mrr"],
            4
        )
    )

    print(
        "  Precision@5:",
        round(
            faiss_metrics["precision"],
            4
        )
    )

    print(
        "  Recall@5:",
        round(
            faiss_metrics["recall"],
            4
        )
    )

    print(
        "\nReranked metrics:"
    )

    print(
        "  Hit@5:",
        reranked_metrics["hit"]
    )

    print(
        "  MRR:",
        round(
            reranked_metrics["mrr"],
            4
        )
    )

    print(
        "  Precision@5:",
        round(
            reranked_metrics["precision"],
            4
        )
    )

    print(
        "  Recall@5:",
        round(
            reranked_metrics["recall"],
            4
        )
    )

    # --------------------------------------------------------
    # Determine improvement
    # --------------------------------------------------------

    faiss_mrr = faiss_metrics["mrr"]
    reranked_mrr = reranked_metrics["mrr"]

    if reranked_mrr > faiss_mrr:

        status = "IMPROVED"

    elif reranked_mrr < faiss_mrr:

        status = "WORSE"

    else:

        status = "SAME"

    print(
        "\nResult:",
        status
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "RERANKER EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nPDF:",
        PDF_PATH
    )

    print(
        "Questions:",
        QUESTIONS_PATH
    )

    print(
        "Retrieval K:",
        RETRIEVAL_K
    )

    print(
        "Evaluation K:",
        EVALUATION_K
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    evaluation_dataset = load_questions(
        QUESTIONS_PATH
    )

    print(
        "\nNumber of evaluation questions:",
        len(evaluation_dataset)
    )

    # ========================================================
    # IMPORTANT:
    # LOAD PDF AND CREATE CHUNKS
    # ========================================================

    from src.loader import load_pdf
    from src.chunker import chunk_documents

    print(
        "\nLoading PDF..."
    )

    documents = load_pdf(
        PDF_PATH
    )

    print(
        "Number of documents:",
        len(documents)
    )

    print(
        "\nCreating chunks..."
    )

    chunks = chunk_documents(
        documents
    )

    print(
        "Number of chunks:",
        len(chunks)
    )

    # ========================================================
    # INITIALIZE RAG
    # ========================================================

    print(
        "\nInitializing RAG pipeline..."
    )

    rag = RAGPipeline(
        chunks
    )

    # ========================================================
    # STORAGE
    # ========================================================

    faiss_results = []

    reranked_results = []

    diagnostic_results = []

    # ========================================================
    # EVALUATE EACH QUESTION
    # ========================================================

    for question_number, item in enumerate(
        evaluation_dataset,
        start=1
    ):

        question = item[
            "question"
        ]

        relevant_pages = item[
            "relevant_pages"
        ]

        print(
            "\n\n" + "=" * 70
        )

        print(
            f"QUESTION {question_number}"
        )

        print(
            "=" * 70
        )

        print(
            "Question:",
            question
        )

        print(
            "Relevant pages:",
            relevant_pages
        )

        # ====================================================
        # FAISS RETRIEVAL
        # ====================================================

        print(
            "\nRunning FAISS retrieval..."
        )

        faiss_retrieved = rag.retriever.retrieve(
            question,
            k=RETRIEVAL_K
        )

        # ----------------------------------------------------
        # Extract pages
        # ----------------------------------------------------

        faiss_pages = []

        for result in faiss_retrieved:

            page = get_page_from_result(
                result
            )

            if page is not None:

                faiss_pages.append(
                    page
                )

        # ----------------------------------------------------
        # FAISS Top-K
        # ----------------------------------------------------

        faiss_top5_pages = faiss_pages[
            :EVALUATION_K
        ]

        # ====================================================
        # RERANKING
        # ====================================================

        print(
            "\nRunning CrossEncoder reranking..."
        )

        reranked = rag.reranker.rerank(
            query=question,
            results=faiss_retrieved,
            top_k=EVALUATION_K
        )

        # ----------------------------------------------------
        # Extract reranked pages
        # ----------------------------------------------------

        reranked_pages = []

        for result in reranked:

            page = get_page_from_result(
                result
            )

            if page is not None:

                reranked_pages.append(
                    page
                )

        # ====================================================
        # METRICS
        # ====================================================

        faiss_metrics = calculate_metrics(
            faiss_top5_pages,
            relevant_pages,
            EVALUATION_K
        )

        reranked_metrics = calculate_metrics(
            reranked_pages,
            relevant_pages,
            EVALUATION_K
        )

        # ====================================================
        # SAVE AGGREGATE RESULTS
        # ====================================================

        faiss_results.append(
            faiss_metrics
        )

        reranked_results.append(
            reranked_metrics
        )

        # ====================================================
        # FIRST RELEVANT RANK
        # ====================================================

        faiss_rank = first_relevant_rank(
            faiss_top5_pages,
            relevant_pages
        )

        reranked_rank = first_relevant_rank(
            reranked_pages,
            relevant_pages
        )

        # ====================================================
        # STATUS
        # ====================================================

        if (
            reranked_metrics["mrr"]
            >
            faiss_metrics["mrr"]
        ):

            status = "IMPROVED"

        elif (
            reranked_metrics["mrr"]
            <
            faiss_metrics["mrr"]
        ):

            status = "WORSE"

        else:

            status = "SAME"

        # ====================================================
        # SAVE DIAGNOSTIC RESULT
        # ====================================================

        diagnostic_results.append({

            "question": question,

            "relevant_pages": relevant_pages,

            "faiss_pages": faiss_top5_pages,

            "reranked_pages": reranked_pages,

            "faiss_rank": faiss_rank,

            "reranked_rank": reranked_rank,

            "faiss_mrr": faiss_metrics["mrr"],

            "reranked_mrr": reranked_metrics["mrr"],

            "faiss_hit": faiss_metrics["hit"],

            "reranked_hit": reranked_metrics["hit"],

            "faiss_precision":
                faiss_metrics["precision"],

            "reranked_precision":
                reranked_metrics["precision"],

            "faiss_recall":
                faiss_metrics["recall"],

            "reranked_recall":
                reranked_metrics["recall"],

            "status": status
        })

        # ====================================================
        # PRINT QUESTION RESULT
        # ====================================================

        print_question_result(
            question_number,
            question,
            relevant_pages,
            faiss_top5_pages,
            reranked_pages,
            faiss_metrics,
            reranked_metrics
        )

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    def average(
        results,
        key
    ):

        if not results:

            return 0.0

        return statistics.mean(
            result[key]
            for result in results
        )

    # ========================================================
    # FINAL FAISS RESULTS
    # ========================================================

    faiss_hit = average(
        faiss_results,
        "hit"
    )

    faiss_mrr = average(
        faiss_results,
        "mrr"
    )

    faiss_precision = average(
        faiss_results,
        "precision"
    )

    faiss_recall = average(
        faiss_results,
        "recall"
    )

    # ========================================================
    # FINAL RERANKER RESULTS
    # ========================================================

    reranked_hit = average(
        reranked_results,
        "hit"
    )

    reranked_mrr = average(
        reranked_results,
        "mrr"
    )

    reranked_precision = average(
        reranked_results,
        "precision"
    )

    reranked_recall = average(
        reranked_results,
        "recall"
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "FINAL RERANKER EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nNumber of questions:",
        len(evaluation_dataset)
    )

    print(
        "\nFAISS"
    )

    print(
        "Hit@5:",
        f"{faiss_hit:.4f}"
    )

    print(
        "MRR:",
        f"{faiss_mrr:.4f}"
    )

    print(
        "Precision@5:",
        f"{faiss_precision:.4f}"
    )

    print(
        "Recall@5:",
        f"{faiss_recall:.4f}"
    )

    print(
        "\nRERANKED"
    )

    print(
        "Hit@5:",
        f"{reranked_hit:.4f}"
    )

    print(
        "MRR:",
        f"{reranked_mrr:.4f}"
    )

    print(
        "Precision@5:",
        f"{reranked_precision:.4f}"
    )

    print(
        "Recall@5:",
        f"{reranked_recall:.4f}"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    print(
        "\n\n" + "-" * 60
    )

    print(
        "RERANKER IMPROVEMENT"
    )

    print(
        "-" * 60
    )

    print(
        "Hit@5 improvement:",
        f"{reranked_hit - faiss_hit:+.4f}"
    )

    print(
        "MRR improvement:",
        f"{reranked_mrr - faiss_mrr:+.4f}"
    )

    print(
        "Precision@5 improvement:",
        f"{reranked_precision - faiss_precision:+.4f}"
    )

    print(
        "Recall@5 improvement:",
        f"{reranked_recall - faiss_recall:+.4f}"
    )

    # ========================================================
    # PER-QUESTION ANALYSIS
    # ========================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "PER-QUESTION RERANKER ANALYSIS"
    )

    print(
        "=" * 70
    )

    for i, result in enumerate(
        diagnostic_results,
        start=1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Question {i}"
        )

        print(
            "-" * 70
        )

        print(
            "Question:",
            result["question"]
        )

        print(
            "Relevant pages:",
            result["relevant_pages"]
        )

        print(
            "FAISS Top-5:",
            result["faiss_pages"]
        )

        print(
            "Reranked Top-5:",
            result["reranked_pages"]
        )

        print(
            "FAISS first relevant rank:",
            result["faiss_rank"]
        )

        print(
            "Reranked first relevant rank:",
            result["reranked_rank"]
        )

        print(
            "FAISS MRR:",
            f'{result["faiss_mrr"]:.4f}'
        )

        print(
            "Reranked MRR:",
            f'{result["reranked_mrr"]:.4f}'
        )

        print(
            "FAISS Hit@5:",
            result["faiss_hit"]
        )

        print(
            "Reranked Hit@5:",
            result["reranked_hit"]
        )

        print(
            "FAISS Precision@5:",
            f'{result["faiss_precision"]:.4f}'
        )

        print(
            "Reranked Precision@5:",
            f'{result["reranked_precision"]:.4f}'
        )

        print(
            "FAISS Recall@5:",
            f'{result["faiss_recall"]:.4f}'
        )

        print(
            "Reranked Recall@5:",
            f'{result["reranked_recall"]:.4f}'
        )

        print(
            "Result:",
            result["status"]
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    improved = sum(
        1
        for result in diagnostic_results
        if result["status"] == "IMPROVED"
    )

    worse = sum(
        1
        for result in diagnostic_results
        if result["status"] == "WORSE"
    )

    same = sum(
        1
        for result in diagnostic_results
        if result["status"] == "SAME"
    )

    print(
        "\n\n" + "=" * 70
    )

    print(
        "RERANKER SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Improved:",
        improved
    )

    print(
        "Worse:",
        worse
    )

    print(
        "Same:",
        same
    )

    print(
        "\nEvaluation completed."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()