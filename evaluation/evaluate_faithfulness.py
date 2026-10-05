import os
import sys


# ==========================================================
# Add project root to Python path
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# ==========================================================
# Imports
# ==========================================================

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.rag_pipeline import RAGPipeline


# ==========================================================
# Configuration
# ==========================================================

PDF_PATH = "data/documents/Rahul.pdf"


# ==========================================================
# Ground Truth Dataset
# ==========================================================

EVALUATION_DATASET = [

    {
        "question": "What is the candidate's name?",

        "expected_answer": "Rahul Ray",

        "relevant_pages": [1]
    },

    {
        "question":
            "What was the first DSA question asked?",

        "expected_answer":
            "Longest Substring Without Repeating Characters",

        "relevant_pages": [2]
    },

    {
        "question":
            "What type of round was Round 1?",

        "expected_answer":
            "DSA and project",

        "relevant_pages": [2]
    },

    {
        "question":
            "What was the candidate asked about his RAG project?",

        "expected_answer":
            "The prompt used in the RAG application",

        "relevant_pages": [2]
    },

    {
        "question":
            "What was the interview date?",

        "expected_answer":
            "29/09/2026",

        "relevant_pages": [2]
    }

]


# ==========================================================
# Normalize text
# ==========================================================

def normalize(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = text.replace(
        "-",
        " "
    )

    text = text.replace(
        "/",
        " "
    )

    text = " ".join(
        text.split()
    )

    return text


# ==========================================================
# Extract metadata
# ==========================================================

def get_metadata(result):

    chunk = result.get(
        "chunk",
        {}
    )

    return chunk.get(
        "metadata",
        {}
    )


# ==========================================================
# Extract chunk text
# ==========================================================

def get_text(result):

    chunk = result.get(
        "chunk",
        {}
    )

    return chunk.get(
        "text",
        ""
    )


# ==========================================================
# Check whether expected answer exists in context
# ==========================================================

def expected_answer_supported(
    context,
    expected_answer
):

    context_normalized = normalize(
        context
    )

    expected_normalized = normalize(
        expected_answer
    )

    return (
        expected_normalized
        in
        context_normalized
    )


# ==========================================================
# Simple faithfulness check
# ==========================================================

def evaluate_faithfulness(
    answer,
    context,
    expected_answer
):

    # ------------------------------------------------------
    # Basic deterministic rule:
    #
    # If the expected answer exists in the retrieved
    # context, and the generated answer contains it,
    # consider the answer supported.
    # ------------------------------------------------------

    answer_normalized = normalize(
        answer
    )

    expected_normalized = normalize(
        expected_answer
    )

    context_normalized = normalize(
        context
    )


    answer_contains_expected = (
        expected_normalized
        in
        answer_normalized
    )


    context_contains_expected = (
        expected_normalized
        in
        context_normalized
    )


    faithful = (
        answer_contains_expected
        and
        context_contains_expected
    )


    return {
        "faithful": faithful,

        "answer_contains_expected":
            answer_contains_expected,

        "context_contains_expected":
            context_contains_expected
    }


# ==========================================================
# Citation evaluation
# ==========================================================

def evaluate_citations(
    results,
    relevant_pages
):

    relevant_pages = set(
        relevant_pages
    )

    correct = 0
    total = 0


    for result in results:

        metadata = get_metadata(
            result
        )

        page = metadata.get(
            "page"
        )

        if page is None:
            continue

        total += 1

        if page in relevant_pages:
            correct += 1


    if total == 0:

        accuracy = 0.0

    else:

        accuracy = (
            correct / total
        )


    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


# ==========================================================
# Main
# ==========================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "FAITHFULNESS / HALLUCINATION EVALUATION"
    )

    print(
        "=" * 70
    )


    # ======================================================
    # Load PDF
    # ======================================================

    print(
        "\nLoading PDF..."
    )

    documents = load_pdf(
        PDF_PATH
    )

    print(
        f"Loaded {len(documents)} pages."
    )


    # ======================================================
    # Chunk documents
    # ======================================================

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    # ======================================================
    # RAG pipeline
    # ======================================================

    rag = RAGPipeline(
        chunks
    )


    # ======================================================
    # Metric storage
    # ======================================================

    faithfulness_scores = []

    citation_scores = []

    answer_correctness_scores = []


    # ======================================================
    # Evaluate each question
    # ======================================================

    for number, item in enumerate(
        EVALUATION_DATASET,
        start=1
    ):

        question = item[
            "question"
        ]

        expected_answer = item[
            "expected_answer"
        ]

        relevant_pages = item[
            "relevant_pages"
        ]


        print(
            "\n" + "=" * 70
        )

        print(
            f"QUESTION {number}"
        )

        print(
            "=" * 70
        )

        print(
            "Question:",
            question
        )

        print(
            "Expected:",
            expected_answer
        )


        # ==================================================
        # Run RAG
        # ==================================================

        result = rag.ask(
            question=question,
            chat_history=[]
        )


        answer = result.get(
            "answer",
            ""
        )

        results = result.get(
            "results",
            []
        )


        # ==================================================
        # Build retrieved context
        # ==================================================

        context_parts = []

        for retrieved in results:

            text = get_text(
                retrieved
            )

            if text:

                context_parts.append(
                    text
                )


        context = "\n\n".join(
            context_parts
        )


        # ==================================================
        # Answer correctness
        # ==================================================

        answer_correct = (
            normalize(expected_answer)
            in
            normalize(answer)
        )


        # ==================================================
        # Faithfulness
        # ==================================================

        faithfulness = evaluate_faithfulness(
            answer,
            context,
            expected_answer
        )


        # ==================================================
        # Citation
        # ==================================================

        citation = evaluate_citations(
            results,
            relevant_pages
        )


        # ==================================================
        # Print results
        # ==================================================

        print(
            "\nGenerated answer:"
        )

        print(
            answer
        )


        print(
            "\nRetrieved pages:"
        )

        retrieved_pages = []

        for retrieved in results:

            metadata = get_metadata(
                retrieved
            )

            retrieved_pages.append(
                metadata.get(
                    "page"
                )
            )

        print(
            retrieved_pages
        )


        print(
            "\nAnswer correct:",
            answer_correct
        )


        print(
            "Expected answer in context:",
            faithfulness[
                "context_contains_expected"
            ]
        )


        print(
            "Expected answer in generated answer:",
            faithfulness[
                "answer_contains_expected"
            ]
        )


        print(
            "Faithful:",
            faithfulness[
                "faithful"
            ]
        )


        print(
            "Citation accuracy:",
            round(
                citation["accuracy"],
                4
            )
        )


        # ==================================================
        # Store metrics
        # ==================================================

        answer_correctness_scores.append(
            int(answer_correct)
        )

        faithfulness_scores.append(
            int(
                faithfulness[
                    "faithful"
                ]
            )
        )

        citation_scores.append(
            citation["accuracy"]
        )


    # ======================================================
    # Final metrics
    # ======================================================

    total = len(
        EVALUATION_DATASET
    )


    answer_accuracy = (
        sum(answer_correctness_scores)
        / total
        if total
        else 0.0
    )


    faithfulness = (
        sum(faithfulness_scores)
        / total
        if total
        else 0.0
    )


    citation_accuracy = (
        sum(citation_scores)
        / total
        if total
        else 0.0
    )


    # ======================================================
    # Final report
    # ======================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "FINAL FAITHFULNESS EVALUATION"
    )

    print(
        "=" * 70
    )


    print(
        f"\nNumber of questions: "
        f"{total}"
    )


    print(
        f"Answer Accuracy: "
        f"{answer_accuracy:.4f}"
    )


    print(
        f"Faithfulness: "
        f"{faithfulness:.4f}"
    )


    print(
        f"Citation Accuracy: "
        f"{citation_accuracy:.4f}"
    )


    print(
        "\n" + "-" * 70
    )

    print(
        "INTERPRETATION"
    )

    print(
        "-" * 70
    )


    print(
        "\nAnswer Accuracy:"
    )

    print(
        "Whether the generated answer contains "
        "the expected information."
    )


    print(
        "\nFaithfulness:"
    )

    print(
        "Whether the expected information is present "
        "in the retrieved context and the generated answer."
    )


    print(
        "\nCitation Accuracy:"
    )

    print(
        "Whether retrieved source pages match the "
        "ground-truth relevant pages."
    )


    print(
        "\nEvaluation completed."
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()