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

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ==========================================================
# Imports
# ==========================================================

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.rag_pipeline import RAGPipeline


# ==========================================================
# Configuration
# ==========================================================

PDF_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "documents",
    "Rahul.pdf"
)


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

    text = str(text)

    text = text.lower()

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
# Answer matching
# ==========================================================

def answer_contains_expected(
    answer,
    expected
):

    answer_normalized = normalize(
        answer
    )

    expected_normalized = normalize(
        expected
    )

    return (
        expected_normalized
        in
        answer_normalized
    )


# ==========================================================
# Get page from selected citation
# ==========================================================

def get_page_from_source(source):

    if not isinstance(
        source,
        dict
    ):
        return None

    return source.get(
        "page"
    )


# ==========================================================
# Citation evaluation
# ==========================================================

def evaluate_citations(
    sources,
    relevant_pages
):

    # ------------------------------------------------------
    # Convert ground-truth pages to set
    # ------------------------------------------------------

    relevant_pages = set(
        relevant_pages
    )

    # ------------------------------------------------------
    # No sources
    # ------------------------------------------------------

    if not sources:

        return {
            "citation_accuracy": 0.0,
            "correct_citations": 0,
            "total_citations": 0,
            "retrieved_pages": [],
            "correct_pages": []
        }

    # ------------------------------------------------------
    # Track citations
    # ------------------------------------------------------

    correct = 0
    total = 0

    retrieved_pages = []
    correct_pages = []

    # ------------------------------------------------------
    # Evaluate selected citations
    # ------------------------------------------------------

    for source in sources:

        page = get_page_from_source(
            source
        )

        if page is None:
            continue

        total += 1

        retrieved_pages.append(
            page
        )

        if page in relevant_pages:

            correct += 1

            correct_pages.append(
                page
            )

    # ------------------------------------------------------
    # Calculate citation precision
    # ------------------------------------------------------

    if total == 0:

        accuracy = 0.0

    else:

        accuracy = (
            correct / total
        )

    return {

        "citation_accuracy":
            accuracy,

        "correct_citations":
            correct,

        "total_citations":
            total,

        "retrieved_pages":
            retrieved_pages,

        "correct_pages":
            correct_pages
    }


# ==========================================================
# Main evaluation
# ==========================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "END-TO-END ANSWER EVALUATION"
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

    print(
        "\nCreating chunks..."
    )

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # ======================================================
    # Create RAG pipeline
    # ======================================================

    print(
        "\nInitializing RAG pipeline..."
    )

    rag = RAGPipeline(
        chunks
    )

    # ======================================================
    # Metric storage
    # ======================================================

    answer_results = []

    citation_results = []

    # ======================================================
    # Evaluate every question
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
            "Expected answer:",
            expected_answer
        )

        print(
            "Relevant pages:",
            relevant_pages
        )

        # ==================================================
        # Run complete RAG pipeline
        # ==================================================

        result = rag.ask(
            question=question,
            chat_history=[]
        )

        # ==================================================
        # Extract answer
        # ==================================================

        answer = result.get(
            "answer",
            ""
        )

        # ==================================================
        # IMPORTANT:
        #
        # Use the NEW citation list generated by
        # RAGPipeline.select_citations()
        #
        # DO NOT use result["results"] here.
        # ==================================================

        sources = result.get(
            "sources",
            []
        )

        # ==================================================
        # Print generated answer
        # ==================================================

        print(
            "\nGenerated answer:"
        )

        print(
            answer
        )

        # ==================================================
        # Answer correctness
        # ==================================================

        answer_correct = (
            answer_contains_expected(
                answer,
                expected_answer
            )
        )

        print(
            "\nAnswer correct:",
            answer_correct
        )

        # ==================================================
        # Citation evaluation
        # ==================================================

        citation_metric = evaluate_citations(
            sources,
            relevant_pages
        )

        # ==================================================
        # Print selected citations
        # ==================================================

        print(
            "\nSelected citation pages:",
            citation_metric[
                "retrieved_pages"
            ]
        )

        print(
            "Correct citation pages:",
            citation_metric[
                "correct_pages"
            ]
        )

        print(
            "Citation accuracy:",
            round(
                citation_metric[
                    "citation_accuracy"
                ],
                4
            )
        )

        print(
            "Correct citations:",
            citation_metric[
                "correct_citations"
            ]
        )

        print(
            "Total citations:",
            citation_metric[
                "total_citations"
            ]
        )

        # ==================================================
        # Store metrics
        # ==================================================

        answer_results.append(
            int(answer_correct)
        )

        citation_results.append(
            citation_metric[
                "citation_accuracy"
            ]
        )

    # ======================================================
    # Final metrics
    # ======================================================

    total_questions = len(
        EVALUATION_DATASET
    )

    # ------------------------------------------------------
    # Answer accuracy
    # ------------------------------------------------------

    answer_accuracy = (

        sum(answer_results)
        /
        total_questions

        if total_questions
        else 0.0
    )

    # ------------------------------------------------------
    # Citation accuracy
    # ------------------------------------------------------

    citation_accuracy = (

        sum(citation_results)
        /
        total_questions

        if total_questions
        else 0.0
    )

    # ======================================================
    # Final report
    # ======================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "FINAL ANSWER EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nNumber of questions: "
        f"{total_questions}"
    )

    print(
        f"Answer Accuracy: "
        f"{answer_accuracy:.4f}"
    )

    print(
        f"Citation Accuracy: "
        f"{citation_accuracy:.4f}"
    )

    # ======================================================
    # Percentage
    # ======================================================

    print(
        f"\nAnswer Accuracy: "
        f"{answer_accuracy * 100:.2f}%"
    )

    print(
        f"Citation Accuracy: "
        f"{citation_accuracy * 100:.2f}%"
    )

    # ======================================================
    # Interpretation
    # ======================================================

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
        "Measures how many questions were "
        "answered with the expected information."
    )

    print(
        "\nCitation Accuracy:"
    )

    print(
        "Measures the precision of the pages "
        "selected as final citations."
    )

    print(
        "\nThe citation metric now evaluates "
        "result['sources'] rather than the "
        "raw retrieved chunks."
    )

    print(
        "\nEvaluation completed."
    )


# ==========================================================
# Entry point
# ==========================================================

if __name__ == "__main__":

    main()