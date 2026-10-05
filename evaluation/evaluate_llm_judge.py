import os
import sys


# ==========================================================
# Add project root
# ==========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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

from evaluation.llm_judge import LLMJudge


# ==========================================================
# Configuration
# ==========================================================

PDF_PATH = "data/documents/Rahul.pdf"


# ==========================================================
# Evaluation questions
# ==========================================================

EVALUATION_DATASET = [

    {
        "question":
            "What is the candidate's name?"
    },

    {
        "question":
            "What was the first DSA question asked?"
    },

    {
        "question":
            "What type of round was Round 1?"
    },

    {
        "question":
            "What was the candidate asked about his RAG project?"
    },

    {
        "question":
            "What was the interview date?"
    }

]


# ==========================================================
# Extract text from retrieved result
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
# Main
# ==========================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "LLM-AS-A-JUDGE EVALUATION"
    )

    print(
        "=" * 70
    )


    # ======================================================
    # Load document
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
    # Chunk document
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
    # Create RAG pipeline
    # ======================================================

    rag = RAGPipeline(
        chunks
    )


    # ======================================================
    # Create evaluator
    # ======================================================

    judge = LLMJudge()


    # ======================================================
    # Metric storage
    # ======================================================

    relevance_scores = []

    faithfulness_scores = []

    context_scores = []


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
            question
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
        # Build context
        # ==================================================

        context_parts = []


        for retrieved_result in results:

            text = get_text(
                retrieved_result
            )

            if text:

                context_parts.append(
                    text
                )


        context = "\n\n".join(
            context_parts
        )


        # ==================================================
        # Display answer
        # ==================================================

        print(
            "\nGENERATED ANSWER"
        )

        print(
            "-" * 50
        )

        print(
            answer
        )


        # ==================================================
        # Run LLM judge
        # ==================================================

        evaluation = judge.evaluate(
            question=question,
            context=context,
            answer=answer
        )


        # ==================================================
        # Extract scores
        # ==================================================

        answer_relevance = evaluation.get(
            "answer_relevance",
            0
        )

        faithfulness = evaluation.get(
            "faithfulness",
            0
        )

        context_relevance = evaluation.get(
            "context_relevance",
            0
        )


        # ==================================================
        # Store scores
        # ==================================================

        relevance_scores.append(
            answer_relevance
        )

        faithfulness_scores.append(
            faithfulness
        )

        context_scores.append(
            context_relevance
        )


        # ==================================================
        # Print evaluation
        # ==================================================

        print(
            "\nLLM JUDGE"
        )

        print(
            "-" * 50
        )

        print(
            "Answer Relevance:",
            answer_relevance,
            "/ 5"
        )

        print(
            "Faithfulness:",
            faithfulness,
            "/ 5"
        )

        print(
            "Context Relevance:",
            context_relevance,
            "/ 5"
        )

        print(
            "Reason:",
            evaluation.get(
                "reason",
                ""
            )
        )


    # ======================================================
    # Final averages
    # ======================================================

    total = len(
        EVALUATION_DATASET
    )


    avg_relevance = (
        sum(relevance_scores)
        / total
        if total
        else 0
    )


    avg_faithfulness = (
        sum(faithfulness_scores)
        / total
        if total
        else 0
    )


    avg_context = (
        sum(context_scores)
        / total
        if total
        else 0
    )


    # ======================================================
    # Final report
    # ======================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "FINAL LLM-AS-A-JUDGE EVALUATION"
    )

    print(
        "=" * 70
    )


    print(
        f"\nNumber of questions: {total}"
    )


    print(
        f"Average Answer Relevance: "
        f"{avg_relevance:.2f} / 5"
    )


    print(
        f"Average Faithfulness: "
        f"{avg_faithfulness:.2f} / 5"
    )


    print(
        f"Average Context Relevance: "
        f"{avg_context:.2f} / 5"
    )


    # ======================================================
    # Convert to percentage
    # ======================================================

    relevance_percentage = (
        avg_relevance / 5
    ) * 100


    faithfulness_percentage = (
        avg_faithfulness / 5
    ) * 100


    context_percentage = (
        avg_context / 5
    ) * 100


    print(
        "\n" + "-" * 70
    )

    print(
        "NORMALIZED SCORES"
    )

    print(
        "-" * 70
    )


    print(
        f"\nAnswer Relevance: "
        f"{relevance_percentage:.2f}%"
    )


    print(
        f"Faithfulness: "
        f"{faithfulness_percentage:.2f}%"
    )


    print(
        f"Context Relevance: "
        f"{context_percentage:.2f}%"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "Evaluation completed."
    )

    print(
        "=" * 70
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()