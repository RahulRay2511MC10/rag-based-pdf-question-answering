import json

from src.rag_pipeline import RAGPipeline
from src.embeddings import get_embedding_model
from evaluation.metrics import cosine_similarity


PDF_PATH = "data/documents/Rahul.pdf"


# ==========================================
# Load Evaluation Dataset
# ==========================================

with open(
    "evaluation/generation_questions.json",
    "r"
) as f:

    questions = json.load(f)


# ==========================================
# Create RAG Pipeline
# ==========================================

rag = RAGPipeline(
    PDF_PATH
)


# ==========================================
# Embedding Model
# ==========================================

embedding_model = (
    get_embedding_model()
)


# ==========================================
# Store Scores
# ==========================================

similarity_scores = []


# ==========================================
# Evaluate Each Question
# ==========================================

for item in questions:

    question = item["question"]

    ground_truth = (
        item["ground_truth"]
    )

    print("\n")
    print("=" * 70)

    print(
        f"Question: {question}"
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
    # Retrieved Pages
    # ======================================

    pages = [
        result["chunk"]["metadata"]["page"]
        for result in results
    ]

    # ======================================
    # Generate Embeddings
    # ======================================

    answer_vector = (
        embedding_model.embed_query(
            answer
        )
    )

    ground_truth_vector = (
        embedding_model.embed_query(
            ground_truth
        )
    )

    # ======================================
    # Calculate Similarity
    # ======================================

    score = cosine_similarity(
        answer_vector,
        ground_truth_vector
    )

    similarity_scores.append(
        score
    )

    # ======================================
    # Display
    # ======================================

    print(
        f"Retrieved Pages: {pages}"
    )

    print(
        f"\nGenerated Answer:\n{answer}"
    )

    print(
        f"\nGround Truth:\n{ground_truth}"
    )

    print(
        f"\nSemantic Similarity: "
        f"{score:.4f}"
    )


# ==========================================
# Average Score
# ==========================================

if similarity_scores:

    average_score = (
        sum(similarity_scores)
        / len(similarity_scores)
    )

    print("\n")
    print("=" * 70)

    print(
        "GENERATION EVALUATION"
    )

    print("=" * 70)

    print(
        f"Average Semantic Similarity: "
        f"{average_score:.4f}"
    )

    print("=" * 70)