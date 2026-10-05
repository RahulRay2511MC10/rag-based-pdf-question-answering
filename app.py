from src.loader import load_pdf
from src.chunker import chunk_documents
from src.rag_pipeline import RAGPipeline


# ==========================================================
# Configuration
# ==========================================================

PDF_PATH = "data/documents/Rahul.pdf"


# ==========================================================
# Load PDF
# ==========================================================

print("\nLoading PDF...")

documents = load_pdf(
    PDF_PATH
)

print(
    f"Loaded {len(documents)} document pages."
)


# ==========================================================
# Create chunks
# ==========================================================

print("\nCreating chunks...")

chunks = chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

print(
    f"Created {len(chunks)} chunks."
)


# ==========================================================
# Debug chunk structure
# ==========================================================

if chunks:

    print("\nFirst chunk:")

    print(
        chunks[0]
    )

else:

    raise RuntimeError(
        "No chunks were created from the PDF."
    )


# ==========================================================
# Create RAG pipeline
# ==========================================================

rag = RAGPipeline(
    chunks
)


# ==========================================================
# Conversation history
# ==========================================================

chat_history = []


# ==========================================================
# Main loop
# ==========================================================

while True:

    question = input(
        "\nAsk a question about the document: "
    ).strip()


    # ------------------------------------------------------
    # Exit
    # ------------------------------------------------------

    if question.lower() in [
        "exit",
        "quit",
        "q"
    ]:

        print(
            "\nExiting..."
        )

        break


    # ------------------------------------------------------
    # Ignore empty questions
    # ------------------------------------------------------

    if not question:

        continue


    # ======================================================
    # Ask RAG pipeline
    # ======================================================

    result = rag.ask(
        question=question,
        chat_history=chat_history
    )


    # ======================================================
    # Display answer
    # ======================================================

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(
        result["answer"]
    )


    # ======================================================
    # Display sources
    # ======================================================

    print("\nSources:")

    for source in result.get(
        "sources",
        []
    ):

        print(
            f"- {source.get('source')} "
            f"(Page {source.get('page')})"
        )


    # ======================================================
    # Update conversation history
    # ======================================================

    chat_history.append(
        f"User: {question}"
    )

    chat_history.append(
        f"Assistant: {result['answer']}"
    )