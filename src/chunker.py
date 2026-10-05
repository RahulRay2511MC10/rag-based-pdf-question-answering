from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Split documents into smaller chunks.

    Parameters:
        documents: List of documents returned by load_pdf()
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters shared between chunks

    Returns:
        List of chunks containing text and metadata.
    """

    # --------------------------------------------------
    # 1. Validate chunk parameters
    # --------------------------------------------------

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )


    # --------------------------------------------------
    # 2. Create text splitter
    # --------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


    # --------------------------------------------------
    # 3. Store all chunks
    # --------------------------------------------------

    chunks = []


    # Global chunk ID
    global_chunk_id = 0


    # --------------------------------------------------
    # 4. Process every document
    # --------------------------------------------------

    for document in documents:

        # Get the document text
        text = document["text"]

        # Get original metadata
        metadata = document["metadata"]


        # --------------------------------------------------
        # 5. Split document text
        # --------------------------------------------------

        split_texts = splitter.split_text(
            text
        )


        # --------------------------------------------------
        # 6. Create chunk objects
        # --------------------------------------------------

        for local_chunk_id, chunk_text in enumerate(
            split_texts
        ):

            chunk = {
                "text": chunk_text,

                "metadata": {
                    **metadata,

                    # ID within the page/document
                    "chunk_id": local_chunk_id,

                    # Globally unique ID
                    "global_chunk_id": global_chunk_id
                }
            }


            # Add chunk
            chunks.append(chunk)


            # Increment global ID
            global_chunk_id += 1


    # --------------------------------------------------
    # 7. Return chunks
    # --------------------------------------------------

    return chunks