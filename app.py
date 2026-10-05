import os
import tempfile
import shutil
import json
import re

import streamlit as st

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from src.loader import load_pdf
from src.chunker import chunk_documents
from src.rag_pipeline import RAGPipeline


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Question Answering",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("RAG-Based PDF Question Answering")

st.markdown(
    """
    Upload multiple PDFs → Process them together → Select a
    document or search across all documents → Retrieve → Rerank
    → Generate → Cite → Evaluate
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:
    st.session_state.rag = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

if "question" not in st.session_state:
    st.session_state.question = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "index_directory" not in st.session_state:
    st.session_state.index_directory = None

if "source_filter" not in st.session_state:
    st.session_state.source_filter = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cleanup_old_index():

    """
    Remove the previous temporary FAISS index directory.
    """

    old_directory = (
        st.session_state.get(
            "index_directory"
        )
    )

    if old_directory and os.path.exists(
        old_directory
    ):

        try:

            shutil.rmtree(
                old_directory
            )

        except Exception:

            pass

    st.session_state.index_directory = None


# ============================================================
# LOAD MULTIPLE PDFs
# ============================================================

def build_multi_pdf_documents(
    uploaded_files
):

    """
    Save uploaded PDFs temporarily, load them, and preserve
    the original uploaded filename in metadata.

    Returns:
        all_documents
        temp_files
    """

    all_documents = []
    temp_files = []

    for uploaded_file in uploaded_files:

        # ----------------------------------------------------
        # Create temporary PDF file
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        )

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_file.close()

        temp_pdf_path = temp_file.name

        temp_files.append(
            temp_pdf_path
        )

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        documents = load_pdf(
            temp_pdf_path
        )

        # ----------------------------------------------------
        # Preserve original filename
        # ----------------------------------------------------

        for document in documents:

            metadata = document.get(
                "metadata",
                {}
            )

            metadata["source"] = (
                uploaded_file.name
            )

            metadata["filename"] = (
                uploaded_file.name
            )

            document["metadata"] = (
                metadata
            )

        all_documents.extend(
            documents
        )

    return (
        all_documents,
        temp_files
    )


# ============================================================
# CREATE MULTI-PDF CHUNKS
# ============================================================

def build_multi_pdf_chunks(
    documents,
    chunk_size,
    chunk_overlap
):

    """
    Chunk all documents together.

    Adds a globally unique chunk ID.
    """

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # --------------------------------------------------------
    # Add globally unique chunk IDs
    # --------------------------------------------------------

    for global_id, chunk in enumerate(
        chunks
    ):

        metadata = chunk.get(
            "metadata",
            {}
        )

        metadata["global_chunk_id"] = (
            global_id
        )

        chunk["metadata"] = (
            metadata
        )

    return chunks


# ============================================================
# UNIQUE CITATIONS
# ============================================================

def select_unique_citations(
    retrieved_results,
    max_citations=3
):

    """
    Select unique (source, page) citations.

    For multiple PDFs:

        Rahul.pdf Page 1

    and

        Interview.pdf Page 1

    are different citations.
    """

    citations = []

    seen = set()

    for result in retrieved_results:

        chunk = result.get(
            "chunk",
            {}
        )

        metadata = chunk.get(
            "metadata",
            {}
        )

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        key = (
            source,
            page
        )

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            {
                "source": source,

                "page": page,

                "distance": result.get(
                    "distance"
                ),

                "rerank_score": result.get(
                    "rerank_score"
                )
            }
        )

        if len(citations) >= max_citations:
            break

    return citations


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Configuration")

    # --------------------------------------------------------
    # Chunk Size
    # --------------------------------------------------------

    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=50
    )

    # --------------------------------------------------------
    # Chunk Overlap
    # --------------------------------------------------------

    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=10
    )

    # --------------------------------------------------------
    # FAISS Retrieval K
    # --------------------------------------------------------

    retrieval_k = st.number_input(
        "FAISS Retrieval K",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )

    # --------------------------------------------------------
    # Final Reranked K
    # --------------------------------------------------------

    rerank_k = st.number_input(
        "Final Reranked K",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )

    st.divider()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    st.markdown("### Models")

    st.write(
        "**Embedding:** "
        "`text-embedding-3-small`"
    )

    st.write(
        "**Generator:** "
        "`gpt-4.1-mini`"
    )

    st.write(
        "**Vector Store:** "
        "`FAISS IndexFlatL2`"
    )

    st.write(
        "**Reranker:** "
        "`Cross-Encoder`"
    )


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("1. Upload PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# SHOW SELECTED FILES
# ============================================================

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} PDF(s) selected."
    )

    with st.expander(
        "View selected PDFs",
        expanded=True
    ):

        for i, uploaded_file in enumerate(
            uploaded_files,
            start=1
        ):

            st.write(
                f"{i}. **{uploaded_file.name}** "
                f"({uploaded_file.size / 1024:.1f} KB)"
            )


# ============================================================
# PROCESS PDFs
# ============================================================

if uploaded_files:

    if st.button(
        "Process PDFs",
        type="primary"
    ):

        with st.spinner(
            "Processing all PDFs..."
        ):

            try:

                # ==================================================
                # CLEAN OLD INDEX
                # ==================================================

                cleanup_old_index()

                # ==================================================
                # LOAD ALL PDFs
                # ==================================================

                documents, temp_files = (
                    build_multi_pdf_documents(
                        uploaded_files
                    )
                )

                if not documents:

                    st.error(
                        "No readable content was found "
                        "in the uploaded PDFs."
                    )

                    st.stop()

                # ==================================================
                # CHUNK ALL PDFs
                # ==================================================

                chunks = build_multi_pdf_chunks(
                    documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                if not chunks:

                    st.error(
                        "No chunks were created."
                    )

                    st.stop()

                # ==================================================
                # CREATE NEW FAISS INDEX
                # ==================================================

                index_directory = tempfile.mkdtemp(
                    prefix="rag_multi_pdf_"
                )

                index_path = os.path.join(
                    index_directory,
                    "index.faiss"
                )

                # ==================================================
                # CREATE RAG PIPELINE
                # ==================================================

                rag = RAGPipeline(
                    chunks,
                    vector_store_path=index_path
                )

                # ==================================================
                # SAVE SESSION STATE
                # ==================================================

                st.session_state.rag = rag

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.documents = (
                    documents
                )

                st.session_state.uploaded_filenames = [
                    uploaded_file.name
                    for uploaded_file
                    in uploaded_files
                ]

                st.session_state.result = None

                st.session_state.question = ""

                st.session_state.chat_history = []

                st.session_state.evaluation = None

                st.session_state.source_filter = None

                st.session_state.processing_complete = True

                st.session_state.index_directory = (
                    index_directory
                )

                # ==================================================
                # DELETE TEMPORARY PDF FILES
                # ==================================================

                for temp_pdf_path in temp_files:

                    try:

                        os.remove(
                            temp_pdf_path
                        )

                    except OSError:

                        pass

                # ==================================================
                # SUCCESS
                # ==================================================

                st.success(
                    "All PDFs were processed successfully."
                )

                # ==================================================
                # PROCESSING STATISTICS
                # ==================================================

                col1, col2, col3, col4 = (
                    st.columns(4)
                )

                with col1:

                    st.metric(
                        "PDFs",
                        len(uploaded_files)
                    )

                with col2:

                    st.metric(
                        "Pages",
                        len(documents)
                    )

                with col3:

                    st.metric(
                        "Chunks",
                        len(chunks)
                    )

                with col4:

                    st.metric(
                        "Chunk Size",
                        chunk_size
                    )

            except Exception as e:

                st.error(
                    f"Error while processing PDFs: {e}"
                )

                st.exception(e)


# ============================================================
# STOP IF PDFs HAVE NOT BEEN PROCESSED
# ============================================================

if not st.session_state.processing_complete:

    st.info(
        "Upload one or more PDFs and click "
        "**Process PDFs** to begin."
    )

    st.stop()


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

st.divider()

st.header("Document Information")


# ============================================================
# DISPLAY UPLOADED DOCUMENTS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Number of PDFs",
        len(
            st.session_state.uploaded_filenames
        )
    )

with col2:

    st.metric(
        "Total Pages",
        len(
            st.session_state.documents
        )
    )


with st.expander(
    "Uploaded Documents",
    expanded=True
):

    for i, filename in enumerate(
        st.session_state.uploaded_filenames,
        start=1
    ):

        st.write(
            f"{i}. **{filename}**"
        )


st.write(
    f"**Total Chunks:** "
    f"{len(st.session_state.chunks)}"
)


# ============================================================
# DOCUMENT SELECTOR
# ============================================================

st.subheader("Search Scope")

document_options = [
    "All Documents"
] + st.session_state.uploaded_filenames

selected_document = st.selectbox(
    "Search in",
    document_options
)

if selected_document == "All Documents":

    source_filter = None

else:

    source_filter = selected_document


st.session_state.source_filter = (
    source_filter
)


# ============================================================
# DISPLAY CURRENT SEARCH SCOPE
# ============================================================

if source_filter is None:

    st.info(
        "Current scope: **All uploaded documents**"
    )

else:

    st.info(
        f"Current scope: **{source_filter}**"
    )


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.header("2. Ask a Question")

question = st.text_input(
    "Enter your question",
    value=st.session_state.question,
    placeholder=(
        "Ask something about one or all "
        "uploaded PDFs..."
    )
)


# ============================================================
# ASK QUESTION
# ============================================================

if st.button(
    "Ask Question",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Running RAG pipeline..."
        ):

            try:

                # ==================================================
                # RUN RAG
                # ==================================================

                result = (
                    st.session_state.rag.ask(
                        question=question,

                        chat_history=(
                            st.session_state.chat_history
                        ),

                        source_filter=source_filter
                    )
                )

                # ==================================================
                # REBUILD CITATIONS
                # ==================================================

                retrieved_results = result.get(
                    "results",
                    []
                )

                result["sources"] = (
                    select_unique_citations(
                        retrieved_results,
                        max_citations=3
                    )
                )

                # ==================================================
                # SAVE RESULT
                # ==================================================

                st.session_state.question = (
                    question
                )

                st.session_state.result = (
                    result
                )

                # ==================================================
                # UPDATE CHAT HISTORY
                # ==================================================

                st.session_state.chat_history.append(
                    f"User: {question}"
                )

                st.session_state.chat_history.append(
                    f"Assistant: "
                    f"{result.get('answer', '')}"
                )

            except Exception as e:

                st.error(
                    f"Error while asking question: {e}"
                )

                st.exception(e)


# ============================================================
# DISPLAY RESULT
# ============================================================

result = st.session_state.result


if result is not None:

    # ========================================================
    # ANSWER
    # ========================================================

    st.divider()

    st.header("3. Answer")

    answer = result.get(
        "answer",
        ""
    )

    st.markdown(answer)


    # ========================================================
    # ORIGINAL QUERY
    # ========================================================

    with st.expander(
        "Original Query"
    ):

        st.write(
            st.session_state.question
        )


    # ========================================================
    # REWRITTEN QUERY
    # ========================================================

    rewritten_query = result.get(
        "rewritten_query",
        ""
    )

    with st.expander(
        "Retrieved Query",
        expanded=True
    ):

        st.write(
            rewritten_query
        )


    # ========================================================
    # SEARCH SCOPE
    # ========================================================

    with st.expander(
        "Search Scope"
    ):

        if source_filter is None:

            st.write(
                "All Documents"
            )

        else:

            st.write(
                source_filter
            )


    # ========================================================
    # CITATIONS
    # ========================================================

    st.divider()

    st.header("4. Citations")

    sources = result.get(
        "sources",
        []
    )

    if sources:

        for i, source in enumerate(
            sources,
            start=1
        ):

            filename = source.get(
                "source",
                "Unknown"
            )

            page = source.get(
                "page",
                "Unknown"
            )

            distance = source.get(
                "distance"
            )

            rerank_score = source.get(
                "rerank_score"
            )

            st.markdown(
                f"### [{i}] {filename} — Page {page}"
            )

            citation_col1, citation_col2 = (
                st.columns(2)
            )

            with citation_col1:

                if distance is not None:

                    st.metric(
                        "FAISS L2 Distance",
                        f"{distance:.4f}"
                    )

            with citation_col2:

                if rerank_score is not None:

                    st.metric(
                        "Cross-Encoder Score",
                        f"{rerank_score:.4f}"
                    )

    else:

        st.info(
            "No citations were returned."
        )


    # ========================================================
    # RETRIEVED RESULTS
    # ========================================================

    st.divider()

    st.header("5. Retrieved Evidence")

    retrieved_results = result.get(
        "results",
        []
    )

    if retrieved_results:

        for rank, retrieved in enumerate(
            retrieved_results,
            start=1
        ):

            chunk = retrieved.get(
                "chunk",
                {}
            )

            metadata = chunk.get(
                "metadata",
                {}
            )

            text = chunk.get(
                "text",
                ""
            )

            page = metadata.get(
                "page",
                "Unknown"
            )

            filename = metadata.get(
                "source",
                "Unknown"
            )

            distance = retrieved.get(
                "distance"
            )

            rerank_score = retrieved.get(
                "rerank_score"
            )

            with st.expander(
                f"Rank {rank} | "
                f"{filename} | "
                f"Page {page}"
            ):

                st.markdown(
                    "**Source:** "
                    f"`{filename}`"
                )

                st.markdown(
                    f"**Page:** `{page}`"
                )

                st.markdown(
                    "**Chunk:**"
                )

                st.write(
                    text
                )

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    if distance is not None:

                        st.metric(
                            "FAISS L2 Distance",
                            f"{distance:.4f}"
                        )

                with col2:

                    if rerank_score is not None:

                        st.metric(
                            "Cross-Encoder Score",
                            f"{rerank_score:.4f}"
                        )

    else:

        st.info(
            "No retrieved chunks."
        )


    # ========================================================
    # CONTEXT
    # ========================================================

    st.divider()

    st.header("6. Retrieved Context")

    context_parts = []

    for retrieved in retrieved_results:

        chunk = retrieved.get(
            "chunk",
            {}
        )

        metadata = chunk.get(
            "metadata",
            {}
        )

        text = chunk.get(
            "text",
            ""
        )

        filename = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            "Unknown"
        )

        if text:

            context_parts.append(
                f"[Source: {filename} | Page: {page}]\n"
                f"{text}"
            )

    context = "\n\n".join(
        context_parts
    )

    with st.expander(
        "Show context sent to the LLM"
    ):

        st.text(
            context
        )


    # ========================================================
    # RAG PIPELINE TRACE
    # ========================================================

    st.divider()

    st.header("7. RAG Pipeline")

    st.markdown(
        """
        **User Question**

        ↓

        **Query Rewriting**

        ↓

        **Embedding — `text-embedding-3-small`**

        ↓

        **FAISS Retrieval**

        ↓

        **Source Filtering**

        ↓

        **Top-K Candidates**

        ↓

        **Cross-Encoder Reranking**

        ↓

        **Top-K Relevant Chunks**

        ↓

        **Context Construction**

        ↓

        **LLM — `gpt-4.1-mini`**

        ↓

        **Answer + Multi-Document Citations**
        """
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    st.divider()

    st.header("8. Answer Evaluation")

    st.markdown(
        """
        Provide a reference answer to evaluate the generated
        response.

        **Semantic Similarity** compares the generated answer
        with the reference answer using embeddings.

        **LLM-as-a-Judge** evaluates correctness, relevance,
        and faithfulness on a 1–5 scale.
        """
    )

    reference_answer = st.text_area(
        "Reference Answer",
        placeholder=(
            "Enter the expected/reference answer..."
        ),
        height=120
    )

    if st.button(
        "Evaluate Answer"
    ):

        if not reference_answer.strip():

            st.warning(
                "Please provide a reference answer."
            )

        else:

            with st.spinner(
                "Evaluating answer..."
            ):

                try:

                    # ==================================================
                    # 1. EMBEDDING-BASED SEMANTIC SIMILARITY
                    # ==================================================

                    embedding_model = (
                        OpenAIEmbeddings(
                            model="text-embedding-3-small"
                        )
                    )

                    evaluation_embeddings = (
                        embedding_model.embed_documents(
                            [
                                reference_answer,
                                answer
                            ]
                        )
                    )

                    vector_a = (
                        evaluation_embeddings[0]
                    )

                    vector_b = (
                        evaluation_embeddings[1]
                    )

                    dot_product = sum(
                        x * y
                        for x, y in zip(
                            vector_a,
                            vector_b
                        )
                    )

                    norm_a = (
                        sum(
                            x * x
                            for x in vector_a
                        )
                    ) ** 0.5

                    norm_b = (
                        sum(
                            x * x
                            for x in vector_b
                        )
                    ) ** 0.5

                    if (
                        norm_a == 0
                        or norm_b == 0
                    ):

                        semantic_similarity = 0.0

                    else:

                        semantic_similarity = (
                            dot_product
                            / (
                                norm_a
                                * norm_b
                            )
                        )


                    # ==================================================
                    # 2. LLM-AS-A-JUDGE
                    # ==================================================

                    judge_llm = ChatOpenAI(
                        model="gpt-4.1-mini",
                        temperature=0
                    )

                    judge_prompt = f"""
You are evaluating the quality of an answer produced by
a Retrieval-Augmented Generation (RAG) system.

Evaluate the generated answer against the reference answer
and the supplied context.

Do NOT use outside knowledge.

QUESTION:
{st.session_state.question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{answer}

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

IMPORTANT:

5 = Completely correct / highly relevant / fully supported
4 = Mostly correct with minor issues
3 = Partially correct
2 = Mostly incorrect
1 = Completely incorrect

The reference answer is the expected answer to the question.

Do not give a high correctness score merely because the generated
answer is related to the retrieved context.

Return ONLY valid JSON in exactly this format:

{{
    "correctness": 1,
    "relevance": 1,
    "faithfulness": 1,
    "reason": "brief explanation"
}}
"""

                    judge_response = (
                        judge_llm.invoke(
                            judge_prompt
                        )
                    )

                    judge_content = (
                        judge_response.content
                        .strip()
                    )

                    judge_content = re.sub(
                        r"```json|```",
                        "",
                        judge_content
                    ).strip()

                    try:

                        judge_result = (
                            json.loads(
                                judge_content
                            )
                        )

                    except json.JSONDecodeError:

                        judge_result = {

                            "correctness": 0,

                            "relevance": 0,

                            "faithfulness": 0,

                            "reason": (
                                "Judge returned "
                                "invalid JSON."
                            )
                        }

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


                    # ==================================================
                    # DISPLAY SEMANTIC SIMILARITY
                    # ==================================================

                    st.subheader(
                        "Semantic Similarity"
                    )

                    st.metric(
                        "Similarity",
                        f"{semantic_similarity:.4f}"
                    )

                    st.progress(
                        min(
                            max(
                                float(
                                    semantic_similarity
                                ),
                                0.0
                            ),
                            1.0
                        )
                    )


                    # ==================================================
                    # DISPLAY LLM JUDGE
                    # ==================================================

                    st.subheader(
                        "LLM-as-a-Judge"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:

                        st.metric(
                            "Correctness",
                            f"{correctness:.1f} / 5"
                        )

                    with col2:

                        st.metric(
                            "Relevance",
                            f"{relevance:.1f} / 5"
                        )

                    with col3:

                        st.metric(
                            "Faithfulness",
                            f"{faithfulness:.1f} / 5"
                        )


                    # ==================================================
                    # NORMALIZED SCORES
                    # ==================================================

                    st.subheader(
                        "Normalized Scores"
                    )

                    col1, col2, col3 = (
                        st.columns(3)
                    )

                    with col1:

                        st.metric(
                            "Correctness",
                            f"{correctness / 5 * 100:.1f}%"
                        )

                    with col2:

                        st.metric(
                            "Relevance",
                            f"{relevance / 5 * 100:.1f}%"
                        )

                    with col3:

                        st.metric(
                            "Faithfulness",
                            f"{faithfulness / 5 * 100:.1f}%"
                        )


                    # ==================================================
                    # JUDGE EXPLANATION
                    # ==================================================

                    st.subheader(
                        "Judge Explanation"
                    )

                    st.write(
                        judge_result.get(
                            "reason",
                            ""
                        )
                    )


                    # ==================================================
                    # STORE EVALUATION
                    # ==================================================

                    st.session_state.evaluation = {

                        "semantic_similarity":
                            semantic_similarity,

                        "correctness":
                            correctness,

                        "relevance":
                            relevance,

                        "faithfulness":
                            faithfulness,

                        "reason":
                            judge_result.get(
                                "reason",
                                ""
                            )
                    }

                except Exception as e:

                    st.error(
                        f"Evaluation failed: {e}"
                    )

                    st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "RAG PDF Question Answering | "
    "Multi-PDF FAISS Retrieval + "
    "text-embedding-3-small + "
    "Cross-Encoder Reranking + "
    "GPT-4.1-mini"
)