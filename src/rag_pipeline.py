import os

from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.query_rewriter import QueryRewriter
from src.reranker import Reranker
from src.generator import Generator


class RAGPipeline:

    def __init__(
        self,
        chunks,
        vector_store_path=None
    ):

        """
        Initialize the RAG pipeline.

        Parameters
        ----------
        chunks : list
            Chunked documents with text and metadata.

        vector_store_path : str or None
            If a path is provided and the FAISS index exists,
            load that index.

            Otherwise create a fresh FAISS index.
        """

        # ======================================================
        # STORE CHUNKS
        # ======================================================

        self.chunks = chunks

        if not self.chunks:

            raise ValueError(
                "No chunks were provided."
            )

        # ======================================================
        # 1. EMBEDDING MODEL
        # ======================================================

        self.embedding_model = (
            get_embedding_model()
        )

        # ======================================================
        # 2. QUERY REWRITER
        # ======================================================

        self.query_rewriter = (
            QueryRewriter()
        )

        # ======================================================
        # 3. CROSS-ENCODER RERANKER
        # ======================================================

        self.reranker = (
            Reranker()
        )

        # ======================================================
        # 4. LLM GENERATOR
        # ======================================================

        self.generator = (
            Generator()
        )

        # ======================================================
        # 5. VECTOR STORE
        # ======================================================

        if (
            vector_store_path
            and os.path.exists(
                vector_store_path
            )
        ):

            print(
                "Loading existing vector store..."
            )

            self.vector_store = (
                VectorStore.load(
                    vector_store_path
                )
            )

        else:

            print(
                "Creating new vector store..."
            )

            # --------------------------------------------------
            # Extract text from chunks
            # --------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in self.chunks
            ]

            # --------------------------------------------------
            # Generate document embeddings
            # --------------------------------------------------

            embeddings = (
                self.embedding_model
                .embed_documents(
                    texts
                )
            )

            if not embeddings:

                raise ValueError(
                    "No embeddings were generated."
                )

            # --------------------------------------------------
            # Determine embedding dimension
            # --------------------------------------------------

            dimension = len(
                embeddings[0]
            )

            # --------------------------------------------------
            # Create FAISS index
            # --------------------------------------------------

            self.vector_store = (
                VectorStore(
                    dimension
                )
            )

            # --------------------------------------------------
            # Add embeddings to FAISS
            # --------------------------------------------------

            self.vector_store.add(
                embeddings
            )

            # --------------------------------------------------
            # Save index if path provided
            # --------------------------------------------------

            if vector_store_path:

                directory = os.path.dirname(
                    vector_store_path
                )

                if directory:

                    os.makedirs(
                        directory,
                        exist_ok=True
                    )

                self.vector_store.save(
                    vector_store_path
                )

                print(
                    "Vector store saved."
                )


    # ==========================================================
    # RETRIEVAL + RERANKING
    # ==========================================================

    def retrieve_reranked(
        self,
        question,
        retrieval_k=10,
        final_k=5,
        source_filter=None
    ):

        """
        Retrieve top-k candidates from FAISS,
        optionally filter them by source,
        and rerank them using the Cross-Encoder.

        Parameters
        ----------
        question : str
            Rewritten user query.

        retrieval_k : int
            Number of candidates requested from FAISS.

        final_k : int
            Number of results returned after reranking.

        source_filter : str or None
            If None:
                Search all uploaded documents.

            If a filename is provided:
                Search only chunks belonging to that PDF.
        """

        print("\n" + "=" * 60)
        print("RAG RETRIEVAL")
        print("=" * 60)

        print(
            "Query:",
            question
        )

        print(
            "Retrieval K:",
            retrieval_k
        )

        print(
            "Final K:",
            final_k
        )

        print(
            "Source filter:",
            source_filter
        )

        # ======================================================
        # 1. EMBED QUERY
        # ======================================================

        query_embedding = (
            self.embedding_model
            .embed_query(
                question
            )
        )

        # ======================================================
        # 2. FAISS SEARCH
        # ======================================================

        # ------------------------------------------------------
        # IMPORTANT
        #
        # If we have a source filter, simply searching only
        # `retrieval_k` vectors can be problematic.
        #
        # Example:
        #
        # FAISS top 10:
        #
        # 1. Resume.pdf
        # 2. Resume.pdf
        # 3. Interview.pdf
        # ...
        # 10. Resume.pdf
        #
        # Rahul.pdf may be relevant but not appear in top 10.
        #
        # Therefore, when filtering by source, retrieve more
        # candidates first and then apply the source filter.
        # ======================================================

        total_chunks = len(
            self.chunks
        )

        if source_filter is not None:

            search_k = min(
                max(
                    retrieval_k * 5,
                    50
                ),
                total_chunks
            )

        else:

            search_k = min(
                retrieval_k,
                total_chunks
            )

        print(
            "FAISS search K:",
            search_k
        )

        distances, indices = (
            self.vector_store.search(
                query_embedding,
                search_k
            )
        )

        # ======================================================
        # 3. CONVERT FAISS RESULTS
        # ======================================================

        results = []

        for distance, index in zip(
            distances,
            indices
        ):

            index = int(index)

            # --------------------------------------------------
            # Invalid FAISS index
            # --------------------------------------------------

            if index < 0:
                continue

            # --------------------------------------------------
            # Out-of-range index
            # --------------------------------------------------

            if index >= len(
                self.chunks
            ):
                continue

            # --------------------------------------------------
            # Get chunk
            # --------------------------------------------------

            chunk = self.chunks[
                index
            ]

            # --------------------------------------------------
            # Get metadata
            # --------------------------------------------------

            metadata = chunk.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "source",
                "Unknown"
            )

            # ==================================================
            # SOURCE FILTER
            # ==================================================

            if source_filter is not None:

                if source != source_filter:

                    continue

            # ==================================================
            # ADD RESULT
            # ==================================================

            results.append(
                {
                    "chunk": chunk,

                    "distance": float(
                        distance
                    )
                }
            )

            # --------------------------------------------------
            # We can stop once we have enough candidates.
            # --------------------------------------------------

            if len(results) >= retrieval_k:

                break

        # ======================================================
        # DEBUG
        # ======================================================

        print("\nFILTERED RETRIEVAL RESULTS")
        print("-" * 60)

        print(
            "Number of candidates:",
            len(results)
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            chunk = result[
                "chunk"
            ]

            metadata = chunk.get(
                "metadata",
                {}
            )

            print(
                f"\nCandidate {rank}"
            )

            print(
                "Distance:",
                result.get(
                    "distance"
                )
            )

            print(
                "Source:",
                metadata.get(
                    "source"
                )
            )

            print(
                "Page:",
                metadata.get(
                    "page"
                )
            )

            print(
                "Text:",
                chunk.get(
                    "text",
                    ""
                )[:300]
            )

        # ======================================================
        # NO RESULTS
        # ======================================================

        if not results:

            print(
                "No matching chunks found."
            )

            return []

        # ======================================================
        # 4. CROSS-ENCODER RERANKING
        # ======================================================

        results = self.reranker.rerank(
            question,
            results,
            top_k=final_k
        )

        # ======================================================
        # FINAL RESULTS
        # ======================================================

        print("\nRERANKED RESULTS")
        print("-" * 60)

        for rank, result in enumerate(
            results,
            start=1
        ):

            chunk = result[
                "chunk"
            ]

            metadata = chunk.get(
                "metadata",
                {}
            )

            print(
                f"\nRank {rank}"
            )

            print(
                "Source:",
                metadata.get(
                    "source"
                )
            )

            print(
                "Page:",
                metadata.get(
                    "page"
                )
            )

            print(
                "Rerank score:",
                result.get(
                    "rerank_score"
                )
            )

        return results


    # ==========================================================
    # BUILD LLM CONTEXT
    # ==========================================================

    def build_context(
        self,
        results
    ):

        """
        Build the context that will be sent to the LLM.
        """

        context_parts = []

        for i, result in enumerate(
            results,
            start=1
        ):

            chunk = result[
                "chunk"
            ]

            text = chunk.get(
                "text",
                ""
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

            context_parts.append(
                f"""
[Context {i}]

Source: {source}

Page: {page}

{text}
"""
            )

        return "\n".join(
            context_parts
        )


    # ==========================================================
    # SELECT CITATIONS
    # ==========================================================

    def select_citations(
        self,
        results,
        max_citations=3
    ):

        """
        Select unique citations.

        For multi-PDF retrieval, the uniqueness key is:

            (source, page)

        rather than just:

            page

        because Page 1 of Rahul.pdf and Page 1 of
        Interview.pdf are different sources.
        """

        citations = []

        seen = set()

        for result in results:

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

            seen.add(
                key
            )

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

            if len(
                citations
            ) >= max_citations:

                break

        return citations


    # ==========================================================
    # MAIN RAG FUNCTION
    # ==========================================================

    def ask(
        self,
        question,
        chat_history=None,
        source_filter=None
    ):

        """
        Execute the complete RAG pipeline.

        Flow:

        Question
            ↓
        Query Rewriting
            ↓
        Embedding
            ↓
        FAISS Retrieval
            ↓
        Optional Source Filtering
            ↓
        Cross-Encoder Reranking
            ↓
        Context Construction
            ↓
        LLM Generation
            ↓
        Citation Selection
        """

        # ======================================================
        # VALIDATE QUESTION
        # ======================================================

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        # ======================================================
        # CONVERSATION HISTORY
        # ======================================================

        if chat_history is None:

            chat_history = []

        # ======================================================
        # DEBUG
        # ======================================================

        print("\n" + "=" * 60)
        print("RAG PIPELINE")
        print("=" * 60)

        print(
            "Original question:",
            question
        )

        print(
            "Source filter:",
            source_filter
        )

        # ======================================================
        # 1. QUERY REWRITING
        # ======================================================

        rewritten_query = (
            self.query_rewriter.rewrite(
                question,
                chat_history
            )
        )

        print(
            "\nRewritten query:",
            rewritten_query
        )

        # ======================================================
        # 2. RETRIEVAL + RERANKING
        # ======================================================

        results = (
            self.retrieve_reranked(
                question=rewritten_query,

                retrieval_k=10,

                final_k=5,

                source_filter=source_filter
            )
        )

        # ======================================================
        # 3. BUILD CONTEXT
        # ======================================================

        context = (
            self.build_context(
                results
            )
        )

        # ======================================================
        # 4. GENERATE ANSWER
        # ======================================================

        answer = (
            self.generator.generate(
                question=question,

                context=context
            )
        )

        # ======================================================
        # 5. SELECT CITATIONS
        # ======================================================

        sources = (
            self.select_citations(
                results,
                max_citations=3
            )
        )

        # ======================================================
        # 6. RETURN EVERYTHING
        # ======================================================

        return {

            "answer": answer,

            "rewritten_query": (
                rewritten_query
            ),

            "results": results,

            "sources": sources,

            "source_filter": source_filter
        }