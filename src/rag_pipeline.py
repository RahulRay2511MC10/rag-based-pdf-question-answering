import os

from src.embeddings import get_embedding_model
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.reranker import Reranker
from src.query_rewriter import QueryRewriter
from src.generator import Generator


class RAGPipeline:

    def __init__(
        self,
        chunks,
        vector_store_path="vector_store/index.faiss"
    ):

        # =========================================================
        # 1. VALIDATE CHUNKS
        # =========================================================

        if not isinstance(chunks, list):
            raise TypeError(
                "chunks must be a list"
            )

        self.chunks = chunks

        # =========================================================
        # 2. EMBEDDING MODEL
        # =========================================================

        self.embedding_model = get_embedding_model()

        # =========================================================
        # 3. VECTOR STORE
        # =========================================================

        if os.path.exists(vector_store_path):

            print(
                "Loading existing vector store..."
            )

            self.vector_store = VectorStore.load(
                vector_store_path
            )

        else:

            print(
                "Creating vector store..."
            )

            # IMPORTANT:
            # Embed chunk TEXT, not the complete chunk dictionary.
            texts = [
                chunk["text"]
                for chunk in self.chunks
            ]

            embeddings = (
                self.embedding_model.embed_documents(
                    texts
                )
            )

            dimension = len(
                embeddings[0]
            )

            self.vector_store = VectorStore(
                dimension
            )

            self.vector_store.add(
                embeddings
            )

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

        # =========================================================
        # 4. RETRIEVER
        # =========================================================

        self.retriever = Retriever(
            self.embedding_model,
            self.vector_store,
            self.chunks
        )

        # =========================================================
        # 5. QUERY REWRITER
        # =========================================================

        self.query_rewriter = QueryRewriter()

        # =========================================================
        # 6. CROSS-ENCODER RERANKER
        # =========================================================

        self.reranker = Reranker()

        # =========================================================
        # 7. GENERATOR
        # =========================================================

        self.generator = Generator()


    # =============================================================
    # BASIC FAISS RETRIEVAL
    # =============================================================

    def retrieve(
        self,
        question,
        k=5,
        page=None,
        source=None
    ):

        return self.retriever.retrieve(
            question,
            k=k,
            page=page,
            source=source
        )


    # =============================================================
    # FAISS + CROSS-ENCODER RERANKING
    # =============================================================

    def retrieve_reranked(
        self,
        question,
        retrieval_k=10,
        final_k=5,
        page=None,
        source=None
    ):

        # ---------------------------------------------------------
        # STEP 1: FAISS RETRIEVAL
        # ---------------------------------------------------------

        results = self.retriever.retrieve(
            question,
            k=retrieval_k,
            page=page,
            source=source
        )

        if not results:
            return []

        # ---------------------------------------------------------
        # STEP 2: CROSS-ENCODER RERANKING
        # ---------------------------------------------------------

        results = self.reranker.rerank(
            query=question,
            results=results,
            top_k=final_k
        )

        # ---------------------------------------------------------
        # STEP 3: DEBUG RERANKED RESULTS
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("RERANKED RESULTS")
        print("=" * 60)

        for rank, result in enumerate(
            results,
            start=1
        ):

            chunk = result.get(
                "chunk",
                {}
            )

            metadata = chunk.get(
                "metadata",
                {}
            )

            print(
                f"\nRank {rank}"
            )

            print(
                "FAISS Distance:",
                result.get("distance")
            )

            print(
                "Rerank Score:",
                result.get("rerank_score")
            )

            print(
                "Page:",
                metadata.get("page")
            )

            print(
                "Source:",
                metadata.get("source")
            )

            print("Text:")

            print(
                chunk.get(
                    "text",
                    ""
                )[:500]
            )

        return results


    # =============================================================
    # PAGE-LEVEL CITATION SELECTION
    # =============================================================

    def select_citations(
        self,
        results,
        max_citations=3
    ):
        """
        Select citation pages from reranked results.

        Strategy:

        1. Group retrieved chunks by page.
        2. If multiple chunks belong to the same page,
           keep the highest reranker score.
        3. Rank pages using their best reranker score.
        4. Return only the strongest unique pages.

        This prevents output such as:

            Page 2
            Page 2
            Page 2
            Page 1
            Page 4

        and instead produces:

            Page 2
            Page 1
            Page 4
        """

        if not results:
            return []

        # ---------------------------------------------------------
        # Group results by page
        # ---------------------------------------------------------

        page_candidates = {}

        for result in results:

            chunk = result.get(
                "chunk",
                {}
            )

            metadata = chunk.get(
                "metadata",
                {}
            )

            page_number = metadata.get(
                "page"
            )

            source_name = metadata.get(
                "source"
            )

            rerank_score = result.get(
                "rerank_score"
            )

            # Ignore results without page information
            if page_number is None:
                continue

            # If reranker score is unavailable,
            # use a very low score.
            if rerank_score is None:
                rerank_score = float(
                    "-inf"
                )

            citation_candidate = {
                "source": source_name,
                "page": page_number,
                "distance": result.get(
                    "distance"
                ),
                "rerank_score": rerank_score,
                "text": chunk.get(
                    "text",
                    ""
                )
            }

            # -----------------------------------------------------
            # Keep only the strongest chunk from each page
            # -----------------------------------------------------

            if page_number not in page_candidates:

                page_candidates[
                    page_number
                ] = citation_candidate

            else:

                existing_score = (
                    page_candidates[
                        page_number
                    ]["rerank_score"]
                )

                if rerank_score > existing_score:

                    page_candidates[
                        page_number
                    ] = citation_candidate

        # ---------------------------------------------------------
        # Sort pages according to best reranker score
        # ---------------------------------------------------------

        citations = list(
            page_candidates.values()
        )

        citations.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        # ---------------------------------------------------------
        # Limit number of citation pages
        # ---------------------------------------------------------

        citations = citations[
            :max_citations
        ]

        # ---------------------------------------------------------
        # DEBUG CITATIONS
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("SELECTED CITATIONS")
        print("=" * 60)

        for rank, citation in enumerate(
            citations,
            start=1
        ):

            print(
                f"Rank {rank} | "
                f"Page {citation['page']} | "
                f"Rerank Score "
                f"{citation['rerank_score']}"
            )

        return citations


    # =============================================================
    # COMPLETE RAG PIPELINE
    # =============================================================

    def ask(
        self,
        question,
        page=None,
        source=None,
        chat_history=None
    ):

        # =========================================================
        # STEP 1: ORIGINAL QUESTION
        # =========================================================

        print("\n" + "=" * 60)
        print("ORIGINAL QUESTION")
        print("=" * 60)

        print(
            question
        )

        # =========================================================
        # STEP 2: QUERY REWRITING
        # =========================================================

        if chat_history is None:

            chat_history = []

        rewritten_query = (
            self.query_rewriter.rewrite(
                question,
                chat_history
            )
        )

        print("\n" + "=" * 60)
        print("REWRITTEN QUERY")
        print("=" * 60)

        print(
            rewritten_query
        )

        # =========================================================
        # STEP 3: RETRIEVAL + RERANKING
        # =========================================================

        results = self.retrieve_reranked(
            question=rewritten_query,
            retrieval_k=10,
            final_k=5,
            page=page,
            source=source
        )

        # =========================================================
        # STEP 4: NO RESULTS
        # =========================================================

        if not results:

            return {
                "answer": (
                    "The information is not available "
                    "in the provided documents."
                ),
                "results": [],
                "sources": []
            }

        # =========================================================
        # STEP 5: BUILD CONTEXT
        # =========================================================

        context_parts = []

        for result in results:

            chunk = result.get(
                "chunk",
                {}
            )

            text = chunk.get(
                "text",
                ""
            )

            if text:

                context_parts.append(
                    text
                )

        context = "\n\n".join(
            context_parts
        )

        # =========================================================
        # PRINT CONTEXT
        # =========================================================

        print("\n" + "=" * 60)
        print("CONTEXT")
        print("=" * 60)

        print(
            context
        )

        # =========================================================
        # STEP 6: GENERATE ANSWER
        # =========================================================

        answer = self.generator.generate(
            question,
            context
        )

        # =========================================================
        # STEP 7: SELECT PAGE-LEVEL CITATIONS
        # =========================================================

        selected_citations = (
            self.select_citations(
                results,
                max_citations=1
            )
        )

        # =========================================================
        # STEP 8: BUILD SOURCE INFORMATION
        # =========================================================

        sources = []

        for citation in selected_citations:

            sources.append({
                "source": citation.get(
                    "source"
                ),
                "page": citation.get(
                    "page"
                ),
                "distance": citation.get(
                    "distance"
                ),
                "rerank_score": citation.get(
                    "rerank_score"
                )
            })

        # =========================================================
        # STEP 9: RETURN FINAL RESULT
        # =========================================================

        return {
            "answer": answer,
            "results": results,
            "sources": sources
        }