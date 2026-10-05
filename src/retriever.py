import numpy as np


class Retriever:

    def __init__(
        self,
        embedding_model,
        vector_store,
        chunks
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.chunks = chunks


    def retrieve(
        self,
        query,
        k=5,
        page=None,
        source=None
    ):

        print("\n" + "=" * 60)
        print("RETRIEVER DEBUG")
        print("=" * 60)

        print("Query:")
        print(query)

        print("Requested k:", k)
        print("Page filter:", page)
        print("Source filter:", source)

        # =====================================================
        # 1. CREATE QUERY EMBEDDING
        # =====================================================

        query_embedding = self.embedding_model.embed_query(
            query
        )

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        print("\nQuery embedding dimension:")
        print(len(query_embedding))

        # =====================================================
        # 2. SEARCH FAISS
        # =====================================================

        distances, indices = self.vector_store.search(
            query_embedding,
            k
        )

        print("\nFAISS RESULTS")
        print("-" * 60)

        print("Distances:")
        print(distances)

        print("\nIndices:")
        print(indices)

        # =====================================================
        # 3. PROCESS RESULTS
        # =====================================================

        results = []

        # FAISS can return:
        #
        # index = -1
        #
        # when there are fewer vectors than requested.
        #
        # Ignore those invalid entries.

        for distance, index in zip(
            distances,
            indices
        ):

            index = int(index)

            # -------------------------------------------------
            # Invalid FAISS result
            # -------------------------------------------------

            if index < 0:
                continue

            # -------------------------------------------------
            # Prevent out-of-range access
            # -------------------------------------------------

            if index >= len(self.chunks):
                continue

            chunk = self.chunks[index]

            # =================================================
            # HANDLE STRING CHUNKS
            # =================================================

            if isinstance(
                chunk,
                str
            ):

                text = chunk

                metadata = {}

                # If metadata is available separately
                # through your chunk structure, it can be
                # added here later.

            # =================================================
            # HANDLE DICTIONARY CHUNKS
            # =================================================

            elif isinstance(
                chunk,
                dict
            ):

                text = chunk.get(
                    "text",
                    ""
                )

                metadata = chunk.get(
                    "metadata",
                    {}
                )

            else:

                # Unknown chunk format
                continue

            # =================================================
            # PAGE FILTER
            # =================================================

            if page is not None:

                if isinstance(
                    page,
                    list
                ):

                    if page and metadata.get(
                        "page"
                    ) not in page:

                        continue

                else:

                    if metadata.get(
                        "page"
                    ) != page:

                        continue

            # =================================================
            # SOURCE FILTER
            # =================================================

            if source is not None:

                if metadata.get(
                    "source"
                ) != source:

                    continue

            # =================================================
            # BUILD RESULT
            # =================================================

            results.append(
                {
                    "chunk": {
                        "text": text,
                        "metadata": metadata
                    },

                    "distance": float(
                        distance
                    )
                }
            )

        # =====================================================
        # 4. LIMIT RESULTS
        # =====================================================

        results = results[:k]

        # =====================================================
        # 5. DEBUG OUTPUT
        # =====================================================

        print("\nFINAL RETRIEVED RESULTS")
        print("-" * 60)

        print(
            "Number of results:",
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
                f"\nResult {rank}"
            )

            print(
                "Distance:",
                result.get(
                    "distance"
                )
            )

            print(
                "Page:",
                metadata.get(
                    "page"
                )
            )

            print(
                "Source:",
                metadata.get(
                    "source"
                )
            )

            print("Text:")

            print(
                chunk.get(
                    "text",
                    ""
                )[:500]
            )

        return results