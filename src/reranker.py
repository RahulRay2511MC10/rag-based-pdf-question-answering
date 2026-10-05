from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        results,
        top_k=5
    ):

        pairs = []

        for result in results:

            pairs.append([
                query,
                result["chunk"]["text"]
            ])

        scores = self.model.predict(pairs)

        for result, score in zip(
            results,
            scores
        ):

            result["rerank_score"] = float(score)

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return results[:top_k]