import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)

    def add(self, vectors):

        vectors = np.array(
            vectors,
            dtype="float32"
        )

        self.index.add(vectors)

    def search(self, query_vector, k=5):

        query_vector = np.array(
            [query_vector],
            dtype="float32"
        )

        distances, indices = self.index.search(
            query_vector,
            k
        )

        return distances[0], indices[0]

    def save(self, path):

        faiss.write_index(
            self.index,
            path
        )

    @classmethod
    def load(cls, path):

        index = faiss.read_index(path)

        obj = cls.__new__(cls)
        obj.index = index

        return obj