import faiss
import numpy as np
import sqlite3
from typing import List, Dict, Tuple
from data_schema import IndexData
from contextlib import closing
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import unittest
from termcolor import colored

class FqlDb:
    def __init__(self, index_name: str = "fql_index", dimension: int = 384, db_name: str = "fql.db", model_name: str = "all-MiniLM-L6-v2", step_size: int = 3):
        self.index_name = index_name
        self.dimension = dimension
        self.db_name = db_name
        self.model = SentenceTransformer(model_name)
        self.step_size = step_size
        self.index = self.build_index(dimension=self.dimension)
        self.connection = sqlite3.connect(self.db_name, isolation_level=None)
        sqlite3.register_adapter(np.int64, int)

    def build_index(self, dimension: int) -> faiss.IndexIDMap2:
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        return index

    def add_to_index(self, data: List[IndexData]) -> None:
        ids = []
        vectors = []
        for point in data:
            ids.append(point.id)
            vectors.append(point.vector)

        ids = np.array(ids, dtype=np.int64)
        vectors = np.array(vectors, dtype=np.float32)
        self.index.add_with_ids(vectors, ids)

    def save_index(self) -> None:
        chunk = faiss.serialize_index(self.index)
        with open(f"""{self.index_name}.pkl""", "wb") as f:
            pickle.dump(chunk, f)

    def load_index(self) -> faiss.Index:
        with open(f"""{self.index_name}.pkl""", "rb") as f:
            index = faiss.deserialize_index(pickle.load(f))
        return index

    def store_to_db(self, data: List[IndexData]) -> None:
        try:
            values = []
            for point in data:
                values.append((point.id, point.content, str(point.metadata)))
            with self.connection:

                res = self.connection.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(id INTEGER PRIMARY KEY, content TEXT, metadata TEXT)""")

                res = self.connection.executemany(
                    f"""INSERT INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""", values)

                res = self.connection.execute(f"""SELECT * FROM {self.index_name}""")
                rows = res.fetchall()
                # print("here",rows)  # Print

        except Exception as e:
            print('Could not complete operation:', e)

    def search_index(self, query: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        D, I = self.index.search(query, k)
        # print(type(D),type(I))
        return list(D[0]), list(I[0])

    def retrieve(self, ids: List[int]):
        # ids = list(map(int,))
        cur = self.connection.cursor()
        rows = []
        qs = ", ".join("?" * len(ids))
        with closing(self.connection.cursor()) as cur:

            cur.execute(f"""SELECT * FROM {self.index_name} WHERE id IN ({','.join(['?']*len(ids))})""", ids)
            rows = cur.fetchall()
        return rows

    def grouper(self, iterable: list, n: int):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    def index_data(self, data_chunks: List[str]):
        id = 0
        for batch in tqdm(list(self.grouper(data_chunks, self.step_size))):
            # Calculate embeddings
            embeddings = self.model.encode(batch)
            all_points = []
            for i in range(len(batch)):
                point = IndexData(vector=embeddings[i], content=batch[i], id=id)
                id += 1
                all_points.append(point)
            # add to index
            self.add_to_index(index=self.index, data=all_points)
            self.store_to_db(data=all_points, connection=self.connection, index_name=self.index_name)

        self.save_index()
        self.index = self.load_index()

class TestFqlDb(unittest.TestCase):
    def setUp(self):
        self.db = FqlDb(index_name="test_index", db_name="test.db")
        self.data_chunks = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
            "Mayank is a software developer.",
            "I don't think it works.",
            "why do you want to know about my profession?",
            "what is your salary?",
            "have conviction in the power of goodness.",
            "Mayank works as a software developer."
        ]
        self.db.index_data(self.data_chunks)

    def test_search_and_retrieve(self):
        query = "Who is Mayank?"
        embeddings = self.db.model.encode([query])
        D, I = self.db.search_index(query=embeddings, k=2)
        res = self.db.retrieve(ids=I)

        self.assertEqual(len(res), 2)
        print(colored("\nTest: Search and Retrieve", "blue"))
        print(colored("Query:", "green"), query)
        print(colored("Results:", "green"), res)
        print(colored("Distances:", "green"), D)
        print(colored("IDs:", "green"), I)
        print(colored("Status: PASSED", "green"))

    def tearDown(self):
        self.db.connection.close()
        import os
        os.remove("test.db")
        os.remove("test_index.pkl")

if __name__ == '__main__':
    unittest.main()
