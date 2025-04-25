import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union
from data_schema import IndexData
from contextlib import closing
import pickle
import os
from termcolor import cprint

class FqlDb:
    """
    A class for combining vector similarity search with a SQLite database.
    """

    def __init__(self, index_name: str, dimension: int, db_name: str = "fql.db"):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name of the index.
            dimension (int): The dimension of the vectors.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
        """
        self.index_name = index_name
        self.dimension = dimension
        self.db_name = db_name
        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> faiss.IndexIDMap2:
        """
        Loads the index from file if it exists, otherwise builds a new index.

        Returns:
            faiss.IndexIDMap2: The loaded or newly built index.
        """
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file):
            try:
                self.index = self.load_index(self.index_name)
                cprint(f"Index {self.index_name} loaded successfully.", "green")
            except Exception as e:
                cprint(f"Failed to load index {self.index_name}: {e}. Building a new one.", "yellow")
                self.index = self.build_index(self.dimension)
                cprint(f"Index {self.index_name} created successfully.", "green")
        else:
            self.index = self.build_index(self.dimension)
            cprint(f"Index {self.index_name} created successfully.", "green")
        return self.index


    def build_index(self, dimension: int) -> faiss.IndexIDMap2:
        """
        Builds a FAISS index.

        Args:
            dimension (int): The dimension of the vectors.

        Returns:
            faiss.IndexIDMap2: The FAISS index.
        """
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        return index

    def add_to_index(self, data: List[IndexData]) -> None:
        """
        Adds data to the FAISS index.

        Args:
            data (List[IndexData]): A list of IndexData objects to add.
        """
        ids = []
        vectors = []
        for point in data:
            ids.append(point.id)
            vectors.append(point.vector)

        ids = np.array(ids, dtype=np.int64)
        vectors = np.array(vectors, dtype=np.float32)
        self.index.add_with_ids(vectors, ids)
        cprint(f"Added {len(data)} vectors to index {self.index_name}.", "green")

    def save_index(self) -> None:
        """
        Saves the FAISS index to a file.
        """
        chunk = faiss.serialize_index(self.index)
        with open(f"{self.index_name}.pkl", "wb") as f:
            pickle.dump(chunk, f)
        cprint(f"Index {self.index_name} saved successfully.", "green")

    def load_index(self, index_name: str) -> faiss.Index:
        """
        Loads a FAISS index from a file.

        Args:
            index_name (str): The name of the index to load.

        Returns:
            faiss.Index: The loaded FAISS index.
        """
        with open(f"{index_name}.pkl", "rb") as f:
            index = faiss.deserialize_index(pickle.load(f))
        return index

    def store_to_db(self, data: List[IndexData]) -> None:
        """
        Stores data to the SQLite database.

        Args:
            data (List[IndexData]): A list of IndexData objects to store.
        """
        try:
            values = []
            for point in data:
                values.append((point.id, point.content, str(point.metadata)))
            with self.connection:
                res = self.connection.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(id INTEGER PRIMARY KEY, content TEXT, metadata TEXT)"""
                )

                # Use INSERT OR IGNORE to avoid errors if IDs already exist,
                # or handle potential duplicates based on desired behavior.
                # For this test, ignoring duplicates seems reasonable if the goal
                # is just to ensure the data exists. If overwriting is desired,
                # use INSERT OR REPLACE.
                res = self.connection.executemany(
                    f"""INSERT OR IGNORE INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""", values
                )
            cprint(f"Stored/updated {len(data)} records in database table {self.index_name}.", "green")

        except Exception as e:
            cprint(f"Could not complete database operation: {e}", "red")
            raise

    def search_index(self, query: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """
        Searches the FAISS index for the nearest neighbors of a query vector.

        Args:
            query (np.ndarray): The query vector.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing the distances and IDs of the nearest neighbors.
        """
        D, I = self.index.search(query, k)
        return list(D[0]), list(I[0])

    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """
        Retrieves data from the SQLite database based on a list of IDs.

        Args:
            ids (List[int]): A list of IDs to retrieve.

        Returns:
            List[Tuple]: A list of tuples containing the retrieved data.
        """
        if not ids:
            return []
        cur = self.connection.cursor()
        rows = []
        qs = ", ".join("?" * len(ids))
        with closing(self.connection.cursor()) as cur:
            cur.execute(f"""SELECT * FROM {self.index_name} WHERE id IN ({','.join(['?']*len(ids))})""", ids)
            rows = cur.fetchall()
        return rows

    def __del__(self):
         """
         Closes the database connection when the object is deleted.
         """
         if hasattr(self, "connection") and self.connection:
             self.connection.close()
             cprint("Database connection closed.", "yellow")


def cleanup_test_files(index_name: str, db_name: str):
    """Removes test files."""
    index_file = f"{index_name}.pkl"
    db_file = db_name
    files_removed = False
    try:
        if os.path.exists(index_file):
            os.remove(index_file)
            files_removed = True
        if os.path.exists(db_file):
            os.remove(db_file)
            files_removed = True
        if files_removed:
            cprint("Test files removed.", "yellow")
    except Exception as e:
        cprint(f"Error during test file cleanup: {e}", "red")


def test_fql_db():
    """
    Tests the FqlDb class.
    """
    cprint("Starting FqlDb tests...", "blue")

    # Test setup
    index_name = "test_index"
    dimension = 2
    db_name = "test.db"

    # --- Ensure clean state before test ---
    cleanup_test_files(index_name, db_name)
    # --------------------------------------

    test_data = [
        IndexData(vector=np.array([1.0, 2.0]), id=1, content="Test content 1", metadata={"key1": "value1"}),
        IndexData(vector=np.array([3.0, 4.0]), id=2, content="Test content 2", metadata={"key2": "value2"}),
    ]

    # Create FqlDb instance
    fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name)

    # Test add_to_index and store_to_db
    fql_db.add_to_index(test_data)
    fql_db.store_to_db(test_data)

    # Test search_index and retrieve
    query_vector = np.array([[2.0, 3.0]], dtype=np.float32)
    distances, ids = fql_db.search_index(query_vector, k=2)
    assert len(distances) == 2
    assert len(ids) == 2
    cprint(f"Search results - Distances: {distances}, IDs: {ids}", "cyan")

    retrieved_data = fql_db.retrieve(ids)
    assert len(retrieved_data) == 2, f"Expected 2 results, got {len(retrieved_data)}"
    retrieved_ids = {row[0] for row in retrieved_data}
    assert set(ids) == retrieved_ids, f"Expected IDs {set(ids)}, got {retrieved_ids}"
    cprint(f"Retrieved data: {retrieved_data}", "cyan")


    # Test save_index and load_index
    fql_db.save_index()
    # Close the current connection before loading a new instance
    fql_db.connection.close()
    fql_db.connection = None # Prevent __del__ from trying to close again

    loaded_fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name)  # Load index
    # Verify loaded index has data
    assert loaded_fql_db.index.ntotal == len(test_data)
    cprint("Index loaded successfully after save.", "green")

    # Test search and retrieve with loaded index
    distances_loaded, ids_loaded = loaded_fql_db.search_index(query_vector, k=2)
    assert ids_loaded == ids
    retrieved_data_loaded = loaded_fql_db.retrieve(ids_loaded)
    assert len(retrieved_data_loaded) == 2

    # --- Clean up test files ---
    del fql_db # Ensure __del__ is called if connection wasn't manually closed
    del loaded_fql_db # Ensure __del__ is called
    cleanup_test_files(index_name, db_name)
    # ---------------------------

    cprint("All FqlDb tests passed!", "green")


if __name__ == "__main__":
    test_fql_db()
