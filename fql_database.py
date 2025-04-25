import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Any
from data_schema import IndexData
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse

# Import fastembed if USE_FASTEMBED is True
USE_FASTEMBED = True
if USE_FASTEMBED:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        cprint("fastembed is not installed. Please install it to use fastembed features.", "yellow")
        USE_FASTEMBED = False
else:
    TextEmbedding = Any

class FqlDb:
    """
    A class for combining vector similarity search with a SQLite database,
    offering options for FAISS or fastembed indexing.
    """

    def __init__(
        self,
        index_name: str,
        dimension: int,
        db_name: str = "fql.db",
        use_fastembed: bool = False,
        fastembed_model_name: str = "BAAI/bge-small-en-v1.5",  # Default fastembed model
    ):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name of the index and database table.
            dimension (int): The dimension of the vectors.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            use_fastembed (bool, optional): Whether to use fastembed for embeddings. Defaults to False (FAISS).
            fastembed_model_name (str, optional): The name of the fastembed model to use.
                Defaults to "BAAI/bge-small-en-v1.5".
        """
        self.index_name = index_name
        self.dimension = dimension
        self.db_name = db_name
        self.use_fastembed = use_fastembed
        self.fastembed_model_name = fastembed_model_name
        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self.embedding_model = None  # Initialize embedding_model
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> Union[faiss.IndexIDMap2, Any]:
        """Loads the index from file if it exists, otherwise builds a new index.

        Returns:
            Union[faiss.IndexIDMap2, fastembed.embedding.TextEmbedding]: The loaded or newly built index.
        """
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file):
            try:
                self.index = self.load_index()
                cprint(f"Index {self.index_name} loaded successfully.", "green")
            except Exception as e:
                cprint(f"Failed to load index {self.index_name}: {e}. Building a new one.", "yellow")
                self.index = self.build_index()
                cprint(f"Index {self.index_name} created successfully.", "green")
        else:
            self.index = self.build_index()
            cprint(f"Index {self.index_name} created successfully.", "green")
        return self.index

    def build_index(self) -> Union[faiss.IndexIDMap2, Any]:
        """Builds a FAISS index or initializes a fastembed model based on configuration.

        Returns:
            Union[faiss.IndexIDMap2, fastembed.embedding.TextEmbedding]: The FAISS index or fastembed model.
        """
        if self.use_fastembed:
            try:
                self.embedding_model = TextEmbedding(model_name=self.fastembed_model_name)
                cprint(
                    f"fastembed model '{self.fastembed_model_name}' initialized successfully.",
                    "green",
                )
                return self.embedding_model #Return the fastembed model
            except Exception as e:
                cprint(
                    f"Failed to initialize fastembed model: {e}. Falling back to FAISS.",
                    "red",
                )
                self.use_fastembed = False  # Fallback to FAISS
                return self._build_faiss_index(self.dimension)
        else:
            return self._build_faiss_index(self.dimension)

    def _build_faiss_index(self, dimension: int) -> faiss.IndexIDMap2:
        """Builds a FAISS index.

        Args:
            dimension (int): The dimension of the vectors.

        Returns:
            faiss.IndexIDMap2: The FAISS index.
        """
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        return index

    def add_to_index(self, data: List[IndexData]) -> None:
        """Adds data to the FAISS index or generates embeddings using fastembed.

        Args:
            data (List[IndexData]): A list of IndexData objects to add.
        """
        if self.use_fastembed:
            self._add_to_index_fastembed(data)
        else:
            self._add_to_index_faiss(data)

    def _add_to_index_faiss(self, data: List[IndexData]) -> None:
        """Adds data to the FAISS index.

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
        cprint(f"Added {len(data)} vectors to FAISS index {self.index_name}.", "green")

    def _add_to_index_fastembed(self, data: List[IndexData]) -> None:
         """Adds data to the fastembed index.

         Args:
             data (List[IndexData]): A list of IndexData objects to add.
         """
         cprint("fastembed indexing is not supported.  Please use FAISS indexing.", "red")
         raise NotImplementedError("fastembed indexing is not supported.  Please use FAISS indexing.")

    def save_index(self) -> None:
        """Saves the FAISS index to a file.
        """
        if self.use_fastembed:
            cprint("Saving index with fastembed is not supported.  Please use FAISS indexing.", "red")
        else:
            chunk = faiss.serialize_index(self.index)
            with open(f"{self.index_name}.pkl", "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"FAISS index {self.index_name} saved successfully.", "green")

    def load_index(self) -> faiss.Index:
        """Loads the FAISS index from file using the instance's index_name.

        Returns:
            faiss.Index: The loaded FAISS index.
        """
        index_file = f"{self.index_name}.pkl"
        with open(index_file, "rb") as f:
            index = faiss.deserialize_index(pickle.load(f))
        return index

    def store_to_db(self, data: List[IndexData]) -> None:
        """Stores data to the SQLite database.

        Args:
            data (List[IndexData]): A list of IndexData objects to store.
        """
        try:
            values = []
            for point in data:
                values.append((point.id, point.content, str(point.metadata)))
            # Use a dedicated cursor for table creation and insertion
            with closing(self.connection.cursor()) as cur:
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(id INTEGER PRIMARY KEY, content TEXT, metadata TEXT)"""
                )
                cur.executemany(
                    f"""INSERT OR IGNORE INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""", values
                )
            # Since isolation_level=None (autocommit), changes should be persisted immediately.
            # No explicit commit needed here.
            cprint(f"Stored/updated {len(data)} records in database table {self.index_name}.", "green")

        except Exception as e:
            cprint(f"Could not complete database operation: {e}", "red")
            raise

    def search_index(self, query: str, k: int = 3) -> Tuple[List[float], List[int]]:
        """Searches the FAISS index or generates embeddings using fastembed for the nearest neighbors of a query.

        Args:
            query (str): The query string.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing the distances and IDs of the nearest neighbors.
        """
        if self.use_fastembed:
            return self._search_index_fastembed(query, k)
        else:
            return self._search_index_faiss(query, k)

    def _search_index_faiss(self, query: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """Searches the FAISS index for the nearest neighbors of a query vector.

        Args:
            query (np.ndarray): The query vector.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing the distances and IDs of the nearest neighbors.
        """
        if not isinstance(query, np.ndarray):
            cprint("Query must be a numpy array for FAISS search.", "red")
            raise ValueError("Query must be a numpy array for FAISS search.")
        D, I = self.index.search(query, k)
        # Convert numpy types to standard Python types
        distances = [float(d) for d in D[0]]
        ids = [int(i) for i in I[0]]
        return distances, ids

    def _search_index_fastembed(self, query: str, k: int = 3) -> Tuple[List[float], List[int]]:
        """Searches the fastembed index for the nearest neighbors of a query string.

        Args:
            query (str): The query string.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing the distances and IDs of the nearest neighbors.
        """
        cprint("fastembed search is not supported.  Please use FAISS indexing.", "red")
        raise NotImplementedError("fastembed search is not supported.  Please use FAISS indexing.")

    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """Retrieves data from the SQLite database based on a list of IDs.

        Args:
            ids (List[int]): A list of IDs to retrieve. Expects standard Python ints.

        Returns:
            List[Tuple]: A list of tuples containing the retrieved data.
        """
        if not ids:
            return []

        rows = []
        cur = None
        try:
            cur = self.connection.cursor()
            # Ensure IDs are standard Python integers (should be guaranteed by search_index now)
            placeholders = ','.join('?' * len(ids))
            sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"
            cur.execute(sql, ids)
            rows = cur.fetchall()
        except sqlite3.Error as e: # Catch specific SQLite errors
            cprint(f"Error during retrieve: {e}", "red")
            # Optionally re-raise or handle
            raise # Re-raise the exception to make test failures clear
        except Exception as e:
             cprint(f"An unexpected error occurred during retrieve: {e}", "red")
             raise
        finally:
            if cur:
                cur.close()
        return rows

    def __del__(self):
         """
         Closes the database connection when the object is deleted.
         """
         if hasattr(self, "connection") and self.connection:
             self.connection.close()
             cprint("Database connection closed.", "yellow")

    def usage(self):
        """Prints usage instructions for the FqlDb class."""
        cprint("Usage:","cyan")
        cprint("Initialize FqlDb with FAISS (default):","green")
        cprint("  fql_db = FqlDb(index_name='my_index', dimension=128, db_name='my_db.db')","white")
        if USE_FASTEMBED:
            cprint("Initialize FqlDb with fastembed:","green")
            cprint("  fql_db = FqlDb(index_name='my_index', dimension=384, db_name='my_db.db', use_fastembed=True, fastembed_model_name='BAAI/bge-small-en-v1.5')","white")
        cprint("Add data to the index:","green")
        cprint("  fql_db.add_to_index(data=[IndexData(...)])","white")
        cprint("Store data to the database:","green")
        cprint("  fql_db.store_to_db(data=[IndexData(...)])","white")
        cprint("Search the index:","green")
        cprint("  distances, ids = fql_db.search_index(query=np.array([1.0, 2.0], dtype=np.float32), k=3)","white")
        cprint("Retrieve data from the database:","green")
        cprint("  retrieved_data = fql_db.retrieve(ids=[1, 2, 3])","white")
        cprint("Save the index:","green")
        cprint("  fql_db.save_index()","white")
        cprint("Load the index:","green")
        cprint("  fql_db.load_index()","white")


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

    fql_db = None # Initialize to None for finally block
    loaded_fql_db = None # Initialize to None for finally block
    try:
        test_data = [
            IndexData(vector=np.array([1.0, 2.0], dtype=np.float32), id=1, content="Test content 1", metadata={"key1": "value1"}),
            IndexData(vector=np.array([3.0, 4.0], dtype=np.float32), id=2, content="Test content 2", metadata={"key2": "value2"}),
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
        # Sort IDs to ensure consistent order for assertion
        ids.sort()
        assert ids == [1, 2]
        cprint(f"Search results - Distances: {distances}, IDs: {ids}", "cyan")

        retrieved_data = fql_db.retrieve(ids)
        assert len(retrieved_data) == 2, f"Expected 2 results, got {len(retrieved_data)}. Data: {retrieved_data}"
        retrieved_ids = sorted([row[0] for row in retrieved_data]) # Sort retrieved IDs
        assert retrieved_ids == ids, f"Expected IDs {ids}, got {retrieved_ids}"
        cprint(f"Retrieved data: {retrieved_data}", "cyan")


        # Test save_index and load_index
        fql_db.save_index()
        # Close the current connection before loading a new instance
        if fql_db.connection:
            fql_db.connection.close()
            fql_db.connection = None # Prevent __del__ from trying to close again

        loaded_fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name)  # Load index
        # Verify loaded index has data
        assert loaded_fql_db.index.ntotal == len(test_data)
        cprint("Index loaded successfully after save.", "green")

        # Test search and retrieve with loaded index
        distances_loaded, ids_loaded = loaded_fql_db.search_index(query_vector, k=2)
        ids_loaded.sort() # Sort for consistent assertion
        assert ids_loaded == ids
        retrieved_data_loaded = loaded_fql_db.retrieve(ids_loaded)
        assert len(retrieved_data_loaded) == 2
        retrieved_ids_loaded = sorted([row[0] for row in retrieved_data_loaded])
        assert retrieved_ids_loaded == ids_loaded

        cprint("All FqlDb tests passed!", "green")

    except Exception as e:
        cprint(f"Test failed: {e}", "red")
        raise # Re-raise exception to make test runner aware of failure
    finally:
        # --- Clean up test files ---
        # Ensure connections are closed if objects were created
        if fql_db and fql_db.connection:
            fql_db.connection.close()
            fql_db.connection = None
        if loaded_fql_db and loaded_fql_db.connection:
            loaded_fql_db.connection.close()
            loaded_fql_db.connection = None
        # Explicitly delete objects before cleanup to trigger __del__ if needed (though connection is now None)
        del fql_db
        del loaded_fql_db
        cleanup_test_files(index_name, db_name)
        # ---------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FqlDb or print usage.")
    parser.add_argument("--usage", action="store_true", help="Print FqlDb usage instructions.")
    args = parser.parse_args()

    if args.usage:
        fql_db = FqlDb(index_name='temp', dimension=1)  # Dummy instantiation for usage printing
        fql_db.usage()
        del fql_db
    else:
        test_fql_db()
