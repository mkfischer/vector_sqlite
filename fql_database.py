import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union
from data_schema import IndexData
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse

try:
    from fastembed import TextEmbedding  # Optional dependency
    has_fastembed = True
except ImportError:
    cprint("FastEmbed not found. Install with 'pip install fastembed'.", "yellow")
    has_fastembed = False

class FqlDb:
    """
    A class for combining vector similarity search with a SQLite database,
    offering options for FAISS-based indexing and FastEmbed integration.
    """

    def __init__(
        self,
        index_name: str,
        dimension: int,
        db_name: str = "fql.db",
        use_fastembed: bool = False,
        fastembed_model_name: str = "BAAI/bge-small-en-v1.5"  # Default FastEmbed model
    ):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name of the index and database table.
            dimension (int): The dimension of the vectors. Required if not using FastEmbed.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            use_fastembed (bool, optional): Whether to use FastEmbed for embedding generation. Defaults to False.
            fastembed_model_name (str, optional): The name of the FastEmbed model to use.
                Defaults to "BAAI/bge-small-en-v1.5".
        Raises:
            ValueError: If FastEmbed is specified but the library is not installed.
        """
        self.index_name = index_name
        self.db_name = db_name
        self.use_fastembed = use_fastembed
        self.fastembed_model_name = fastembed_model_name

        if use_fastembed:
            if not has_fastembed:
                raise ValueError("FastEmbed is specified but the library is not installed. Please install it using 'pip install fastembed'.")
            self.embedding_model = TextEmbedding(model_name=self.fastembed_model_name)
            self.dimension = self.embedding_model.dimension  # Infer dimension from FastEmbed model
            cprint(f"Using FastEmbed model: {self.fastembed_model_name} with dimension {self.dimension}", "green")
        else:
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("Dimension must be a positive integer when not using FastEmbed.")
            self.dimension = dimension
            self.embedding_model = None

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
                self.index = self.load_index()
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
        Adds data to the FAISS index. If using FastEmbed, generates embeddings
        before adding to the index.

        Args:
            data (List[IndexData]): A list of IndexData objects to add.
        """
        ids = []
        vectors = []
        for point in data:
            ids.append(point.id)
            if self.use_fastembed and self.embedding_model:
                try:
                    vectors.append(self.embedding_model.embed(point.content)[0])  # Generate embedding using FastEmbed
                except Exception as e:
                    cprint(f"FastEmbed embedding generation failed for id {point.id}: {e}", "red")
                    raise
            else:
                vectors.append(point.vector)

        ids = np.array(ids, dtype=np.int64)
        vectors = np.array(vectors, dtype=np.float32)
        self.index.add_with_ids(vectors, ids)
        cprint(f"Added {len(data)} vectors to index {self.index_name}.", "green")

    def save_index(self) -> None:
        """
        Saves the FAISS index to a file.
        """
        try:
            chunk = faiss.serialize_index(self.index)
            with open(f"{self.index_name}.pkl", "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"Index {self.index_name} saved successfully.", "green")
        except Exception as e:
            cprint(f"Error saving index: {e}", "red")
            raise

    def load_index(self) -> faiss.Index:
        """
        Loads the FAISS index from file using the instance's index_name.

        Returns:
            faiss.Index: The loaded FAISS index.
        """
        index_file = f"{self.index_name}.pkl"
        try:
            with open(index_file, "rb") as f:
                index = faiss.deserialize_index(pickle.load(f))
            return index
        except FileNotFoundError:
            cprint(f"Index file {index_file} not found.", "red")
            raise
        except Exception as e:
            cprint(f"Error loading index from {index_file}: {e}", "red")
            raise

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

    def search_index(self, query: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """
        Searches the FAISS index for the nearest neighbors of a query vector.
        If using FastEmbed, generates embeddings for the query.

        Args:
            query (np.ndarray): The query vector.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing the distances and IDs of the nearest neighbors.
        """
        if self.use_fastembed and self.embedding_model:
             try:
                 query_vector = self.embedding_model.embed(query)[0]
                 query = np.array([query_vector], dtype=np.float32)
             except Exception as e:
                 cprint(f"FastEmbed embedding generation failed for query: {e}", "red")
                 raise
        D, I = self.index.search(query, k)
        # Convert numpy types to standard Python types
        distances = [float(d) for d in D[0]]
        ids = [int(i) for i in I[0]]
        return distances, ids

    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """
        Retrieves data from the SQLite database based on a list of IDs.

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
        cprint("Initialize FqlDb with FAISS:","green")
        cprint("  fql_db = FqlDb(index_name='my_index', dimension=128, db_name='my_db.db')","white")
        cprint("Initialize FqlDb with FastEmbed:","green")
        cprint("  fql_db = FqlDb(index_name='my_index', db_name='my_db.db', use_fastembed=True, fastembed_model_name='BAAI/bge-small-en-v1.5')","white")
        cprint("Add data to the index:","green")
        cprint("  fql_db.add_to_index(data=[IndexData(...)])","white")
        cprint("Store data to the database:","green")
        cprint("  fql_db.store_to_db(data=[IndexData(...)])","white")
        cprint("Search the index:","green")
        cprint("  distances, ids = fql_db.search_index(query=np.array([1.0, 2.0], dtype=np.float32), k=3)","white")
        cprint("  # OR, if using FastEmbed, pass the query text:","white")
        cprint("  distances, ids = fql_db.search_index(query='search query', k=3)","white")
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
    Tests the FqlDb class with both FAISS and FastEmbed.
    """
    cprint("Starting FqlDb tests...", "blue")

    # Test setup
    index_name = "test_index"
    dimension = 2
    db_name = "test.db"
    fastembed_model_name = "BAAI/bge-small-en-v1.5"

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

        # --- Test FAISS ---
        cprint("\nTesting FAISS...", "green")
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
        cprint(f"FAISS Search results - Distances: {distances}, IDs: {ids}", "cyan")

        retrieved_data = fql_db.retrieve(ids)
        assert len(retrieved_data) == 2, f"Expected 2 results, got {len(retrieved_data)}. Data: {retrieved_data}"
        retrieved_ids = sorted([row[0] for row in retrieved_data]) # Sort retrieved IDs
        assert retrieved_ids == ids, f"Expected IDs {ids}, got {retrieved_ids}"
        cprint(f"FAISS Retrieved data: {retrieved_data}", "cyan")


        # Test save_index and load_index
        fql_db.save_index()
        # Close the current connection before loading a new instance
        if fql_db.connection:
            fql_db.connection.close()
            fql_db.connection = None # Prevent __del__ from trying to close again

        loaded_fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name)  # Load index
        # Verify loaded index has data
        assert loaded_fql_db.index.ntotal == len(test_data)
        cprint("FAISS Index loaded successfully after save.", "green")

        # Test search and retrieve with loaded index
        distances_loaded, ids_loaded = loaded_fql_db.search_index(query_vector, k=2)
        ids_loaded.sort() # Sort for consistent assertion
        assert ids_loaded == ids
        retrieved_data_loaded = loaded_fql_db.retrieve(ids_loaded)
        assert len(retrieved_data_loaded) == 2
        retrieved_ids_loaded = sorted([row[0] for row in retrieved_data_loaded])
        assert retrieved_ids_loaded == ids_loaded
        # --- End Test FAISS ---

        # --- Test FastEmbed ---
        cprint("\nTesting FastEmbed...", "green")
        # Create a new set of test data without pre-defined vectors
        test_data_fastembed = [
            IndexData(vector=None, id=3, content="FastEmbed test content 1", metadata={"key3": "value3"}),
            IndexData(vector=None, id=4, content="FastEmbed test content 2", metadata={"key4": "value4"}),
        ]
        # Clean up old database
        if loaded_fql_db and loaded_fql_db.connection:
            loaded_fql_db.connection.close()
            loaded_fql_db.connection = None
        del loaded_fql_db
        cleanup_test_files(index_name, db_name)

        fql_db_fastembed = FqlDb(index_name=index_name, dimension=0, db_name=db_name, use_fastembed=True, fastembed_model_name=fastembed_model_name)

        fql_db_fastembed.add_to_index(test_data_fastembed)
        fql_db_fastembed.store_to_db(test_data_fastembed)

        # Search with text query
        query_text = "FastEmbed query"
        distances_fastembed, ids_fastembed = fql_db_fastembed.search_index(query_text, k=2)
        assert len(distances_fastembed) == 2
        assert len(ids_fastembed) == 2
        ids_fastembed.sort()
        assert ids_fastembed == [3, 4]
        cprint(f"FastEmbed Search results - Distances: {distances_fastembed}, IDs: {ids_fastembed}", "cyan")

        retrieved_data_fastembed = fql_db_fastembed.retrieve(ids_fastembed)
        assert len(retrieved_data_fastembed) == 2
        retrieved_ids_fastembed = sorted([row[0] for row in retrieved_data_fastembed])
        assert retrieved_ids_fastembed == ids_fastembed
        cprint(f"FastEmbed Retrieved data: {retrieved_data_fastembed}", "cyan")

        # Save and Load FastEmbed Index
        fql_db_fastembed.save_index()
         # Close the current connection before loading a new instance
        if fql_db_fastembed.connection:
            fql_db_fastembed.connection.close()
            fql_db_fastembed.connection = None # Prevent __del__ from trying to close again

        loaded_fql_db_fastembed = FqlDb(index_name=index_name, dimension=0, db_name=db_name, use_fastembed=True, fastembed_model_name=fastembed_model_name)
        assert loaded_fql_db_fastembed.index.ntotal == len(test_data_fastembed)
        cprint("FastEmbed Index loaded successfully after save.", "green")

        distances_loaded, ids_loaded = loaded_fql_db_fastembed.search_index(query_text, k=2)
        ids_loaded.sort()
        assert ids_loaded == ids_fastembed
        retrieved_data_loaded = loaded_fql_db_fastembed.retrieve(ids_loaded)
        assert len(retrieved_data_loaded) == 2
        retrieved_ids_loaded = sorted([row[0] for row in retrieved_data_loaded])
        assert retrieved_ids_loaded == ids_loaded

        # --- End Test FastEmbed ---

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
        if hasattr(fql_db_fastembed, 'connection') and fql_db_fastembed.connection:
            fql_db_fastembed.connection.close()
            fql_db_fastembed.connection = None
        if hasattr(loaded_fql_db_fastembed, 'connection') and loaded_fql_db_fastembed.connection:
            loaded_fql_db_fastembed.connection.close()
            loaded_fql_db_fastembed.connection = None
        # Explicitly delete objects before cleanup to trigger __del__ if needed (though connection is now None)
        del fql_db
        del loaded_fql_db
        if 'fql_db_fastembed' in locals():
            del fql_db_fastembed
        if 'loaded_fql_db_fastembed' in locals():
            del loaded_fql_db_fastembed
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
