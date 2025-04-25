import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Iterable
from data_schema import IndexData
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse

# Attempt to import fastembed conditionally
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

class FqlDb:
    """
    A class combining vector similarity search (FAISS) with a SQLite database.
    Can optionally use fastembed for automatic text embedding generation.
    """

    def __init__(self, index_name: str, db_name: str = "fql.db",
                 use_fastembed: bool = False, embedding_model_name: Optional[str] = None,
                 dimension: Optional[int] = None):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name for the index file (e.g., 'my_index').
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            use_fastembed (bool, optional): If True, use fastembed for embedding generation. Defaults to False.
            embedding_model_name (Optional[str], optional): The name of the fastembed model to use (required if use_fastembed is True). Defaults to None.
            dimension (Optional[int], optional): The dimension of the vectors. Required if use_fastembed is False. If use_fastembed is True, this is inferred from the model. Defaults to None.

        Raises:
            ValueError: If required arguments are missing based on use_fastembed setting.
            ImportError: If use_fastembed is True but fastembed is not installed.
            Exception: If the fastembed model fails to initialize.
        """
        self.index_name = index_name
        self.db_name = db_name
        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self.embedding_model: Optional['TextEmbedding'] = None
        self.use_fastembed = use_fastembed

        if self.use_fastembed:
            if not FASTEMBED_AVAILABLE:
                cprint("fastembed is not installed. Please install it using 'pip install fastembed' or 'pip install fastembed-gpu'.", "red", attrs=["bold"])
                raise ImportError("fastembed package not found, but use_fastembed=True.")
            if not embedding_model_name:
                cprint("Error: embedding_model_name must be provided when use_fastembed is True.", "red", attrs=["bold"])
                raise ValueError("embedding_model_name must be provided when use_fastembed is True.")
            try:
                # Initialize fastembed model
                self.embedding_model = TextEmbedding(embedding_model_name)
                derived_dimension = self.embedding_model.dim
                if dimension is not None and dimension != derived_dimension:
                    cprint(f"Warning: Provided dimension {dimension} conflicts with fastembed model '{embedding_model_name}' dimension {derived_dimension}. Using model dimension.", "yellow")
                self.dimension = derived_dimension
                cprint(f"Initialized with fastembed model '{embedding_model_name}' (Dimension: {self.dimension}).", "green")
            except Exception as e:
                cprint(f"Error: Failed to initialize fastembed model '{embedding_model_name}': {e}", "red", attrs=["bold"])
                raise
        else:
            if dimension is None:
                cprint("Error: dimension must be provided when use_fastembed is False.", "red", attrs=["bold"])
                raise ValueError("dimension must be provided when use_fastembed is False.")
            self.dimension = dimension
            cprint(f"Initialized for manual vector management (Dimension: {self.dimension}).", "green")

        # Load or build the FAISS index using the determined dimension
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> faiss.IndexIDMap2:
        """
        Loads the FAISS index from file if it exists, otherwise builds a new index.
        Uses self.dimension which is set during __init__.

        Returns:
            faiss.IndexIDMap2: The loaded or newly built FAISS index.
        """
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file):
            try:
                self.index = self.load_index() # Uses self.index_name
                # Verify dimension compatibility
                if hasattr(self.index, 'd') and self.index.d != self.dimension:
                     cprint(f"Error: Loaded index dimension ({self.index.d}) does not match required dimension ({self.dimension}). Rebuilding index.", "red", attrs=["bold"])
                     self.index = self.build_index(self.dimension)
                     cprint(f"New index {self.index_name} created successfully.", "green")
                else:
                    cprint(f"Index '{self.index_name}' loaded successfully (Vectors: {self.index.ntotal}, Dimension: {self.index.d}).", "green")

            except Exception as e:
                cprint(f"Warning: Failed to load index '{self.index_name}': {e}. Building a new one.", "yellow")
                self.index = self.build_index(self.dimension)
                cprint(f"New index '{self.index_name}' created successfully.", "green")
        else:
            cprint(f"Index file '{index_file}' not found. Building a new one.", "yellow")
            self.index = self.build_index(self.dimension)
            cprint(f"New index '{self.index_name}' created successfully.", "green")
        return self.index

    def build_index(self, dimension: int) -> faiss.IndexIDMap2:
        """
        Builds a new FAISS index (IndexFlatL2 wrapped in IndexIDMap2).

        Args:
            dimension (int): The dimension of the vectors.

        Returns:
            faiss.IndexIDMap2: The newly created FAISS index.
        """
        try:
            flat_index = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIDMap2(flat_index)
            cprint(f"Built new FAISS index with dimension {dimension}.", "cyan")
            return index
        except Exception as e:
            cprint(f"Error building FAISS index: {e}", "red", attrs=["bold"])
            raise

    def add_to_index(self, data: List[IndexData]) -> None:
        """
        Adds pre-computed vector data to the FAISS index.

        Args:
            data (List[IndexData]): A list of IndexData objects containing vectors and IDs.

        Raises:
            ValueError: If input data is invalid or dimensions mismatch.
        """
        if not data:
            cprint("No data provided to add to index.", "yellow")
            return

        ids = []
        vectors = []
        try:
            for point in data:
                if point.vector.shape[0] != self.dimension:
                    raise ValueError(f"Vector dimension mismatch for ID {point.id}. Expected {self.dimension}, got {point.vector.shape[0]}.")
                ids.append(point.id)
                vectors.append(point.vector)

            ids_np = np.array(ids, dtype=np.int64)
            vectors_np = np.array(vectors, dtype=np.float32)

            self.index.add_with_ids(vectors_np, ids_np)
            cprint(f"Added {len(data)} vectors to index '{self.index_name}'. Total vectors: {self.index.ntotal}.", "green")
        except ValueError as ve:
             cprint(f"Error adding data to index: {ve}", "red", attrs=["bold"])
             raise
        except Exception as e:
            cprint(f"Unexpected error adding data to index: {e}", "red", attrs=["bold"])
            raise

    def add_texts(self, texts: List[str], metadata_list: Optional[List[Dict]] = None, ids: Optional[List[int]] = None) -> List[int]:
        """
        Generates embeddings for texts using fastembed (if configured) and adds them to the index and database.

        Args:
            texts (List[str]): The list of text documents to add.
            metadata_list (Optional[List[Dict]], optional): A list of metadata dictionaries corresponding to each text. Defaults to None.
            ids (Optional[List[int]], optional): A list of integer IDs for each text. If None, sequential IDs starting from the current index size will be generated. Defaults to None.

        Returns:
            List[int]: The list of IDs assigned to the added documents.

        Raises:
            RuntimeError: If FqlDb was not initialized with use_fastembed=True.
            ValueError: If the lengths of texts, metadata_list (if provided), and ids (if provided) do not match.
        """
        if not self.use_fastembed or self.embedding_model is None:
            cprint("Error: add_texts requires FqlDb to be initialized with use_fastembed=True and a valid embedding_model_name.", "red", attrs=["bold"])
            raise RuntimeError("Fastembed is not configured for this FqlDb instance.")
        if not texts:
            cprint("No texts provided to add.", "yellow")
            return []

        num_texts = len(texts)
        if metadata_list is not None and len(metadata_list) != num_texts:
            raise ValueError("Length mismatch: 'texts' and 'metadata_list' must have the same number of elements.")
        if ids is not None and len(ids) != num_texts:
             raise ValueError("Length mismatch: 'texts' and 'ids' must have the same number of elements.")

        # Generate IDs if not provided
        if ids is None:
            start_id = self.index.ntotal
            generated_ids = list(range(start_id, start_id + num_texts))
            cprint(f"Generating sequential IDs starting from {start_id}.", "cyan")
        else:
            generated_ids = ids

        # Generate embeddings
        cprint(f"Generating embeddings for {num_texts} texts using '{self.embedding_model.model_name}'...", "cyan")
        try:
            embeddings: Iterable[np.ndarray] = self.embedding_model.embed(texts)
        except Exception as e:
            cprint(f"Error during embedding generation: {e}", "red", attrs=["bold"])
            raise

        # Prepare data for indexing and storage
        index_data_list: List[IndexData] = []
        for i, (text, embedding) in enumerate(embeddings):
            doc_id = generated_ids[i]
            metadata = metadata_list[i] if metadata_list else {}
            index_data_list.append(IndexData(
                vector=embedding,
                id=doc_id,
                content=text,
                metadata=metadata
            ))

        # Add to FAISS index and SQLite DB
        self.add_to_index(index_data_list)
        self.store_to_db(index_data_list)

        return generated_ids


    def save_index(self) -> None:
        """
        Saves the current FAISS index to a pickle file named after the index_name.
        """
        index_file = f"{self.index_name}.pkl"
        try:
            chunk = faiss.serialize_index(self.index)
            with open(index_file, "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"Index '{self.index_name}' saved successfully to '{index_file}'.", "green")
        except Exception as e:
            cprint(f"Error saving index '{self.index_name}' to '{index_file}': {e}", "red", attrs=["bold"])
            raise

    def load_index(self) -> faiss.IndexIDMap2:
        """
        Loads the FAISS index from a pickle file named after the instance's index_name.

        Returns:
            faiss.IndexIDMap2: The loaded FAISS index.

        Raises:
            FileNotFoundError: If the index file does not exist.
            Exception: For other errors during loading or deserialization.
        """
        index_file = f"{self.index_name}.pkl"
        if not os.path.exists(index_file):
             cprint(f"Error: Index file '{index_file}' not found.", "red", attrs=["bold"])
             raise FileNotFoundError(f"Index file '{index_file}' not found.")
        try:
            with open(index_file, "rb") as f:
                index_data = pickle.load(f)
                index = faiss.deserialize_index(index_data)
                # Ensure it's the correct type, although deserialize_index should handle it
                if not isinstance(index, faiss.IndexIDMap2):
                     # This case might indicate a corrupted file or incorrect saving method
                     cprint(f"Warning: Loaded index from '{index_file}' is not of type IndexIDMap2. Type: {type(index)}", "yellow")
                     # Attempt to wrap if it's just the flat index? Risky. Better to raise.
                     raise TypeError("Loaded index is not of the expected type IndexIDMap2.")
            cprint(f"Index '{self.index_name}' loaded from '{index_file}'.", "cyan")
            return index
        except pickle.UnpicklingError as pe:
            cprint(f"Error deserializing index from '{index_file}': Possible file corruption. {pe}", "red", attrs=["bold"])
            raise
        except Exception as e:
            cprint(f"Error loading index '{self.index_name}' from '{index_file}': {e}", "red", attrs=["bold"])
            raise

    def store_to_db(self, data: List[IndexData]) -> None:
        """
        Stores or updates data (ID, content, metadata) in the SQLite database table named after the index_name.

        Args:
            data (List[IndexData]): A list of IndexData objects to store.
        """
        if not data:
            cprint("No data provided to store in database.", "yellow")
            return

        try:
            values = []
            for point in data:
                # Ensure metadata is stored as a string (e.g., JSON)
                metadata_str = str(point.metadata) # Simple string conversion, consider JSON for robustness
                values.append((point.id, point.content, metadata_str))

            with closing(self.connection.cursor()) as cur:
                # Create table if it doesn't exist
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(
                        id INTEGER PRIMARY KEY,
                        content TEXT,
                        metadata TEXT
                    )"""
                )
                # Insert or replace data based on primary key (id)
                cur.executemany(
                    f"""INSERT OR REPLACE INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""", values
                )
            # isolation_level=None enables autocommit
            cprint(f"Stored/updated {len(data)} records in database table '{self.index_name}'.", "green")

        except sqlite3.Error as e:
            cprint(f"SQLite error during store_to_db for table '{self.index_name}': {e}", "red", attrs=["bold"])
            raise
        except Exception as e:
            cprint(f"Unexpected error during store_to_db for table '{self.index_name}': {e}", "red", attrs=["bold"])
            raise

    def search_index(self, query_vector: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """
        Searches the FAISS index for the nearest neighbors of a query vector.

        Args:
            query_vector (np.ndarray): The query vector (should be 2D, e.g., np.array([[1.0, 2.0]])).
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing lists of distances and IDs of the nearest neighbors.

        Raises:
            ValueError: If the query vector dimension doesn't match the index dimension.
        """
        if query_vector.shape[1] != self.dimension:
             cprint(f"Error: Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.dimension}).", "red", attrs=["bold"])
             raise ValueError("Query vector dimension mismatch.")
        if query_vector.ndim != 2:
             cprint(f"Error: Query vector should be 2D (e.g., np.array([[1.0, 2.0]])). Got shape {query_vector.shape}", "red", attrs=["bold"])
             raise ValueError("Query vector must be 2-dimensional.")

        try:
            # Adjust k if the index contains fewer vectors than requested
            actual_k = min(k, self.index.ntotal)
            if actual_k == 0:
                cprint("Index is empty, cannot perform search.", "yellow")
                return [], []
            if actual_k < k:
                 cprint(f"Warning: Requested k={k} neighbors, but index only contains {self.index.ntotal}. Returning {actual_k}.", "yellow")


            D, I = self.index.search(query_vector, actual_k)

            # FAISS returns -1 for IDs if fewer than k neighbors are found (shouldn't happen with IndexIDMap2 unless k > ntotal)
            # Filter out invalid IDs and corresponding distances if necessary
            valid_indices = [idx for idx, i in enumerate(I[0]) if i != -1]
            distances = [float(D[0][idx]) for idx in valid_indices]
            ids = [int(I[0][idx]) for idx in valid_indices]

            cprint(f"Search completed. Found {len(ids)} neighbors.", "cyan")
            return distances, ids
        except Exception as e:
            cprint(f"Error during FAISS search: {e}", "red", attrs=["bold"])
            raise

    def search_text(self, query_text: str, k: int = 3) -> Tuple[List[float], List[int]]:
        """
        Generates an embedding for the query text using fastembed (if configured) and searches the index.

        Args:
            query_text (str): The text to search for.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing lists of distances and IDs of the nearest neighbors.

        Raises:
            RuntimeError: If FqlDb was not initialized with use_fastembed=True.
        """
        if not self.use_fastembed or self.embedding_model is None:
            cprint("Error: search_text requires FqlDb to be initialized with use_fastembed=True.", "red", attrs=["bold"])
            raise RuntimeError("Fastembed is not configured for this FqlDb instance.")

        cprint(f"Generating embedding for query: '{query_text}'...", "cyan")
        try:
            # fastembed embed returns a generator, get the first (and only) item
            query_embedding_list = list(self.embedding_model.embed([query_text]))
            if not query_embedding_list:
                 cprint("Error: Failed to generate embedding for the query text.", "red", attrs=["bold"])
                 raise ValueError("Embedding generation failed.")
            query_vector = query_embedding_list[0]
            # Ensure the vector is 2D for FAISS search
            query_vector_2d = np.array([query_vector], dtype=np.float32)
        except Exception as e:
            cprint(f"Error generating query embedding: {e}", "red", attrs=["bold"])
            raise

        return self.search_index(query_vector_2d, k)


    def retrieve(self, ids: List[int]) -> List[Dict]:
        """
        Retrieves data (ID, content, metadata) from the SQLite database based on a list of IDs.

        Args:
            ids (List[int]): A list of integer IDs to retrieve.

        Returns:
            List[Dict]: A list of dictionaries, each containing 'id', 'content', and 'metadata' for a retrieved record. Returns empty list if no IDs are provided or found.
        """
        if not ids:
            cprint("No IDs provided for retrieval.", "yellow")
            return []
        # Ensure IDs are integers
        try:
            int_ids = [int(i) for i in ids]
        except (ValueError, TypeError) as e:
             cprint(f"Error: IDs must be integers. Received: {ids}. Error: {e}", "red", attrs=["bold"])
             raise TypeError("All elements in the 'ids' list must be integers.")


        results = []
        try:
            with closing(self.connection.cursor()) as cur:
                placeholders = ','.join('?' * len(int_ids))
                sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"
                cur.execute(sql, int_ids)
                rows = cur.fetchall()

                # Convert rows to dictionaries
                for row in rows:
                    try:
                        # Attempt to parse metadata back into a dict (assuming it was stored as JSON or similar)
                        # Use a simple eval for now, but json.loads is safer if stored as JSON
                        metadata_dict = eval(row[2]) if row[2] else {}
                    except:
                        metadata_dict = {"raw_metadata": row[2]} # Fallback if eval fails

                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": metadata_dict
                    })
                cprint(f"Retrieved {len(results)} records from database table '{self.index_name}'.", "cyan")

        except sqlite3.Error as e:
            cprint(f"SQLite error during retrieve from table '{self.index_name}': {e}", "red", attrs=["bold"])
            raise
        except Exception as e:
             cprint(f"An unexpected error occurred during retrieve: {e}", "red", attrs=["bold"])
             raise

        # Return results in the order of the input IDs if possible
        id_map = {item["id"]: item for item in results}
        ordered_results = [id_map[i] for i in int_ids if i in id_map]

        return ordered_results

    def retrieve_by_text(self, query_text: str, k: int = 3) -> List[Dict]:
        """
        Searches for the given text and retrieves the corresponding full documents from the database.

        Args:
            query_text (str): The text to search for.
            k (int, optional): The number of results to retrieve. Defaults to 3.

        Returns:
            List[Dict]: A list of dictionaries containing the retrieved document data ('id', 'content', 'metadata'), ordered by relevance.
        """
        try:
            distances, ids = self.search_text(query_text, k)
            if not ids:
                return []
            retrieved_data = self.retrieve(ids)
            # Add distance/score? For now, just return data ordered by search result.
            return retrieved_data
        except Exception as e:
            cprint(f"Error during retrieve_by_text: {e}", "red", attrs=["bold"])
            # Decide whether to return empty list or re-raise
            return []


    def __del__(self):
         """
         Closes the database connection when the FqlDb object is garbage collected.
         """
         if hasattr(self, "connection") and self.connection:
             try:
                 self.connection.close()
                 # cprint("Database connection closed.", "yellow") # Maybe too verbose for __del__
             except Exception as e:
                  cprint(f"Warning: Error closing database connection in __del__: {e}", "yellow")

    def usage(self):
        """Prints usage instructions for the FqlDb class."""
        cprint("\n--- FqlDb Usage Instructions ---", "blue", attrs=["bold"])

        cprint("\n1. Initialization:", "green", attrs=["bold"])
        cprint("  a) With manual vector management:", "white")
        cprint("     fql_db = FqlDb(index_name='my_manual_index', dimension=128, db_name='my_db.db')", "cyan")
        cprint("  b) With fastembed for automatic embeddings:", "white")
        cprint("     # Make sure fastembed is installed: pip install fastembed", "grey")
        cprint("     fql_db_fast = FqlDb(index_name='my_fastembed_index', use_fastembed=True, embedding_model_name='BAAI/bge-small-en-v1.5', db_name='my_db.db')", "cyan")

        cprint("\n2. Adding Data:", "green", attrs=["bold"])
        cprint("  a) Adding pre-computed vectors (manual mode):", "white")
        cprint("     from data_schema import IndexData", "grey")
        cprint("     data = [IndexData(vector=np.array([1.0]*128, dtype=np.float32), id=1, content='Doc 1', metadata={'topic': 'A'})]", "grey")
        cprint("     fql_db.add_to_index(data)", "cyan")
        cprint("     fql_db.store_to_db(data) # Store content/metadata separately", "cyan")
        cprint("  b) Adding texts (fastembed mode):", "white")
        cprint("     texts = ['This is document one.', 'Another document here.']", "grey")
        cprint("     metadata = [{'source': 'web'}, {'source': 'pdf'}] # Optional", "grey")
        cprint("     added_ids = fql_db_fast.add_texts(texts, metadata_list=metadata)", "cyan")
        cprint("     print(f'Added document IDs: {added_ids}')", "grey")


        cprint("\n3. Searching:", "green", attrs=["bold"])
        cprint("  a) Searching with a pre-computed vector (manual mode):", "white")
        cprint("     query_vec = np.array([[0.5]*128], dtype=np.float32)", "grey")
        cprint("     distances, ids = fql_db.search_index(query=query_vec, k=3)", "cyan")
        cprint("     print(f'Search results (manual): IDs={ids}, Distances={distances}')", "grey")
        cprint("  b) Searching with text (fastembed mode):", "white")
        cprint("     query = 'Tell me about document one'", "grey")
        cprint("     distances_fast, ids_fast = fql_db_fast.search_text(query_text=query, k=3)", "cyan")
        cprint("     print(f'Search results (fastembed): IDs={ids_fast}, Distances={distances_fast}')", "grey")

        cprint("\n4. Retrieving Full Data:", "green", attrs=["bold"])
        cprint("  a) Retrieving by IDs (works in both modes):", "white")
        cprint("     retrieved_data = fql_db.retrieve(ids=ids) # Use IDs from search_index", "cyan")
        cprint("     print(f'Retrieved data: {retrieved_data}')", "grey")
        cprint("  b) Searching and Retrieving by text (fastembed mode):", "white")
        cprint("     retrieved_by_text = fql_db_fast.retrieve_by_text(query_text=query, k=3)", "cyan")
        cprint("     print(f'Retrieved by text: {retrieved_by_text}')", "grey")

        cprint("\n5. Saving the Index:", "green", attrs=["bold"])
        cprint("  fql_db.save_index() # Saves the FAISS index to 'index_name.pkl'", "cyan")

        cprint("\n6. Loading an Index:", "green", attrs=["bold"])
        cprint("  # When initializing, FqlDb automatically loads if 'index_name.pkl' exists.", "grey")
        cprint("  loaded_db = FqlDb(index_name='my_manual_index', dimension=128) # Will load 'my_manual_index.pkl'", "cyan")
        cprint("  loaded_db_fast = FqlDb(index_name='my_fastembed_index', use_fastembed=True, embedding_model_name='BAAI/bge-small-en-v1.5')", "cyan")

        cprint("\n--- End Usage ---", "blue", attrs=["bold"])


def cleanup_test_files(index_name: str, db_name: str):
    """Removes test index and database files."""
    index_file = f"{index_name}.pkl"
    db_file = db_name
    files_removed = False
    try:
        if os.path.exists(index_file):
            os.remove(index_file)
            cprint(f"Removed test index file: {index_file}", "magenta")
            files_removed = True
        if os.path.exists(db_file):
            os.remove(db_file)
            cprint(f"Removed test database file: {db_file}", "magenta")
            files_removed = True
        # if files_removed:
        #     cprint("Test files removed.", "yellow")
    except Exception as e:
        cprint(f"Error during test file cleanup: {e}", "red")


def test_fql_db():
    """
    Tests the FqlDb class functionality, including manual and fastembed modes.
    """
    cprint("\n--- Starting FqlDb Tests ---", "blue", attrs=["bold"])

    # Test setup parameters
    manual_index_name = "test_manual_index"
    fastembed_index_name = "test_fastembed_index"
    dimension = 2 # Low dimension for manual test
    fastembed_model = 'BAAI/bge-small-en-v1.5' # Use a known small model
    db_name = "test_fql.db"

    # --- Ensure clean state before all tests ---
    cleanup_test_files(manual_index_name, db_name)
    cleanup_test_files(fastembed_index_name, db_name) # Clean fastembed files too
    # -------------------------------------------

    manual_db = None
    fast_db = None
    loaded_manual_db = None
    loaded_fast_db = None

    try:
        # == Test 1: Manual Vector Management ==
        cprint("\n[Test 1: Manual Vector Management]", "yellow", attrs=["bold"])
        manual_data = [
            IndexData(vector=np.array([1.0, 2.0], dtype=np.float32), id=101, content="Manual content 1", metadata={"source": "manual"}),
            IndexData(vector=np.array([3.0, 4.0], dtype=np.float32), id=102, content="Manual content 2", metadata={"source": "manual"}),
            IndexData(vector=np.array([1.5, 2.5], dtype=np.float32), id=103, content="Manual content 3", metadata={"source": "manual"}),
        ]
        manual_db = FqlDb(index_name=manual_index_name, dimension=dimension, db_name=db_name)
        manual_db.add_to_index(manual_data)
        manual_db.store_to_db(manual_data)

        query_vector_manual = np.array([[1.8, 2.8]], dtype=np.float32)
        distances, ids = manual_db.search_index(query_vector_manual, k=2)
        assert len(ids) == 2, f"Manual search expected 2 IDs, got {len(ids)}"
        # The closest should be ID 103 ([1.5, 2.5]), then 101 ([1.0, 2.0])
        assert 103 in ids, "Manual search missing ID 103"
        assert 101 in ids, "Manual search missing ID 101"
        cprint(f"Manual Search results - IDs: {ids}, Distances: {distances}", "cyan")

        retrieved_data = manual_db.retrieve(ids)
        assert len(retrieved_data) == 2, f"Manual retrieve expected 2 results, got {len(retrieved_data)}"
        retrieved_ids = sorted([item['id'] for item in retrieved_data])
        assert retrieved_ids == sorted(ids), f"Manual retrieve IDs mismatch. Expected {sorted(ids)}, got {retrieved_ids}"
        cprint(f"Manual Retrieved data: {retrieved_data}", "cyan")

        manual_db.save_index()
        # Close connection before loading
        if manual_db.connection: manual_db.connection.close(); manual_db.connection = None

        loaded_manual_db = FqlDb(index_name=manual_index_name, dimension=dimension, db_name=db_name)
        assert loaded_manual_db.index.ntotal == len(manual_data), "Loaded manual index count mismatch."
        cprint("Manual index saved and loaded successfully.", "green")
        # Test search on loaded index
        _, ids_loaded = loaded_manual_db.search_index(query_vector_manual, k=2)
        assert sorted(ids_loaded) == sorted(ids), "Search results differ after loading manual index."
        cprint("Search on loaded manual index successful.", "green")
        cprint("[Test 1 Passed]", "green", attrs=["bold"])

        # == Test 2: Fastembed Integration ==
        cprint("\n[Test 2: Fastembed Integration]", "yellow", attrs=["bold"])
        if not FASTEMBED_AVAILABLE:
            cprint("Skipping Fastembed tests as fastembed package is not installed.", "yellow")
        else:
            try:
                fast_db = FqlDb(index_name=fastembed_index_name, use_fastembed=True, embedding_model_name=fastembed_model, db_name=db_name)
                fastembed_dim = fast_db.dimension # Get actual dimension
            except Exception as e:
                 cprint(f"Could not initialize fastembed FqlDb: {e}. Skipping fastembed tests.", "red")
                 fast_db = None # Ensure fast_db is None if init fails

            if fast_db:
                texts_to_add = ["The quick brown fox", "jumps over the lazy dog", "Fastembed is fast"]
                metadata_to_add = [{"topic": "animals"}, {"topic": "animals"}, {"topic": "software"}]
                added_ids = fast_db.add_texts(texts_to_add, metadata_list=metadata_to_add)
                assert len(added_ids) == 3, f"add_texts expected 3 IDs, got {len(added_ids)}"
                assert fast_db.index.ntotal == 3, f"Fastembed index should have 3 vectors, has {fast_db.index.ntotal}"
                cprint(f"Added texts with IDs: {added_ids}", "cyan")

                # Test search_text
                query_text = "How fast is fastembed?"
                distances_fast, ids_fast = fast_db.search_text(query_text, k=2)
                assert len(ids_fast) == 2, f"search_text expected 2 IDs, got {len(ids_fast)}"
                # Expect ID 2 ('Fastembed is fast') to be the top result
                assert added_ids[2] in ids_fast, f"search_text results missing expected ID for '{texts_to_add[2]}'"
                # Check if the top result is indeed ID 2
                assert ids_fast[0] == added_ids[2], f"search_text top result mismatch. Expected ID {added_ids[2]}, got {ids_fast[0]}"
                cprint(f"Fastembed Search results - IDs: {ids_fast}, Distances: {distances_fast}", "cyan")

                # Test retrieve_by_text
                retrieved_by_text = fast_db.retrieve_by_text(query_text, k=2)
                assert len(retrieved_by_text) == 2, f"retrieve_by_text expected 2 results, got {len(retrieved_by_text)}"
                assert retrieved_by_text[0]['id'] == added_ids[2], "retrieve_by_text top result mismatch."
                assert retrieved_by_text[0]['content'] == texts_to_add[2], "retrieve_by_text content mismatch."
                assert retrieved_by_text[0]['metadata'] == metadata_to_add[2], "retrieve_by_text metadata mismatch."
                cprint(f"Fastembed Retrieved by text: {retrieved_by_text}", "cyan")

                fast_db.save_index()
                # Close connection before loading
                if fast_db.connection: fast_db.connection.close(); fast_db.connection = None

                loaded_fast_db = FqlDb(index_name=fastembed_index_name, use_fastembed=True, embedding_model_name=fastembed_model, db_name=db_name)
                assert loaded_fast_db.index.ntotal == len(texts_to_add), "Loaded fastembed index count mismatch."
                assert loaded_fast_db.dimension == fastembed_dim, "Loaded fastembed index dimension mismatch."
                cprint("Fastembed index saved and loaded successfully.", "green")

                # Test search on loaded index
                _, ids_loaded_fast = loaded_fast_db.search_text(query_text, k=2)
                assert sorted(ids_loaded_fast) == sorted(ids_fast), "Search results differ after loading fastembed index."
                cprint("Search on loaded fastembed index successful.", "green")
                cprint("[Test 2 Passed]", "green", attrs=["bold"])


        # == Test 3: Error Handling ==
        cprint("\n[Test 3: Error Handling]", "yellow", attrs=["bold"])
        try:
            FqlDb(index_name="error_test", dimension=None, use_fastembed=False)
            assert False, "ValueError not raised for missing dimension in manual mode"
        except ValueError as e:
            cprint(f"Caught expected error: {e}", "green")

        if FASTEMBED_AVAILABLE:
            try:
                FqlDb(index_name="error_test", use_fastembed=True, embedding_model_name=None)
                assert False, "ValueError not raised for missing model name in fastembed mode"
            except ValueError as e:
                cprint(f"Caught expected error: {e}", "green")

            try:
                # Use the manual_db instance which doesn't have fastembed configured
                manual_db_reopened = FqlDb(index_name=manual_index_name, dimension=dimension, db_name=db_name)
                manual_db_reopened.search_text("some query")
                assert False, "RuntimeError not raised when calling search_text in manual mode"
            except RuntimeError as e:
                cprint(f"Caught expected error: {e}", "green")
            finally:
                 if 'manual_db_reopened' in locals() and manual_db_reopened.connection:
                      manual_db_reopened.connection.close()


        cprint("[Test 3 Passed]", "green", attrs=["bold"])


        cprint("\n--- All FqlDb tests passed! ---", "green", attrs=["bold"])

    except Exception as e:
        cprint(f"\n--- Test failed: {e} ---", "red", attrs=["bold"])
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        # raise # Re-raise exception to make test runner aware of failure
    finally:
        # --- Final Cleanup ---
        cprint("\n--- Cleaning up test files ---", "magenta")
        # Ensure connections are closed if objects exist and have connections
        if manual_db and manual_db.connection: manual_db.connection.close()
        if fast_db and fast_db.connection: fast_db.connection.close()
        if loaded_manual_db and loaded_manual_db.connection: loaded_manual_db.connection.close()
        if loaded_fast_db and loaded_fast_db.connection: loaded_fast_db.connection.close()
        # Explicitly delete objects (optional, helps ensure __del__ is called if needed)
        del manual_db, fast_db, loaded_manual_db, loaded_fast_db
        # Cleanup files
        cleanup_test_files(manual_index_name, db_name)
        cleanup_test_files(fastembed_index_name, db_name)
        cprint("--- Cleanup finished ---", "magenta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FqlDb or print usage instructions.")
    parser.add_argument("--usage", action="store_true", help="Print FqlDb usage instructions instead of running tests.")
    args = parser.parse_args()

    if args.usage:
        # Need to instantiate with valid params even for usage
        try:
            # Try fastembed first if available
            if FASTEMBED_AVAILABLE:
                 fql_db_usage = FqlDb(index_name='usage_example', use_fastembed=True, embedding_model_name='BAAI/bge-small-en-v1.5')
            else:
                 fql_db_usage = FqlDb(index_name='usage_example', dimension=10) # Dummy dimension
            fql_db_usage.usage()
            del fql_db_usage
        except Exception as e:
             cprint(f"Could not print usage due to initialization error: {e}", "red")
             cprint("Please ensure required parameters (like dimension or fastembed model) are valid.", "red")
    else:
        test_fql_db()

