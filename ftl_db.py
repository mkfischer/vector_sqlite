import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Any, Generator
from pydantic import BaseModel as PydanticBaseModel, Field, field_validator, ValidationError
from contextlib import closing, contextmanager
import pickle
import os
import logging
from termcolor import cprint
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Schema Definition ---

class IndexData(PydanticBaseModel):
    """
    Represents a single data point for indexing.

    Attributes:
        vector (np.ndarray): The numerical vector representation of the content.
        id (int): A unique identifier for the data point.
        content (str): The original text content.
        metadata (Dict): Optional dictionary for storing additional metadata.
    """
    vector: np.ndarray
    id: int
    content: str
    metadata: Dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @field_validator('vector')
    @classmethod
    def vector_must_be_numpy_array(cls, v):
        if not isinstance(v, np.ndarray):
            raise ValueError("vector must be a numpy array")
        if v.ndim != 1:
            raise ValueError("vector must be a 1D numpy array")
        # Ensure float32 for FAISS compatibility
        return v.astype(np.float32)

    @field_validator('id')
    @classmethod
    def id_must_be_non_negative_int(cls, v):
        if not isinstance(v, int) or v < 0:
            raise ValueError("id must be a non-negative integer")
        return v

# --- Helper Functions ---

def _print_error(message: str, details: Optional[Any] = None):
    """Prints an error message in red."""
    cprint(f"[ERROR] {message}", "red", attrs=["bold"])
    if details:
        cprint(f"  Details: {details}", "red")

def _print_info(message: str):
    """Prints an informational message in blue."""
    cprint(f"[INFO] {message}", "blue")

def _print_success(message: str):
    """Prints a success message in green."""
    cprint(f"[SUCCESS] {message}", "green")

def _grouper(iterable: list, n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

# --- FtlDb Class ---

class FtlDb:
    """
    Manages a FAISS vector index and corresponding metadata storage in SQLite.

    Provides functionalities to add data (generating embeddings if needed),
    search the index, save/load the index, and manage the database connection.
    Includes robust error handling and user feedback.
    """

    def __init__(self,
                 index_name: str,
                 dimension: Optional[int] = None,
                 db_path: Optional[str] = None,
                 index_path: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 rebuild: bool = False):
        """
        Initializes or loads the FtlDb instance.

        Args:
            index_name (str): A unique name for the index and database table.
            dimension (Optional[int]): The dimension of the vectors. Required if creating a new index.
                                      If loading an existing index, it's inferred.
            db_path (Optional[str]): Path to the SQLite database file. Defaults to f"{index_name}.db".
            index_path (Optional[str]): Path to the FAISS index file. Defaults to f"{index_name}.pkl".
            model_name (str): The name of the Sentence Transformer model to use for embedding.
            rebuild (bool): If True, forces creation of a new index and database, potentially overwriting existing ones.
        """
        if not index_name or not isinstance(index_name, str):
            _print_error("Initialization failed: 'index_name' must be a non-empty string.")
            raise ValueError("'index_name' must be a non-empty string.")

        self.index_name = index_name
        self.db_path = db_path or f"{self.index_name}.db"
        self.index_path = index_path or f"{self.index_name}.pkl"
        self.index: Optional[faiss.IndexIDMap2] = None
        self.dimension: Optional[int] = dimension
        self.connection: Optional[sqlite3.Connection] = None
        self._next_id: int = 0

        try:
            _print_info(f"Loading Sentence Transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            inferred_dimension = self.model.get_sentence_embedding_dimension()
            if self.dimension is None:
                self.dimension = inferred_dimension
                _print_info(f"Inferred vector dimension from model: {self.dimension}")
            elif self.dimension != inferred_dimension:
                _print_error(f"Provided dimension ({self.dimension}) does not match model dimension ({inferred_dimension}).")
                raise ValueError("Dimension mismatch between provided value and model.")
        except Exception as e:
            _print_error("Failed to load Sentence Transformer model.", e)
            raise RuntimeError(f"Could not load model '{model_name}': {e}") from e

        if rebuild:
            _print_info(f"Rebuild requested. Removing existing index and DB files if they exist.")
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.db_path):
                os.remove(self.db_path)

        self._load_or_create_index()
        self._connect_db()
        self._update_next_id()

    @contextmanager
    def _db_cursor(self):
        """Provides a database cursor within a context manager."""
        if not self.connection:
            _print_error("Database connection is not established.")
            raise ConnectionError("Database is not connected.")
        cursor = self.connection.cursor()
        try:
            yield cursor
        except sqlite3.Error as e:
            _print_error("Database operation failed.", e)
            self.connection.rollback() # Rollback on error
            raise # Re-raise the exception
        else:
            self.connection.commit() # Commit on success
        finally:
            cursor.close()

    def _connect_db(self):
        """Establishes connection to the SQLite database and creates table if needed."""
        try:
            # Use isolation_level=None for autocommit, managed by context manager
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            sqlite3.register_adapter(np.int64, int) # Adapt numpy int64 for sqlite
            _print_info(f"Connecting to database: {self.db_path}")
            with self._db_cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.index_name} (
                        id INTEGER PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT
                    )""")
            _print_success(f"Database connected and table '{self.index_name}' ensured.")
        except sqlite3.Error as e:
            _print_error("Failed to connect to or initialize the database.", e)
            self.connection = None
            raise ConnectionError(f"Could not connect to database '{self.db_path}': {e}") from e

    def _load_or_create_index(self):
        """Loads an existing FAISS index or creates a new one."""
        if os.path.exists(self.index_path):
            try:
                _print_info(f"Loading existing index from: {self.index_path}")
                with open(self.index_path, "rb") as f:
                    serialized_index = pickle.load(f)
                self.index = faiss.deserialize_index(serialized_index)
                if not isinstance(self.index, faiss.IndexIDMap2):
                     _print_error("Loaded index is not of type IndexIDMap2. Rebuilding.")
                     self._create_new_index()
                else:
                    loaded_dimension = self.index.d
                    if self.dimension is None:
                        self.dimension = loaded_dimension
                        _print_info(f"Inferred dimension from loaded index: {self.dimension}")
                    elif self.dimension != loaded_dimension:
                         _print_error(f"Provided dimension ({self.dimension}) conflicts with loaded index dimension ({loaded_dimension}). Using loaded dimension.")
                         self.dimension = loaded_dimension
                    _print_success(f"Index loaded successfully. Dimension: {self.dimension}, Entries: {self.index.ntotal}")

            except (FileNotFoundError, pickle.UnpicklingError, EOFError, AttributeError, TypeError) as e:
                _print_error(f"Failed to load index from {self.index_path}. A new index will be created.", e)
                self._create_new_index()
            except Exception as e: # Catch other potential FAISS errors
                 _print_error(f"An unexpected error occurred loading the index from {self.index_path}. A new index will be created.", e)
                 self._create_new_index()

        else:
            _print_info(f"No index file found at {self.index_path}. Creating a new index.")
            self._create_new_index()

    def _create_new_index(self):
        """Creates a new FAISS index."""
        if self.dimension is None:
             _print_error("Cannot create index: Vector dimension is not set. Provide 'dimension' during initialization or ensure model loads correctly.")
             raise ValueError("Vector dimension must be specified to create a new index.")
        try:
            _print_info(f"Creating new FAISS index with dimension {self.dimension}.")
            flat_index = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap2(flat_index)
            _print_success("New FAISS index created.")
        except Exception as e:
            _print_error("Failed to create FAISS index.", e)
            raise RuntimeError(f"Could not create FAISS index: {e}") from e

    def _update_next_id(self):
        """ Determines the next available ID based on the database."""
        if not self.connection:
             _print_error("Cannot update next ID: Database not connected.")
             return # Or raise? For now, just return.

        try:
            with self._db_cursor() as cur:
                cur.execute(f"SELECT MAX(id) FROM {self.index_name}")
                max_id = cur.fetchone()[0]
                self._next_id = (max_id + 1) if max_id is not None else 0
                # Also check FAISS index if it has items, though DB should be the source of truth
                if self.index and self.index.ntotal > 0:
                     # This part is tricky as FAISS IDs might not be sequential if items were removed.
                     # Relying on DB MAX(id) is generally safer for sequential additions.
                     pass
            _print_info(f"Next available ID set to: {self._next_id}")
        except sqlite3.Error as e:
            _print_error("Failed to determine next ID from database.", e)
            # Decide on fallback? Maybe assume 0 if table is empty/new?
            self._next_id = 0
            _print_info("Defaulting next available ID to 0 due to DB error.")


    def add(self, data: List[str], batch_size: int = 32):
        """
        Adds text data to the index and database. Generates embeddings automatically.

        Args:
            data (List[str]): A list of text strings to add.
            batch_size (int): The number of items to process in each batch.

        Returns:
            List[int]: A list of IDs assigned to the added data points.

        Raises:
            ValueError: If input data is not a list of strings.
            RuntimeError: If embedding generation or database/index operations fail.
        """
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            _print_error("Input 'data' must be a list of strings.")
            raise ValueError("Input 'data' must be a list of strings.")
        if not self.index or not self.connection:
             _print_error("Cannot add data: Index or database is not initialized.")
             raise RuntimeError("FtlDb is not properly initialized.")

        added_ids = []
        _print_info(f"Starting to add {len(data)} items in batches of {batch_size}...")

        for batch_num, batch_content in enumerate(_grouper(data, batch_size)):
            _print_info(f"Processing batch {batch_num + 1}...")
            try:
                # 1. Generate Embeddings
                embeddings = self.model.encode(batch_content, show_progress_bar=False)
                if embeddings.shape[1] != self.dimension:
                     _print_error(f"Generated embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension}).")
                     raise RuntimeError("Embedding dimension mismatch.")

                # 2. Prepare IndexData objects
                points_to_add = []
                db_values = []
                batch_ids = []
                for i, content in enumerate(batch_content):
                    current_id = self._next_id
                    point = IndexData(vector=embeddings[i], content=content, id=current_id)
                    points_to_add.append(point)
                    # Prepare values for DB insertion (convert metadata dict to string)
                    db_values.append((point.id, point.content, str(point.metadata)))
                    batch_ids.append(current_id)
                    self._next_id += 1

                # 3. Add to FAISS Index
                ids_np = np.array([p.id for p in points_to_add], dtype=np.int64)
                vectors_np = np.array([p.vector for p in points_to_add], dtype=np.float32)
                self.index.add_with_ids(vectors_np, ids_np)

                # 4. Add to SQLite Database
                with self._db_cursor() as cur:
                    cur.executemany(
                        f"""INSERT INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""",
                        db_values
                    )

                added_ids.extend(batch_ids)
                _print_success(f"Batch {batch_num + 1} added successfully ({len(batch_content)} items).")

            except ValidationError as e:
                _print_error(f"Data validation failed for batch {batch_num + 1}.", e)
                # Decide: skip batch or raise error? Raising seems safer.
                raise RuntimeError(f"Data validation error in batch {batch_num + 1}: {e}") from e
            except sqlite3.Error as e:
                _print_error(f"Database error occurred while adding batch {batch_num + 1}.", e)
                 # Attempt to rollback changes for this batch in FAISS? Difficult.
                 # Best to raise and let user decide how to handle inconsistency.
                raise RuntimeError(f"Database error in batch {batch_num + 1}: {e}") from e
            except Exception as e:
                _print_error(f"An unexpected error occurred while adding batch {batch_num + 1}.", e)
                raise RuntimeError(f"Unexpected error in batch {batch_num + 1}: {e}") from e

        _print_success(f"Finished adding {len(data)} items. Total items in index: {self.index.ntotal}")
        return added_ids

    def search(self, query: str, k: int = 3) -> List[Tuple[int, str, float, Dict]]:
        """
        Searches the index for the closest matches to the query text.

        Args:
            query (str): The text query to search for.
            k (int): The number of nearest neighbors to retrieve.

        Returns:
            List[Tuple[int, str, float, Dict]]: A list of tuples, where each tuple contains:
                - id (int): The ID of the matching item.
                - content (str): The text content of the matching item.
                - distance (float): The distance score (lower is more similar).
                - metadata (Dict): The metadata associated with the item.
            Returns an empty list if no results are found or an error occurs.

        Raises:
            ValueError: If the query is not a string or k is not a positive integer.
            RuntimeError: If the index is not initialized or search/retrieval fails.
        """
        if not isinstance(query, str) or not query:
            _print_error("Search query must be a non-empty string.")
            raise ValueError("Search query must be a non-empty string.")
        if not isinstance(k, int) or k <= 0:
            _print_error("Number of neighbors 'k' must be a positive integer.")
            raise ValueError("'k' must be a positive integer.")
        if not self.index or not self.connection:
            _print_error("Cannot search: Index or database is not initialized.")
            raise RuntimeError("FtlDb is not properly initialized.")
        if self.index.ntotal == 0:
            _print_info("Index is empty. Cannot perform search.")
            return []

        _print_info(f"Searching for '{query}' (k={k})...")
        try:
            # 1. Generate Query Embedding
            query_embedding = self.model.encode([query])
            if query_embedding.shape[1] != self.dimension:
                 _print_error(f"Query embedding dimension ({query_embedding.shape[1]}) does not match index dimension ({self.dimension}).")
                 raise RuntimeError("Query embedding dimension mismatch.")

            # 2. Search FAISS Index
            distances, ids = self.index.search(query_embedding.astype(np.float32), k)

            # Process results only if any IDs were found
            if ids.size == 0 or ids[0].size == 0 or ids[0][0] == -1: # FAISS returns -1 for no neighbor
                _print_info("No results found for the query.")
                return []

            found_ids = [int(id_val) for id_val in ids[0] if id_val != -1] # Filter out -1s and convert
            found_distances = [float(dist) for i, dist in enumerate(distances[0]) if ids[0][i] != -1]

            if not found_ids:
                 _print_info("No valid results found after filtering.")
                 return []

            # 3. Retrieve Metadata from SQLite
            results_map = {id_val: dist for id_val, dist in zip(found_ids, found_distances)}
            retrieved_data = []
            placeholders = ','.join('?' * len(found_ids))
            with self._db_cursor() as cur:
                cur.execute(
                    f"""SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})""",
                    found_ids
                )
                rows = cur.fetchall()

            # 4. Combine results and sort by distance
            final_results = []
            for row_id, content, metadata_str in rows:
                try:
                    # Safely evaluate metadata string back to dict
                    metadata = eval(metadata_str) if metadata_str and metadata_str != '{}' else {}
                    if not isinstance(metadata, dict): metadata = {'raw': metadata_str} # Handle non-dict eval results
                except Exception:
                    metadata = {'parsing_error': 'Could not parse metadata string', 'raw': metadata_str}

                if row_id in results_map:
                    final_results.append((row_id, content, results_map[row_id], metadata))

            # Sort by distance (ascending)
            final_results.sort(key=lambda item: item[2])

            _print_success(f"Found {len(final_results)} results.")
            return final_results

        except sqlite3.Error as e:
            _print_error("Database error occurred during search result retrieval.", e)
            # Return empty list as results are incomplete/unavailable
            return []
        except Exception as e:
            _print_error("An unexpected error occurred during search.", e)
            # Depending on the error, might want to raise or return empty
            raise RuntimeError(f"Unexpected search error: {e}") from e


    def save(self):
        """Saves the current FAISS index to disk."""
        if not self.index:
            _print_error("Cannot save: Index is not initialized.")
            raise RuntimeError("Index is not initialized.")
        if not self.index_path:
             _print_error("Cannot save: Index path is not set.")
             raise ValueError("Index path is not configured.")

        _print_info(f"Saving index to: {self.index_path}")
        try:
            chunk = faiss.serialize_index(self.index)
            with open(self.index_path, "wb") as f:
                pickle.dump(chunk, f)
            _print_success("Index saved successfully.")
        except Exception as e:
            _print_error("Failed to save index.", e)
            raise RuntimeError(f"Could not save index to '{self.index_path}': {e}") from e

    def close(self):
        """Closes the database connection."""
        if self.connection:
            _print_info("Closing database connection.")
            try:
                self.connection.close()
                self.connection = None
                _print_success("Database connection closed.")
            except sqlite3.Error as e:
                _print_error("Error closing database connection.", e)
        else:
             _print_info("Database connection already closed or never opened.")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring resources are released."""
        self.save() # Save index on exit
        self.close() # Close DB connection on exit

    def __del__(self):
        """Ensure resources are released if context manager isn't used."""
        self.close()


# --- Usage Example ---

if __name__ == "__main__":
    cprint("\n--- FtlDb Usage Example ---", "cyan", attrs=["bold"])

    # --- Configuration ---
    INDEX_NAME = "my_document_index"
    DIMENSION = 384 # Dimension for "all-MiniLM-L6-v2"
    MODEL = "all-MiniLM-L6-v2"
    SAMPLE_DATA = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
        "Mayank is a software developer.",
        "I don't think it works.",
        "Why do you want to know about my profession?",
        "What is your salary?",
        "Have conviction in the power of goodness.",
        "Mayank works as a software developer in California."
    ]
    QUERY = "Who is Mayank?"
    K_NEIGHBORS = 3

    # --- Initialization (using context manager for auto-save/close) ---
    try:
        # Set rebuild=True to start fresh, False to load existing
        with FtlDb(index_name=INDEX_NAME, dimension=DIMENSION, model_name=MODEL, rebuild=False) as db:
            _print_info(f"FtlDb instance created/loaded for index '{INDEX_NAME}'.")

            # --- Add Data (only if index is empty or rebuilding) ---
            if db.index.ntotal == 0:
                 _print_info("Index is empty, adding sample data...")
                 added_ids = db.add(SAMPLE_DATA, batch_size=4)
                 _print_info(f"Added data with IDs: {added_ids}")
            else:
                 _print_info(f"Index already contains {db.index.ntotal} items. Skipping data addition.")


            # --- Search ---
            cprint("\n--- Performing Search ---", "yellow")
            search_results = db.search(QUERY, k=K_NEIGHBORS)

            if search_results:
                cprint(f"Search results for '{QUERY}' (top {K_NEIGHBORS}):", "green")
                for idx, (res_id, content, distance, metadata) in enumerate(search_results):
                    print(f"  {idx+1}. ID: {res_id}, Distance: {distance:.4f}")
                    print(f"     Content: {content}")
                    if metadata:
                        print(f"     Metadata: {metadata}")
            else:
                cprint(f"No results found for query: '{QUERY}'", "yellow")

            # --- Manual Save (optional, done automatically by context manager) ---
            # db.save()

            # --- Manual Close (optional, done automatically by context manager) ---
            # db.close()

    except (ValueError, RuntimeError, ConnectionError, FileNotFoundError) as e:
        _print_error("An error occurred during the FtlDb operation.", e)
    except Exception as e:
         _print_error("An unexpected critical error occurred.", e)

    cprint("\n--- FtlDb Example Finished ---", "cyan", attrs=["bold"])
