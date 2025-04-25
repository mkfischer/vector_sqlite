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
                 dimension: Optional[int] = None, # User-provided dimension hint/requirement
                 db_path: Optional[str] = None,
                 index_path: Optional[str] = None,
                 model_name: str = "all-MiniLM-L6-v2",
                 rebuild: bool = False):
        """
        Initializes or loads the FtlDb instance.

        Args:
            index_name (str): A unique name for the index and database table.
            dimension (Optional[int]): The dimension of the vectors. If loading an existing index,
                                      this is validated against the loaded index and the model.
                                      If creating a new index, this must match the model's dimension if provided.
            db_path (Optional[str]): Path to the SQLite database file. Defaults to f"{index_name}.db".
            index_path (Optional[str]): Path to the FAISS index file. Defaults to f"{index_name}.pkl".
            model_name (str): The name of the Sentence Transformer model to use for embedding.
            rebuild (bool): If True, forces creation of a new index and database, potentially overwriting existing ones.

        Raises:
            ValueError: If index_name is invalid, or if dimension conflicts arise during initialization.
            RuntimeError: If the embedding model cannot be loaded.
            ConnectionError: If the database connection fails.
        """
        if not index_name or not isinstance(index_name, str):
            _print_error("Initialization failed: 'index_name' must be a non-empty string.")
            raise ValueError("'index_name' must be a non-empty string.")

        self.index_name = index_name
        self.db_path = db_path or f"{self.index_name}.db"
        self.index_path = index_path or f"{self.index_name}.pkl"
        self.index: Optional[faiss.IndexIDMap2] = None
        self.dimension: Optional[int] = None # Final determined dimension
        self.connection: Optional[sqlite3.Connection] = None
        self._next_id: int = 0

        # 1. Load the model and get its dimension FIRST
        try:
            _print_info(f"Loading Sentence Transformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            model_dimension = self.model.get_sentence_embedding_dimension()
            _print_info(f"Model '{model_name}' loaded. Dimension: {model_dimension}")
        except Exception as e:
            _print_error(f"Failed to load Sentence Transformer model '{model_name}'.", e)
            raise RuntimeError(f"Could not load model '{model_name}': {e}") from e

        # 2. Handle rebuild flag
        if rebuild:
            _print_info(f"Rebuild requested. Removing existing index and DB files if they exist.")
            if os.path.exists(self.index_path):
                try:
                    os.remove(self.index_path)
                    _print_info(f"Removed existing index file: {self.index_path}")
                except OSError as e:
                    _print_error(f"Could not remove existing index file: {self.index_path}", e)
            if os.path.exists(self.db_path):
                try:
                    os.remove(self.db_path)
                    _print_info(f"Removed existing database file: {self.db_path}")
                except OSError as e:
                     _print_error(f"Could not remove existing database file: {self.db_path}", e)


        # 3. Attempt to load the index if it exists and not rebuilding
        loaded_index_dimension = None
        index_loaded_successfully = False
        if os.path.exists(self.index_path) and not rebuild:
            try:
                _print_info(f"Attempting to load existing index from: {self.index_path}")
                with open(self.index_path, "rb") as f:
                    serialized_index = pickle.load(f)
                loaded_index = faiss.deserialize_index(serialized_index)

                if not isinstance(loaded_index, faiss.IndexIDMap2):
                     _print_error("Loaded file does not contain a valid FAISS IndexIDMap2 object. Will create a new index.")
                     # Proceed as if index wasn't loaded
                else:
                    self.index = loaded_index
                    loaded_index_dimension = self.index.d
                    index_loaded_successfully = True
                    _print_success(f"Index loaded successfully. Dimension: {loaded_index_dimension}, Entries: {self.index.ntotal}")

            except (FileNotFoundError, pickle.UnpicklingError, EOFError, AttributeError, TypeError, faiss.FaissException) as e:
                _print_error(f"Failed to load or deserialize index from {self.index_path}. A new index will be created.", e)
                self.index = None # Ensure index is None if loading failed
            except Exception as e: # Catch other potential errors
                 _print_error(f"An unexpected error occurred loading the index from {self.index_path}. A new index will be created.", e)
                 self.index = None

        # 4. Determine the final dimension and validate
        if index_loaded_successfully and loaded_index_dimension is not None:
            # Use dimension from the loaded index
            self.dimension = loaded_index_dimension
            _print_info(f"Using dimension from loaded index: {self.dimension}")

            # Check consistency with the provided dimension argument (if any)
            if dimension is not None and dimension != self.dimension:
                _print_error(f"Provided dimension ({dimension}) conflicts with loaded index dimension ({self.dimension}). Using loaded dimension.")
                # This is just a warning, we proceed with the loaded dimension.

            # CRITICAL CHECK: Validate loaded index dimension against the current model's dimension
            if self.dimension != model_dimension:
                _print_error(f"Loaded index dimension ({self.dimension}) is incompatible with the current model's dimension ({model_dimension}).")
                raise ValueError(f"Loaded index dimension ({self.dimension}) conflicts with model '{model_name}' dimension ({model_dimension}). Cannot proceed.")

        else:
            # Creating a new index (or loading failed)
            _print_info("Creating a new index.")
            if dimension is None:
                # No dimension provided, use the model's dimension
                self.dimension = model_dimension
                _print_info(f"Using inferred dimension from model: {self.dimension}")
            else:
                # Dimension provided, it MUST match the model's dimension for a new index
                if dimension != model_dimension:
                    _print_error(f"Provided dimension ({dimension}) does not match model '{model_name}' dimension ({model_dimension}) required for new index.")
                    raise ValueError(f"Provided dimension ({dimension}) must match model dimension ({model_dimension}) for a new index.")
                self.dimension = dimension
                _print_info(f"Using provided dimension for new index: {self.dimension}")

            # Create the actual new index object
            try:
                _print_info(f"Creating new FAISS index (IndexFlatL2 -> IndexIDMap2) with dimension {self.dimension}.")
                flat_index = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIDMap2(flat_index)
                _print_success("New FAISS index created.")
            except Exception as e:
                _print_error("Failed to create new FAISS index.", e)
                raise RuntimeError(f"Could not create FAISS index: {e}") from e

        # 5. Connect DB and update ID
        if self.dimension is None or self.index is None:
             # This should theoretically not happen if logic above is correct, but as a safeguard:
             _print_error("Initialization failed: Index or dimension could not be determined.")
             raise RuntimeError("Failed to initialize index structure.")

        self._connect_db()
        self._update_next_id()
        _print_success(f"FtlDb initialized successfully for index '{self.index_name}'.")


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
            try:
                self.connection.rollback() # Rollback on error
            except sqlite3.Error as rb_err:
                 _print_error("Rollback failed.", rb_err)
            raise # Re-raise the original exception
        else:
            try:
                self.connection.commit() # Commit on success
            except sqlite3.Error as commit_err:
                 _print_error("Commit failed.", commit_err)
                 raise
        finally:
            cursor.close()

    def _connect_db(self):
        """Establishes connection to the SQLite database and creates table if needed."""
        try:
            # Use isolation_level=None for autocommit, managed by context manager
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
            # Enable Write-Ahead Logging for potentially better concurrency
            self.connection.execute("PRAGMA journal_mode=WAL;")
            sqlite3.register_adapter(np.int64, int) # Adapt numpy int64 for sqlite
            _print_info(f"Connecting to database: {self.db_path}")
            with self._db_cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.index_name} (
                        id INTEGER PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT -- Storing as JSON string or similar
                    )""")
                # Consider adding index on id for faster lookups if needed
                # cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.index_name}_id ON {self.index_name}(id);")
            _print_success(f"Database connected and table '{self.index_name}' ensured.")
        except sqlite3.Error as e:
            _print_error("Failed to connect to or initialize the database.", e)
            if self.connection:
                self.connection.close()
            self.connection = None
            raise ConnectionError(f"Could not connect to database '{self.db_path}': {e}") from e

    # _load_or_create_index is removed as its logic is integrated into __init__
    # _create_new_index is removed as its logic is integrated into __init__

    def _update_next_id(self):
        """ Determines the next available ID based on the database."""
        if not self.connection:
             _print_error("Cannot update next ID: Database not connected.")
             # Set a default, but this indicates a problem state
             self._next_id = 0
             return

        try:
            with self._db_cursor() as cur:
                # Use MAX(id) which is reliable for auto-incrementing-like behavior
                cur.execute(f"SELECT MAX(id) FROM {self.index_name}")
                max_id = cur.fetchone()[0]
                self._next_id = (max_id + 1) if max_id is not None else 0

                # Sanity check against FAISS index size if index has items
                # Note: FAISS ntotal might be misleading if remove_ids was used.
                # Relying on DB MAX(id) is generally safer for sequential additions.
                if self.index and self.index.ntotal > 0:
                     if self._next_id < self.index.ntotal:
                         # This might happen if DB was cleared but index wasn't, or IDs are non-sequential
                         _print_info(f"DB MAX(id)+1 ({self._next_id}) is less than index count ({self.index.ntotal}). Using MAX(id)+1.")
                         # Stick with DB-derived ID for adding new items sequentially.
                     pass # No action needed, just observation

            _print_info(f"Next available ID set to: {self._next_id}")
        except sqlite3.Error as e:
            _print_error("Failed to determine next ID from database.", e)
            # Fallback to 0, but log prominently
            self._next_id = 0
            _print_error("Defaulting next available ID to 0 due to database error.")


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
            ConnectionError: If the database is not connected.
        """
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            _print_error("Input 'data' must be a list of strings.")
            raise ValueError("Input 'data' must be a list of strings.")
        if not self.index or not self.connection:
             _print_error("Cannot add data: Index or database is not initialized.")
             raise RuntimeError("FtlDb is not properly initialized. Check for initialization errors.")
        if self.dimension is None:
             _print_error("Cannot add data: Dimension is not set.")
             raise RuntimeError("FtlDb dimension is not set. Check initialization.")


        added_ids = []
        _print_info(f"Starting to add {len(data)} items in batches of {batch_size}...")

        for batch_num, batch_content in enumerate(_grouper(data, batch_size)):
            if not batch_content: continue # Skip empty batches

            _print_info(f"Processing batch {batch_num + 1}/{ (len(data) + batch_size - 1)//batch_size }...")
            try:
                # 1. Generate Embeddings
                embeddings = self.model.encode(batch_content, show_progress_bar=False)
                if embeddings.shape[1] != self.dimension:
                     # This check should ideally be redundant due to init checks, but good safeguard
                     _print_error(f"Generated embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension}).")
                     raise RuntimeError("Embedding dimension mismatch during add operation.")

                # 2. Prepare IndexData objects and DB values
                points_to_add = []
                db_values = []
                batch_ids = []
                start_id = self._next_id
                for i, content in enumerate(batch_content):
                    current_id = start_id + i
                    # Create IndexData (vector validation happens here)
                    point = IndexData(vector=embeddings[i], content=content, id=current_id)
                    points_to_add.append(point)
                    # Prepare values for DB insertion (convert metadata dict to string)
                    # Using json.dumps might be safer than str() for complex metadata
                    db_values.append((point.id, point.content, str(point.metadata))) # TODO: Consider json.dumps
                    batch_ids.append(current_id)

                # 3. Add to FAISS Index
                ids_np = np.array([p.id for p in points_to_add], dtype=np.int64)
                vectors_np = np.array([p.vector for p in points_to_add], dtype=np.float32) # Already float32 from IndexData validation
                self.index.add_with_ids(vectors_np, ids_np)

                # 4. Add to SQLite Database
                with self._db_cursor() as cur:
                    cur.executemany(
                        f"""INSERT INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""",
                        db_values
                    )

                # 5. Update next_id *after* successful batch insertion
                self._next_id = start_id + len(batch_content)
                added_ids.extend(batch_ids)
                _print_success(f"Batch {batch_num + 1} added successfully ({len(batch_content)} items). New next ID: {self._next_id}")

            except ValidationError as e:
                _print_error(f"Data validation failed for batch {batch_num + 1}.", e)
                # Decide: skip batch or raise error? Raising seems safer for data integrity.
                raise RuntimeError(f"Data validation error in batch {batch_num + 1}: {e}") from e
            except sqlite3.Error as e:
                _print_error(f"Database error occurred while adding batch {batch_num + 1}.", e)
                 # FAISS index might now be inconsistent with DB for this batch.
                 # Manual intervention might be needed. Raising is crucial.
                _print_error(f"FAISS index count: {self.index.ntotal}. Next expected DB ID was: {start_id}")
                raise RuntimeError(f"Database error in batch {batch_num + 1}: {e}. Index might be inconsistent.") from e
            except faiss.FaissException as e:
                 _print_error(f"FAISS error occurred while adding batch {batch_num + 1}.", e)
                 raise RuntimeError(f"FAISS error in batch {batch_num + 1}: {e}") from e
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
                - distance (float): The distance score (L2 distance, lower is more similar).
                - metadata (Dict): The metadata associated with the item.
            Returns an empty list if no results are found or an error occurs.

        Raises:
            ValueError: If the query is not a string or k is not a positive integer.
            RuntimeError: If the index is not initialized or search/retrieval fails.
            ConnectionError: If the database is not connected.
        """
        if not isinstance(query, str) or not query:
            _print_error("Search query must be a non-empty string.")
            raise ValueError("Search query must be a non-empty string.")
        if not isinstance(k, int) or k <= 0:
            _print_error("Number of neighbors 'k' must be a positive integer.")
            raise ValueError("'k' must be a positive integer.")
        if not self.index or not self.connection:
            _print_error("Cannot search: Index or database is not initialized.")
            raise RuntimeError("FtlDb is not properly initialized. Check for initialization errors.")
        if self.dimension is None:
             _print_error("Cannot search: Dimension is not set.")
             raise RuntimeError("FtlDb dimension is not set. Check initialization.")

        if self.index.ntotal == 0:
            _print_info("Index is empty. Cannot perform search.")
            return []

        # Adjust k if it's larger than the number of items in the index
        actual_k = min(k, self.index.ntotal)
        if actual_k != k:
            _print_info(f"Requested k={k}, but index only contains {self.index.ntotal} items. Searching for k={actual_k}.")


        _print_info(f"Searching for '{query}' (k={actual_k})...")
        try:
            # 1. Generate Query Embedding
            query_embedding = self.model.encode([query])
            if query_embedding.shape[1] != self.dimension:
                 # Should be redundant due to init checks
                 _print_error(f"Query embedding dimension ({query_embedding.shape[1]}) does not match index dimension ({self.dimension}).")
                 raise RuntimeError("Query embedding dimension mismatch during search.")

            # 2. Search FAISS Index
            # Ensure query embedding is float32 and 2D array
            query_vector_np = query_embedding.astype(np.float32).reshape(1, -1)
            distances, ids = self.index.search(query_vector_np, actual_k)

            # Process results only if any IDs were found
            if ids.size == 0 or ids[0].size == 0 or ids[0][0] == -1: # FAISS returns -1 for no neighbor
                _print_info("No results found for the query in FAISS search.")
                return []

            # Filter out invalid IDs (-1) and flatten results
            found_ids = [int(id_val) for id_val in ids[0] if id_val != -1]
            found_distances = [float(dist) for i, dist in enumerate(distances[0]) if ids[0][i] != -1]

            if not found_ids:
                 _print_info("No valid results found after filtering FAISS IDs.")
                 return []

            # 3. Retrieve Metadata from SQLite
            results_map = {id_val: dist for id_val, dist in zip(found_ids, found_distances)}
            retrieved_data = []
            placeholders = ','.join('?' * len(found_ids))
            query_sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"

            with self._db_cursor() as cur:
                cur.execute(query_sql, found_ids)
                rows = cur.fetchall()

            # 4. Combine results and sort by distance (FAISS already returns sorted by distance)
            final_results = []
            # Create a map of DB results for efficient lookup
            db_results_map = {row_id: (content, metadata_str) for row_id, content, metadata_str in rows}

            # Iterate through FAISS results (which are ordered by distance)
            for res_id, distance in zip(found_ids, found_distances):
                if res_id in db_results_map:
                    content, metadata_str = db_results_map[res_id]
                    try:
                        # Safely evaluate metadata string back to dict
                        # Consider using json.loads if metadata is stored as JSON
                        metadata = eval(metadata_str) if metadata_str and metadata_str != '{}' else {}
                        if not isinstance(metadata, dict):
                            _print_error(f"Metadata for ID {res_id} is not a dict: {metadata_str}. Storing as raw.")
                            metadata = {'raw': metadata_str}
                    except Exception as e:
                         _print_error(f"Could not parse metadata string for ID {res_id}: '{metadata_str}'", e)
                         metadata = {'parsing_error': str(e), 'raw': metadata_str}

                    final_results.append((res_id, content, distance, metadata))
                else:
                    # This indicates inconsistency between FAISS and DB
                    _print_error(f"ID {res_id} found in FAISS index but not in database table '{self.index_name}'. Skipping.")


            # Sorting should not be necessary if we process in the order returned by FAISS
            # final_results.sort(key=lambda item: item[2])

            _print_success(f"Found and retrieved {len(final_results)} results.")
            return final_results

        except sqlite3.Error as e:
            _print_error("Database error occurred during search result retrieval.", e)
            # Return empty list as results are incomplete/unavailable
            return []
        except faiss.FaissException as e:
             _print_error("FAISS error occurred during search.", e)
             return []
        except Exception as e:
            _print_error("An unexpected error occurred during search.", e)
            # Depending on the error, might want to raise or return empty
            raise RuntimeError(f"Unexpected search error: {e}") from e


    def save(self):
        """Saves the current FAISS index to disk."""
        if not self.index:
            _print_error("Cannot save: Index is not initialized.")
            # Optionally raise an error, or just return if saving a non-existent index is acceptable
            # raise RuntimeError("Index is not initialized.")
            return
        if not self.index_path:
             _print_error("Cannot save: Index path is not set.")
             raise ValueError("Index path is not configured.")

        _print_info(f"Saving index ({self.index.ntotal} entries) to: {self.index_path}")
        try:
            # Ensure the directory exists
            index_dir = os.path.dirname(self.index_path)
            if index_dir: # Only create if path includes a directory
                 os.makedirs(index_dir, exist_ok=True)

            chunk = faiss.serialize_index(self.index)
            with open(self.index_path, "wb") as f:
                pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol
            _print_success("Index saved successfully.")
        except pickle.PicklingError as e:
             _print_error("Failed to serialize index for saving.", e)
             raise RuntimeError(f"Could not serialize index: {e}") from e
        except OSError as e:
             _print_error(f"Failed to write index file to '{self.index_path}'. Check permissions or path.", e)
             raise RuntimeError(f"Could not save index to '{self.index_path}': {e}") from e
        except Exception as e:
            _print_error("An unexpected error occurred during index saving.", e)
            raise RuntimeError(f"Could not save index to '{self.index_path}': {e}") from e

    def close(self):
        """Closes the database connection."""
        if self.connection:
            _print_info(f"Closing database connection to {self.db_path}.")
            try:
                # Optional: Add checkpointing for WAL mode before closing
                # self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                self.connection.close()
                self.connection = None
                _print_success("Database connection closed.")
            except sqlite3.Error as e:
                _print_error("Error closing database connection.", e)
                # Connection might be in an unusable state, set to None anyway
                self.connection = None
        else:
             _print_info("Database connection already closed or never opened.")

    def __enter__(self):
        """Enter context manager."""
        # Connection and index are already handled in __init__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager, ensuring resources are released."""
        _print_info("Exiting FtlDb context...")
        try:
            if exc_type is None: # No exception occurred within the 'with' block
                self.save() # Save index only on clean exit
            else:
                 _print_error(f"Exception occurred within context: {exc_type}. Index will not be saved automatically.")
        except Exception as e:
             _print_error("Error occurred during automatic index save on exit.", e)
        finally:
            self.close() # Always close DB connection

    def __del__(self):
        """Ensure resources are released if context manager isn't used (best effort)."""
        # This is less reliable than the context manager. Warn if connection is open.
        if self.connection:
             _print_error("FtlDb instance deleted without closing DB connection explicitly or using context manager. Attempting close.")
             self.close()


# --- Usage Example ---

if __name__ == "__main__":
    cprint("\n--- FtlDb Usage Example ---", "cyan", attrs=["bold"])

    # --- Configuration ---
    INDEX_NAME = "my_document_index"
    # Let FtlDb infer dimension from the model
    # DIMENSION = 384 # Dimension for "all-MiniLM-L6-v2"
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
    DB_FILE = f"{INDEX_NAME}_example.db"
    INDEX_FILE = f"{INDEX_NAME}_example.pkl"

    # Clean up previous run files for fresh example
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    if os.path.exists(INDEX_FILE): os.remove(INDEX_FILE)

    # --- Initialization (using context manager for auto-save/close) ---
    try:
        # Set rebuild=True to start fresh, False to load existing
        # Not providing dimension, letting it be inferred from the model
        with FtlDb(index_name=INDEX_NAME,
                   model_name=MODEL,
                   db_path=DB_FILE,
                   index_path=INDEX_FILE,
                   rebuild=True) as db: # Start fresh for example
            _print_info(f"FtlDb instance created/loaded for index '{INDEX_NAME}'. Dimension: {db.dimension}")

            # --- Add Data ---
            _print_info("Adding sample data...")
            added_ids = db.add(SAMPLE_DATA, batch_size=4)
            _print_info(f"Added data with IDs: {added_ids}")

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

        # --- Example of loading the saved index ---
        cprint("\n--- Reloading the Index ---", "cyan")
        if os.path.exists(INDEX_FILE) and os.path.exists(DB_FILE):
             with FtlDb(index_name=INDEX_NAME, model_name=MODEL, db_path=DB_FILE, index_path=INDEX_FILE) as db_reloaded:
                 _print_success(f"Successfully reloaded index '{db_reloaded.index_name}'.")
                 _print_info(f"Index contains {db_reloaded.index.ntotal} items.")
                 # Perform another search
                 search_results_reloaded = db_reloaded.search("What is the weather like?", k=2)
                 cprint(f"Search results after reload for 'What is the weather like?' (top 2):", "green")
                 for idx, (res_id, content, distance, metadata) in enumerate(search_results_reloaded):
                    print(f"  {idx+1}. ID: {res_id}, Distance: {distance:.4f}, Content: {content}")

        else:
             _print_error("Index or DB file not found for reloading example.")


    except (ValueError, RuntimeError, ConnectionError, FileNotFoundError) as e:
        _print_error("An error occurred during the FtlDb operation.", e)
    except Exception as e:
         _print_error("An unexpected critical error occurred.", e)

    cprint("\n--- FtlDb Example Finished ---", "cyan", attrs=["bold"])

