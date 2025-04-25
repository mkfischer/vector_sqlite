import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Any, Generator
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse
from pydantic import BaseModel as PydanticBaseModel, Field, validator
import json # For storing metadata and potentially complex embeddings

# --- Configuration ---
DEFAULT_DB_NAME = "fql.db"
DEFAULT_BATCH_SIZE = 32

# --- Conditional FastEmbed Imports ---
try:
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding, ImageEmbedding, SparseEmbedding, DefaultEmbedding
    from fastembed.rerank import TextCrossEncoder
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    # Define dummy classes if fastembed is not available to avoid NameErrors
    class DummyEmbedding:
        def __init__(self, *args, **kwargs): pass
        def embed(self, *args, **kwargs): raise ImportError("fastembed is not installed.")
        def query_embed(self, *args, **kwargs): raise ImportError("fastembed is not installed.")
        def passage_embed(self, *args, **kwargs): raise ImportError("fastembed is not installed.")
    class DummyReranker:
         def __init__(self, *args, **kwargs): pass
         def rerank(self, *args, **kwargs): raise ImportError("fastembed is not installed.")

    TextEmbedding = DummyEmbedding
    SparseTextEmbedding = DummyEmbedding
    LateInteractionTextEmbedding = DummyEmbedding
    ImageEmbedding = DummyEmbedding
    TextCrossEncoder = DummyReranker
    SparseEmbedding = object # Placeholder for type hinting

# --- Pydantic Models ---

class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class IndexData(BaseModel):
    """Data structure for items to be indexed."""
    id: int
    content: Union[str, List[str]] # Text, list of texts (for late interaction), or image path
    vector: Optional[Union[np.ndarray, List[np.ndarray], Any]] = None # Dense, list (late interaction), or SparseEmbedding
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_type: str = "dense" # 'dense', 'sparse', 'late_interaction', 'image'

    @validator('vector')
    def vector_must_be_numpy_or_sparse(cls, v):
        if v is None:
            return v # Allow None if vector is generated later
        if isinstance(v, np.ndarray):
            return v
        # Check if it looks like FastEmbed's SparseEmbedding (duck typing if fastembed not installed)
        if hasattr(v, 'indices') and hasattr(v, 'values'):
             return v
        # Allow list of numpy arrays for late interaction
        if isinstance(v, list) and all(isinstance(item, np.ndarray) for item in v):
            return v
        raise TypeError("Vector must be a numpy array, a list of numpy arrays, or a SparseEmbedding-like object")

# --- Helper Functions ---

def grouper(iterable: list, n: int) -> Generator[List, None, None]:
    """Iterate over a list in chunks of size n."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def cleanup_files(*filenames: str):
    """Removes specified files if they exist."""
    files_removed = False
    for filename in filenames:
        try:
            if filename and os.path.exists(filename):
                os.remove(filename)
                files_removed = True
        except Exception as e:
            cprint(f"Error removing file {filename}: {e}", "red")
    if files_removed:
        cprint("Cleanup: Removed generated files.", "yellow")

def _serialize_vector(vector: Any) -> Optional[bytes]:
    """Serialize vector data for SQLite storage."""
    if vector is None:
        return None
    # Handle SparseEmbedding specifically if fastembed is available
    if FASTEMBED_AVAILABLE and isinstance(vector, SparseEmbedding):
         # Store as JSON dictionary
         sparse_dict = {"indices": vector.indices.tolist(), "values": vector.values.tolist()}
         return json.dumps(sparse_dict).encode('utf-8')
    # Handle list of numpy arrays (late interaction)
    if isinstance(vector, list) and all(isinstance(item, np.ndarray) for item in vector):
         # Pickle the list of arrays
         return pickle.dumps([arr.tolist() for arr in vector]) # Store as list of lists
    # Handle dense numpy array
    if isinstance(vector, np.ndarray):
        return pickle.dumps(vector)
    cprint(f"Warning: Attempting to serialize unsupported vector type: {type(vector)}", "yellow")
    return pickle.dumps(vector) # Fallback for other types

def _deserialize_vector(blob: Optional[bytes], vector_type: str) -> Any:
    """Deserialize vector data from SQLite storage."""
    if blob is None:
        return None
    try:
        # Handle SparseEmbedding specifically
        if vector_type == "sparse" and FASTEMBED_AVAILABLE:
            sparse_dict = json.loads(blob.decode('utf-8'))
            return SparseEmbedding(indices=np.array(sparse_dict["indices"]), values=np.array(sparse_dict["values"]))
        # Handle late interaction (list of numpy arrays)
        if vector_type == "late_interaction":
             list_of_lists = pickle.loads(blob)
             return [np.array(item, dtype=np.float32) for item in list_of_lists]
        # Handle dense numpy array (default)
        return pickle.loads(blob)
    except (pickle.UnpicklingError, json.JSONDecodeError, TypeError, AttributeError) as e:
        cprint(f"Error deserializing vector (type: {vector_type}): {e}", "red")
        return None

# --- FqlDb Class ---

class FqlDb:
    """
    A class combining vector similarity search (FAISS or FastEmbed) with a SQLite database.

    Supports dense, sparse, late interaction, and image embeddings using FastEmbed,
    or custom dense vectors with FAISS.
    """

    def __init__(self,
                 index_name: str,
                 dimension: Optional[int] = None,
                 db_name: str = DEFAULT_DB_NAME,
                 use_fastembed: bool = False,
                 model_name: Optional[str] = None, # For dense, sparse, late interaction, image
                 reranker_model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 **fastembed_kwargs):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): A unique name for the index and database table.
            dimension (Optional[int]): The dimension of the dense vectors. Required if not using FastEmbed
                                       or if using FastEmbed without specifying a model_name that allows dimension inference.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            use_fastembed (bool, optional): Whether to use FastEmbed for embeddings and reranking. Defaults to False.
            model_name (Optional[str], optional): The name of the FastEmbed model to use. Required if use_fastembed is True.
                                                  Used for dense, sparse, late interaction, or image embeddings based on context.
            reranker_model_name (Optional[str], optional): The name of the FastEmbed reranker model. Only used if use_fastembed is True.
            cache_dir (Optional[str], optional): Directory to cache downloaded FastEmbed models.
            batch_size (int, optional): Batch size for embedding generation. Defaults to 32.
            **fastembed_kwargs: Additional keyword arguments passed to FastEmbed model initializers (e.g., providers).
        """
        cprint(f"Initializing FqlDb '{index_name}'...", "blue")
        self.index_name = index_name
        self.db_name = db_name
        self.use_fastembed = use_fastembed
        self.model_name = model_name
        self.reranker_model_name = reranker_model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.fastembed_kwargs = fastembed_kwargs

        self.embedding_model: Optional[Any] = None
        self.reranker_model: Optional[Any] = None
        self.dimension = dimension

        if self.use_fastembed:
            if not FASTEMBED_AVAILABLE:
                raise ImportError("FastEmbed is selected but not installed. Please install 'fastembed'.")
            if not self.model_name:
                raise ValueError("`model_name` must be provided when `use_fastembed` is True.")
            self._initialize_fastembed_models()
            # Try to infer dimension if not provided
            if self.dimension is None and hasattr(self.embedding_model, 'dim'):
                 self.dimension = self.embedding_model.dim
                 cprint(f"Inferred dimension {self.dimension} from FastEmbed model '{self.model_name}'.", "cyan")

        if self.dimension is None:
             raise ValueError("`dimension` must be provided if not using FastEmbed or if it cannot be inferred from the model.")

        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self._create_table() # Ensure table exists on init
        self.index = self._load_or_build_faiss_index() # FAISS index is primarily for dense vectors

        cprint(f"FqlDb '{index_name}' initialized successfully.", "green")
        cprint(f"  Mode: {'FastEmbed' if self.use_fastembed else 'FAISS (External Vectors)'}", "cyan")
        if self.use_fastembed:
            cprint(f"  Embedding Model: {self.model_name}", "cyan")
            if self.reranker_model:
                cprint(f"  Reranker Model: {self.reranker_model_name}", "cyan")
        cprint(f"  Dense Vector Dimension: {self.dimension}", "cyan")
        cprint(f"  Database File: {self.db_name}", "cyan")
        cprint(f"  FAISS Index File: {self.index_name}.pkl", "cyan")


    def _initialize_fastembed_models(self):
        """Initializes FastEmbed models based on configuration."""
        cprint(f"Loading FastEmbed model: {self.model_name}...", "cyan")
        # Try initializing as different types - this is a bit heuristic
        try:
            # Try TextEmbedding first (covers dense text)
            self.embedding_model = TextEmbedding(self.model_name, cache_dir=self.cache_dir, **self.fastembed_kwargs)
            cprint(f"Initialized as TextEmbedding.", "green")
            return
        except Exception: pass # Try next type

        try:
            # Try SparseTextEmbedding
            self.embedding_model = SparseTextEmbedding(self.model_name, cache_dir=self.cache_dir, **self.fastembed_kwargs)
            cprint(f"Initialized as SparseTextEmbedding.", "green")
            return
        except Exception: pass

        try:
            # Try LateInteractionTextEmbedding
            self.embedding_model = LateInteractionTextEmbedding(self.model_name, cache_dir=self.cache_dir, **self.fastembed_kwargs)
            cprint(f"Initialized as LateInteractionTextEmbedding.", "green")
            return
        except Exception: pass

        try:
            # Try ImageEmbedding
            self.embedding_model = ImageEmbedding(self.model_name, cache_dir=self.cache_dir, **self.fastembed_kwargs)
            cprint(f"Initialized as ImageEmbedding.", "green")
            return
        except Exception as e:
             cprint(f"Failed to initialize FastEmbed model '{self.model_name}' as any known type: {e}", "red")
             raise ValueError(f"Could not initialize FastEmbed model: {self.model_name}")

        if self.reranker_model_name:
            cprint(f"Loading FastEmbed reranker: {self.reranker_model_name}...", "cyan")
            try:
                self.reranker_model = TextCrossEncoder(self.reranker_model_name, cache_dir=self.cache_dir, **self.fastembed_kwargs)
                cprint(f"Initialized Reranker.", "green")
            except Exception as e:
                cprint(f"Failed to initialize FastEmbed reranker '{self.reranker_model_name}': {e}", "red")
                # Don't raise, reranking is optional

    def _get_embedding_generator(self, texts: Union[str, List[str]], vector_type: str, is_query: bool = False) -> Generator:
        """Internal helper to get embeddings from the FastEmbed model."""
        if not self.use_fastembed or not self.embedding_model:
            raise RuntimeError("FastEmbed is not enabled or model not initialized.")

        if isinstance(texts, str):
            texts = [texts]

        try:
            if vector_type == "dense":
                if isinstance(self.embedding_model, TextEmbedding):
                    # Use query_embed for single query strings if is_query is True
                    if is_query and len(texts) == 1:
                         return self.embedding_model.query_embed(texts, batch_size=self.batch_size)
                    else: # Use passage_embed for documents or batches of queries
                         return self.embedding_model.passage_embed(texts, batch_size=self.batch_size)
                else: # Fallback to generic embed if it's not TextEmbedding but dense is requested
                     return self.embedding_model.embed(texts, batch_size=self.batch_size)
            elif vector_type == "sparse":
                if isinstance(self.embedding_model, SparseTextEmbedding):
                    return self.embedding_model.embed(texts, batch_size=self.batch_size)
                else:
                    raise TypeError("Sparse embeddings requested, but the loaded model is not SparseTextEmbedding.")
            elif vector_type == "late_interaction":
                 if isinstance(self.embedding_model, LateInteractionTextEmbedding):
                     if is_query and len(texts) == 1:
                         return self.embedding_model.query_embed(texts, batch_size=self.batch_size)
                     else:
                         return self.embedding_model.embed(texts, batch_size=self.batch_size) # Passage embed for documents
                 else:
                     raise TypeError("Late interaction embeddings requested, but the loaded model is not LateInteractionTextEmbedding.")
            elif vector_type == "image":
                if isinstance(self.embedding_model, ImageEmbedding):
                    return self.embedding_model.embed(texts, batch_size=self.batch_size) # texts are paths here
                else:
                    raise TypeError("Image embeddings requested, but the loaded model is not ImageEmbedding.")
            else:
                raise ValueError(f"Unsupported vector_type for FastEmbed: {vector_type}")
        except Exception as e:
            cprint(f"Error during FastEmbed embedding generation (type: {vector_type}, query: {is_query}): {e}", "red")
            raise

    def _embed_batch(self, batch_data: List[IndexData]) -> List[IndexData]:
        """Embeds a batch of IndexData using the configured FastEmbed model."""
        if not self.use_fastembed or not self.embedding_model:
            return batch_data # Return as is if not using fastembed

        contents = [item.content for item in batch_data]
        vector_type = batch_data[0].vector_type # Assume uniform type within batch

        # Handle potential list content for late interaction documents
        if vector_type == "late_interaction":
             processed_contents = []
             for content in contents:
                 if isinstance(content, list):
                     # Assuming late interaction documents might be passed as lists of strings
                     processed_contents.append(" ".join(content)) # Join for embedding, actual storage might differ
                 else:
                     processed_contents.append(content)
             contents = processed_contents

        try:
            embeddings_gen = self._get_embedding_generator(contents, vector_type, is_query=False)
            embeddings = list(embeddings_gen)

            if len(embeddings) != len(batch_data):
                 raise ValueError("Number of embeddings generated does not match batch size.")

            for i, item in enumerate(batch_data):
                item.vector = embeddings[i]
            return batch_data
        except Exception as e:
             cprint(f"Failed to embed batch: {e}", "red")
             # Return original data without vectors
             for item in batch_data:
                 item.vector = None
             return batch_data


    def add(self, data: List[IndexData], store: bool = True) -> None:
        """
        Adds data to the index and optionally stores it in the database.
        Generates embeddings using FastEmbed if configured.

        Args:
            data (List[IndexData]): A list of IndexData objects.
                                     If use_fastembed is True, vectors can be None and will be generated.
            store (bool, optional): Whether to store the data in the SQLite database. Defaults to True.
        """
        if not data:
            cprint("No data provided to add.", "yellow")
            return

        processed_data_for_faiss = []
        processed_data_for_db = []

        # Process in batches for embedding
        for batch in grouper(data, self.batch_size):
            batch_to_process = [item.copy(deep=True) for item in batch] # Work on copies

            # Generate embeddings if using FastEmbed and vectors are missing
            if self.use_fastembed and any(item.vector is None for item in batch_to_process):
                 cprint(f"Generating embeddings for batch of size {len(batch_to_process)}...", "cyan")
                 batch_to_process = self._embed_batch(batch_to_process)

            # Prepare for FAISS (only dense vectors) and DB
            for item in batch_to_process:
                 if item.vector is not None:
                     # Add to FAISS list only if it's a dense numpy array
                     if isinstance(item.vector, np.ndarray) and item.vector.ndim == 1:
                          processed_data_for_faiss.append(item)
                     # Always add to DB list (will store serialized vector)
                     processed_data_for_db.append(item)
                 elif store: # Add to DB even if vector is None, if store is True
                      processed_data_for_db.append(item)


        # Add dense vectors to FAISS index
        if processed_data_for_faiss:
            self._add_to_faiss_index(processed_data_for_faiss)
        else:
             cprint("No suitable dense vectors found to add to FAISS index.", "yellow")

        # Store all processed data (including non-dense vectors) to DB if requested
        if store and processed_data_for_db:
            self.store_to_db(processed_data_for_db)
        elif not store:
             cprint("Skipping database storage as requested.", "yellow")
        elif not processed_data_for_db:
             cprint("No data to store in the database.", "yellow")


    def _load_or_build_faiss_index(self) -> faiss.IndexIDMap2:
        """Loads the FAISS index from file or builds a new one."""
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file):
            try:
                cprint(f"Attempting to load FAISS index from {index_file}...", "cyan")
                with open(index_file, "rb") as f:
                    index_data = pickle.load(f)
                    index = faiss.deserialize_index(index_data)
                # Check if the loaded index is IndexIDMap2, otherwise wrap it
                if not isinstance(index, faiss.IndexIDMap2):
                     cprint("Loaded index is not IndexIDMap2, wrapping it.", "yellow")
                     flat_index = index # Assuming the loaded index is the flat part
                     new_index = faiss.IndexIDMap2(flat_index)
                     # Note: This loses existing IDs if the saved index wasn't IDMap2
                     index = new_index

                # Verify dimension
                if index.d != self.dimension:
                     raise ValueError(f"Loaded index dimension ({index.d}) does not match required dimension ({self.dimension}).")
                cprint(f"FAISS index '{self.index_name}' loaded successfully ({index.ntotal} vectors).", "green")
                return index
            except (EOFError, pickle.UnpicklingError, ValueError, TypeError, Exception) as e:
                cprint(f"Failed to load or validate FAISS index '{self.index_name}': {e}. Building a new one.", "yellow")
                return self._build_faiss_index(self.dimension)
        else:
            cprint(f"FAISS index file '{index_file}' not found. Building a new one.", "cyan")
            return self._build_faiss_index(self.dimension)

    def _build_faiss_index(self, dimension: int) -> faiss.IndexIDMap2:
        """Builds a new FAISS index for dense vectors."""
        try:
            flat_index = faiss.IndexFlatL2(dimension) # Using L2 distance for FAISS
            index = faiss.IndexIDMap2(flat_index)
            cprint(f"New FAISS index '{self.index_name}' created successfully (Dimension: {dimension}).", "green")
            return index
        except Exception as e:
             cprint(f"Error building FAISS index: {e}", "red")
             raise

    def _add_to_faiss_index(self, data: List[IndexData]) -> None:
        """Adds dense vector data to the FAISS index."""
        ids = []
        vectors = []
        for point in data:
            # Ensure vector is a numpy array and has the correct dimension
            if isinstance(point.vector, np.ndarray) and point.vector.ndim == 1:
                 if point.vector.shape[0] != self.dimension:
                      cprint(f"Skipping vector with ID {point.id}: dimension mismatch ({point.vector.shape[0]} != {self.dimension})", "yellow")
                      continue
                 ids.append(point.id)
                 vectors.append(point.vector)
            else:
                 cprint(f"Skipping non-dense or invalid vector for FAISS with ID {point.id}", "yellow")


        if not vectors:
             cprint("No valid dense vectors to add to FAISS index.", "yellow")
             return

        try:
            ids_np = np.array(ids, dtype=np.int64)
            vectors_np = np.array(vectors, dtype=np.float32)
            self.index.add_with_ids(vectors_np, ids_np)
            cprint(f"Added {len(vectors)} dense vectors to FAISS index '{self.index_name}'.", "green")
        except Exception as e:
             cprint(f"Error adding vectors to FAISS index: {e}", "red")
             # Consider potential dimension mismatches or other FAISS errors

    def save_index(self) -> None:
        """Saves the FAISS index to a file."""
        index_file = f"{self.index_name}.pkl"
        try:
            chunk = faiss.serialize_index(self.index)
            with open(index_file, "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"FAISS index '{self.index_name}' saved successfully to {index_file}.", "green")
        except Exception as e:
            cprint(f"Error saving FAISS index: {e}", "red")

    def load_index(self) -> faiss.IndexIDMap2:
        """Loads the FAISS index from file."""
        # This method is now primarily for explicit loading if needed,
        # as init handles load/build automatically.
        self.index = self._load_or_build_faiss_index()
        return self.index

    def _create_table(self) -> None:
        """Creates the SQLite table if it doesn't exist."""
        try:
            with closing(self.connection.cursor()) as cur:
                # Store vectors as BLOBs now
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(
                        id INTEGER PRIMARY KEY,
                        content TEXT,
                        metadata TEXT,
                        vector BLOB,
                        vector_type TEXT
                    )"""
                )
                # Add index on vector_type for potentially faster filtering
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.index_name}_vector_type ON {self.index_name}(vector_type)")
            # cprint(f"Database table '{self.index_name}' checked/created.", "cyan") # Less verbose
        except sqlite3.Error as e:
            cprint(f"Error creating database table '{self.index_name}': {e}", "red")
            raise

    def store_to_db(self, data: List[IndexData]) -> None:
        """Stores data (including serialized vectors) to the SQLite database."""
        if not data:
            cprint("No data provided to store.", "yellow")
            return
        try:
            values = []
            for point in data:
                serialized_vector = _serialize_vector(point.vector)
                # Ensure content is stored as string (handle list for late interaction)
                content_str = json.dumps(point.content) if isinstance(point.content, list) else str(point.content)
                values.append((
                    point.id,
                    content_str,
                    json.dumps(point.metadata), # Store metadata as JSON string
                    serialized_vector,
                    point.vector_type
                ))

            with closing(self.connection.cursor()) as cur:
                cur.executemany(
                    f"""INSERT OR REPLACE INTO {self.index_name} (id, content, metadata, vector, vector_type)
                        VALUES (?,?,?,?,?)""", values
                )
            cprint(f"Stored/updated {len(data)} records in database table '{self.index_name}'.", "green")

        except sqlite3.Error as e:
            cprint(f"Database error during store_to_db: {e}", "red")
            raise
        except Exception as e:
            cprint(f"Unexpected error during store_to_db: {e}", "red")
            raise

    def search(self,
               query: Union[str, np.ndarray],
               k: int = 5,
               vector_type: str = "dense",
               rerank: bool = False,
               target_content_field: str = 'content'
               ) -> List[Dict[str, Any]]:
        """
        Performs search and retrieval.

        Handles dense (FAISS), sparse (custom), late interaction (custom), and image search.
        Optionally reranks results if using FastEmbed.

        Args:
            query (Union[str, np.ndarray]): The query (text, image path, or precomputed vector).
            k (int, optional): The number of results to return (before reranking). Defaults to 5.
            vector_type (str, optional): The type of search ('dense', 'sparse', 'late_interaction', 'image'). Defaults to 'dense'.
            rerank (bool, optional): Whether to rerank the results using FastEmbed's reranker. Defaults to False.
                                     Requires `use_fastembed=True` and `reranker_model_name` to be set.
            target_content_field(str, optional): The name of the field in the payload to use for reranking. Defaults to 'content'.


        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the retrieved item's
                                  ID, content, metadata, and score. Sorted by score (descending).
        """
        cprint(f"\n--- Starting Search (Type: {vector_type}, k={k}, Rerank: {rerank}) ---", "blue")
        cprint(f"Query: {str(query)[:100]}{'...' if isinstance(query, str) and len(query) > 100 else ''}", "cyan")

        initial_results: List[Tuple[float, int]] = [] # List of (score, id)

        if vector_type == "dense":
            query_vector = self._get_query_vector(query, "dense")
            if query_vector is None: return []
            distances, ids = self._search_faiss_index(query_vector, k)
            # Convert L2 distance to similarity score (e.g., 1 / (1 + distance))
            initial_results = [(1.0 / (1.0 + d), i) for d, i in zip(distances, ids) if i != -1] # Filter invalid IDs

        elif vector_type == "sparse":
             if not self.use_fastembed or not isinstance(self.embedding_model, SparseTextEmbedding):
                 cprint("Sparse search requires use_fastembed=True and a SparseTextEmbedding model.", "red")
                 return []
             query_vector = self._get_query_vector(query, "sparse")
             if query_vector is None: return []
             initial_results = self._search_sparse_sqlite(query_vector, k)

        elif vector_type == "late_interaction":
             if not self.use_fastembed or not isinstance(self.embedding_model, LateInteractionTextEmbedding):
                 cprint("Late interaction search requires use_fastembed=True and a LateInteractionTextEmbedding model.", "red")
                 return []
             query_vector = self._get_query_vector(query, "late_interaction")
             if query_vector is None: return []
             initial_results = self._search_late_interaction_sqlite(query_vector, k)

        elif vector_type == "image":
             query_vector = self._get_query_vector(query, "image")
             if query_vector is None: return []
             distances, ids = self._search_faiss_index(query_vector, k)
             initial_results = [(1.0 / (1.0 + d), i) for d, i in zip(distances, ids) if i != -1]

        else:
            cprint(f"Unsupported vector_type for search: {vector_type}", "red")
            return []

        if not initial_results:
            cprint("No initial results found.", "yellow")
            return []

        # Sort initial results by score (descending)
        initial_results.sort(key=lambda x: x[0], reverse=True)
        initial_ids = [id_ for score, id_ in initial_results]
        cprint(f"Initial search found {len(initial_ids)} candidates.", "cyan")

        # Retrieve content and metadata for the initial results
        retrieved_docs_raw = self.retrieve(initial_ids)
        if not retrieved_docs_raw:
             cprint("Could not retrieve document details for initial results.", "yellow")
             return []

        # Create a mapping from ID to initial score and retrieved data
        initial_score_map = {id_: score for score, id_ in initial_results}
        retrieved_data_map = {row[0]: {"id": row[0], "content": row[1], "metadata": json.loads(row[2] or '{}')} for row in retrieved_docs_raw}


        # Prepare results in the desired format, preserving initial order
        results_list = []
        for id_ in initial_ids:
             if id_ in retrieved_data_map:
                  doc = retrieved_data_map[id_]
                  # Try to parse content if it looks like JSON (for late interaction lists)
                  try:
                      parsed_content = json.loads(doc['content'])
                  except (json.JSONDecodeError, TypeError):
                      parsed_content = doc['content'] # Keep as string otherwise

                  results_list.append({
                       "id": doc["id"],
                       "score": initial_score_map[id_],
                       "content": parsed_content,
                       "metadata": doc["metadata"]
                  })

        # Apply reranking if requested
        if rerank:
            if not self.use_fastembed or not self.reranker_model:
                cprint("Reranking requested but FastEmbed reranker is not available/configured.", "yellow")
            else:
                cprint(f"Reranking top {len(results_list)} results...", "cyan")
                try:
                    # Prepare documents for reranker (query, passage) pairs
                    passages_for_rerank = [item.get(target_content_field, "") for item in results_list]
                    # Ensure passages are strings
                    passages_for_rerank = [str(p) if not isinstance(p, str) else p for p in passages_for_rerank]

                    if isinstance(query, np.ndarray):
                         cprint("Cannot rerank with a precomputed vector query. Please provide the original query text.", "yellow")
                    else:
                         reranked_scores = self.reranker_model.rerank(query, passages_for_rerank)

                         # Update scores and re-sort
                         for i, item in enumerate(results_list):
                              item["score"] = reranked_scores[i] # Reranker provides new scores
                         results_list.sort(key=lambda x: x["score"], reverse=True)
                         cprint("Reranking complete.", "green")

                except Exception as e:
                    cprint(f"Error during reranking: {e}", "red")
                    # Proceed with non-reranked results

        cprint(f"--- Search Complete ---", "blue")
        return results_list


    def _get_query_vector(self, query: Union[str, np.ndarray], vector_type: str) -> Optional[np.ndarray]:
        """Generates or validates the query vector."""
        query_vector: Optional[Union[np.ndarray, List[np.ndarray], SparseEmbedding]] = None

        if isinstance(query, np.ndarray):
            # Use precomputed vector
            query_vector = query
            # Basic validation for dense vectors
            if vector_type == "dense":
                 if query.ndim != 2 or query.shape[0] != 1 or query.shape[1] != self.dimension:
                      cprint(f"Invalid precomputed dense query vector shape: {query.shape}. Expected (1, {self.dimension})", "red")
                      return None
                 query_vector = query.astype(np.float32) # Ensure correct dtype for FAISS
            # Add validation for other precomputed types if needed
            cprint("Using precomputed query vector.", "cyan")

        elif isinstance(query, str):
            # Generate vector using FastEmbed
            if not self.use_fastembed:
                cprint("Cannot generate query vector: use_fastembed is False and query is text.", "red")
                return None
            try:
                is_image = vector_type == "image"
                if is_image and not os.path.exists(query):
                     cprint(f"Image query path does not exist: {query}", "red")
                     return None

                cprint(f"Generating query vector (type: {vector_type})...", "cyan")
                embedding_gen = self._get_embedding_generator(query, vector_type, is_query=True)
                query_vector = next(embedding_gen) # Get the first (and only) embedding

                # Reshape dense vectors for FAISS search
                if vector_type == "dense" and isinstance(query_vector, np.ndarray):
                     if query_vector.ndim == 1:
                          query_vector = np.expand_dims(query_vector, axis=0).astype(np.float32)
                     elif query_vector.ndim == 2 and query_vector.shape[0] == 1:
                          query_vector = query_vector.astype(np.float32)
                     else:
                          cprint(f"Unexpected dense query vector shape from FastEmbed: {query_vector.shape}", "red")
                          return None

            except StopIteration:
                 cprint("FastEmbed query embedding generation returned no result.", "red")
                 return None
            except Exception as e:
                 cprint(f"Error generating query vector: {e}", "red")
                 return None
        else:
            cprint(f"Invalid query type: {type(query)}. Must be str or np.ndarray.", "red")
            return None

        return query_vector


    def _search_faiss_index(self, query_vector: np.ndarray, k: int) -> Tuple[List[float], List[int]]:
        """Searches the FAISS index for dense vectors."""
        if self.index.ntotal == 0:
            cprint("FAISS index is empty. Cannot search.", "yellow")
            return [], []
        try:
            # Ensure query_vector is float32 and 2D
            if query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                 query_vector = np.expand_dims(query_vector, axis=0)

            cprint(f"Searching FAISS index with vector shape: {query_vector.shape}", "cyan")
            distances, ids = self.index.search(query_vector, k)

            # Process results
            distances_list = [float(d) for d in distances[0]]
            ids_list = [int(i) for i in ids[0]] # FAISS returns int64
            cprint(f"FAISS search results - Distances: {distances_list}, IDs: {ids_list}", "cyan")
            return distances_list, ids_list
        except Exception as e:
            cprint(f"Error during FAISS search: {e}", "red")
            return [], []

    def _search_sparse_sqlite(self, query_sparse_vector: SparseEmbedding, k: int) -> List[Tuple[float, int]]:
        """Retrieves all sparse vectors from SQLite and calculates dot product similarity."""
        if not FASTEMBED_AVAILABLE: raise RuntimeError("FastEmbed required for sparse search.")
        cprint("Performing sparse search via SQLite retrieval and dot product...", "cyan")
        scores = []
        try:
            with closing(self.connection.cursor()) as cur:
                cur.execute(f"SELECT id, vector FROM {self.index_name} WHERE vector_type = 'sparse'")
                while True:
                    batch = cur.fetchmany(self.batch_size)
                    if not batch:
                        break
                    for db_id, vector_blob in batch:
                        doc_sparse_vector = _deserialize_vector(vector_blob, 'sparse')
                        if doc_sparse_vector:
                            # Calculate dot product for sparse vectors
                            score = 0.0
                            query_map = dict(zip(query_sparse_vector.indices, query_sparse_vector.values))
                            doc_map = dict(zip(doc_sparse_vector.indices, doc_sparse_vector.values))
                            for index, value in query_map.items():
                                if index in doc_map:
                                    score += value * doc_map[index]
                            scores.append((float(score), int(db_id)))
        except sqlite3.Error as e:
            cprint(f"Database error during sparse search: {e}", "red")
            return []
        except Exception as e:
            cprint(f"Unexpected error during sparse search: {e}", "red")
            return []

        scores.sort(key=lambda x: x[0], reverse=True) # Sort by score descending
        cprint(f"Calculated {len(scores)} sparse scores.", "cyan")
        return scores[:k]

    def _search_late_interaction_sqlite(self, query_li_vector: List[np.ndarray], k: int) -> List[Tuple[float, int]]:
        """Retrieves late interaction vectors from SQLite and calculates MaxSim score."""
        if not FASTEMBED_AVAILABLE: raise RuntimeError("FastEmbed required for late interaction search.")
        cprint("Performing late interaction search via SQLite retrieval and MaxSim...", "cyan")
        scores = []
        query_matrix = np.array(query_li_vector, dtype=np.float32) # Shape: [query_len, dim]

        try:
            with closing(self.connection.cursor()) as cur:
                cur.execute(f"SELECT id, vector FROM {self.index_name} WHERE vector_type = 'late_interaction'")
                while True:
                    batch = cur.fetchmany(self.batch_size)
                    if not batch:
                        break
                    for db_id, vector_blob in batch:
                        doc_li_vectors = _deserialize_vector(vector_blob, 'late_interaction')
                        if doc_li_vectors:
                            doc_matrix = np.array(doc_li_vectors, dtype=np.float32) # Shape: [doc_len, dim]
                            # Calculate MaxSim score
                            # Similarity matrix: [query_len, doc_len]
                            sim_matrix = np.dot(query_matrix, doc_matrix.T)
                            # Max similarity for each query token across all doc tokens
                            max_sim_per_query_token = np.max(sim_matrix, axis=1)
                            # Sum of max similarities
                            score = np.sum(max_sim_per_query_token)
                            scores.append((float(score), int(db_id)))
        except sqlite3.Error as e:
            cprint(f"Database error during late interaction search: {e}", "red")
            return []
        except Exception as e:
            cprint(f"Unexpected error during late interaction search: {e}", "red")
            return []

        scores.sort(key=lambda x: x[0], reverse=True) # Sort by score descending
        cprint(f"Calculated {len(scores)} late interaction scores.", "cyan")
        return scores[:k]


    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """Retrieves data (id, content, metadata) from SQLite based on IDs."""
        if not ids:
            cprint("Retrieve called with empty ID list.", "yellow")
            return []

        # Ensure IDs are standard Python integers
        safe_ids = [int(i) for i in ids]

        rows = []
        try:
            with closing(self.connection.cursor()) as cur:
                placeholders = ','.join('?' * len(safe_ids))
                sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"
                cur.execute(sql, safe_ids)
                rows = cur.fetchall()
                cprint(f"Retrieved {len(rows)} records from database for {len(safe_ids)} IDs.", "cyan")
        except sqlite3.Error as e:
            cprint(f"Database error during retrieve: {e}", "red")
            raise
        except Exception as e:
             cprint(f"An unexpected error occurred during retrieve: {e}", "red")
             raise
        return rows

    def __del__(self):
         """Closes the database connection when the object is deleted."""
         if hasattr(self, "connection") and self.connection:
             try:
                 self.connection.close()
                 cprint(f"Database connection closed for index '{self.index_name}'.", "yellow")
             except Exception as e:
                  cprint(f"Error closing database connection for index '{self.index_name}': {e}", "red")

    def usage(self):
        """Prints usage instructions for the FqlDb class."""
        cprint("\n--- FqlDb Usage ---", "blue")
        cprint("Initialization:", "green")
        cprint("  # Default (FAISS + SQLite, provide external vectors)", "white")
        cprint("  fql_db_default = FqlDb(index_name='my_index', dimension=384, db_name='my_db.db')", "white")
        cprint("  # FastEmbed (Dense Text)", "white")
        cprint("  fql_db_fe_dense = FqlDb(index_name='fe_dense', use_fastembed=True, model_name='BAAI/bge-small-en-v1.5')", "white")
        cprint("  # FastEmbed (Sparse Text - SPLADE)", "white")
        cprint("  fql_db_fe_sparse = FqlDb(index_name='fe_sparse', use_fastembed=True, model_name='prithvida/Splade_PP_en_v1', dimension=1) # Dim needed for dummy FAISS", "white")
        cprint("  # FastEmbed (Late Interaction - ColBERT)", "white")
        cprint("  fql_db_fe_li = FqlDb(index_name='fe_li', use_fastembed=True, model_name='colbert-ir/colbertv2.0', dimension=128)", "white")
        cprint("  # FastEmbed (Image)", "white")
        cprint("  fql_db_fe_img = FqlDb(index_name='fe_img', use_fastembed=True, model_name='Qdrant/clip-ViT-B-32-vision', dimension=512)", "white")
        cprint("  # FastEmbed (With Reranker)", "white")
        cprint("  fql_db_fe_rerank = FqlDb(index_name='fe_rerank', use_fastembed=True, model_name='BAAI/bge-small-en-v1.5', reranker_model_name='BAAI/bge-reranker-base')", "white")

        cprint("\nAdding Data:", "green")
        cprint("  # Prepare data (vector can be None if use_fastembed=True)", "white")
        cprint("  data = [", "white")
        cprint("      IndexData(id=1, content='Some text', vector_type='dense', vector=np.array([...])) # Default mode", "white")
        cprint("      IndexData(id=2, content='FastEmbed text', vector_type='dense') # FastEmbed dense", "white")
        cprint("      IndexData(id=3, content='FastEmbed sparse text', vector_type='sparse') # FastEmbed sparse", "white")
        cprint("      IndexData(id=4, content='path/to/image.jpg', vector_type='image') # FastEmbed image", "white")
        cprint("      IndexData(id=5, content='ColBERT document text', vector_type='late_interaction') # FastEmbed LI", "white")
        cprint("  ]", "white")
        cprint("  fql_db.add(data=data, store=True)", "white")

        cprint("\nSearching Data:", "green")
        cprint("  # Dense Search (FAISS or FastEmbed Dense)", "white")
        cprint("  results_dense = fql_db.search(query='my dense query', k=5, vector_type='dense')", "white")
        cprint("  # Sparse Search (FastEmbed Sparse only)", "white")
        cprint("  results_sparse = fql_db_fe_sparse.search(query='my sparse query', k=5, vector_type='sparse')", "white")
        cprint("  # Late Interaction Search (FastEmbed LI only)", "white")
        cprint("  results_li = fql_db_fe_li.search(query='my colbert query', k=5, vector_type='late_interaction')", "white")
        cprint("  # Image Search (FastEmbed Image only)", "white")
        cprint("  results_img = fql_db_fe_img.search(query='path/to/query_image.jpg', k=5, vector_type='image')", "white")
        cprint("  # Search with Reranking (FastEmbed with reranker only)", "white")
        cprint("  results_reranked = fql_db_fe_rerank.search(query='query to rerank', k=10, vector_type='dense', rerank=True)", "white")

        cprint("\nRetrieving Data by ID:", "green")
        cprint("  retrieved_data = fql_db.retrieve(ids=[1, 2, 3])", "white")

        cprint("\nSaving FAISS Index:", "green")
        cprint("  fql_db.save_index() # Saves the index for dense vectors", "white")
        cprint("---------------------", "blue")


# --- Test Functions ---

def test_fql_db():
    """Tests the FqlDb class with both default and FastEmbed configurations."""
    cprint("\n========== Starting FqlDb Tests ==========", "blue", attrs=["bold"])

    # --- Test Parameters ---
    index_name_default = "test_default_index"
    index_name_fe_dense = "test_fe_dense_index"
    index_name_fe_sparse = "test_fe_sparse_index"
    index_name_fe_li = "test_fe_li_index"
    index_name_fe_img = "test_fe_img_index"
    index_name_fe_rerank = "test_fe_rerank_index"

    dimension_dense = 4 # Small dimension for testing default/dense
    dimension_sparse = 1 # Dummy dimension for FAISS index when using sparse
    dimension_li = 4 # Using small dim model for LI test
    dimension_img = 512 # Dimension for CLIP vision
    dimension_rerank = dimension_dense

    db_name = "test_fql.db"

    # FastEmbed Models (using small/fast ones for testing)
    # Note: Ensure these models are compatible with the dimensions chosen
    fe_model_dense = "sentence-transformers/all-MiniLM-L6-v2" # Dim 384 - Adjust test dim or model
    fe_model_sparse = "Qdrant/bm25" # Example sparse model
    fe_model_li = "colbert-ir/colbertv2.0" # Dim 128 - Adjust test dim or model
    fe_model_img = "Qdrant/clip-ViT-B-32-vision" # Dim 512
    fe_reranker = "Xenova/ms-marco-MiniLM-L-6-v2" # Example reranker

    # Adjust dimensions based on actual test models if needed
    dimension_dense = 384 # Matching all-MiniLM-L6-v2
    dimension_li = 128 # Matching colbertv2.0
    dimension_rerank = dimension_dense

    # --- Cleanup before starting ---
    cleanup_files(
        f"{index_name_default}.pkl", f"{index_name_fe_dense}.pkl",
        f"{index_name_fe_sparse}.pkl", f"{index_name_fe_li}.pkl",
        f"{index_name_fe_img}.pkl", f"{index_name_fe_rerank}.pkl",
        db_name
    )

    # --- Test Instances ---
    fql_db_default = None
    fql_db_fe_dense = None
    fql_db_fe_sparse = None
    fql_db_fe_li = None
    fql_db_fe_img = None
    fql_db_fe_rerank = None

    try:
        # === Test 1: Default Mode (FAISS + SQLite, External Vectors) ===
        cprint("\n--- Test 1: Default Mode ---", "yellow", attrs=["underline"])
        fql_db_default = FqlDb(index_name=index_name_default, dimension=dimension_dense, db_name=db_name)

        default_data = [
            IndexData(id=101, content="Default text one", vector_type='dense', vector=np.random.rand(dimension_dense).astype(np.float32)),
            IndexData(id=102, content="Default text two", vector_type='dense', vector=np.random.rand(dimension_dense).astype(np.float32)),
        ]
        fql_db_default.add(default_data, store=True)
        assert fql_db_default.index.ntotal == 2
        cprint("Default: Added data.", "green")

        query_vec_default = np.random.rand(1, dimension_dense).astype(np.float32)
        results_default = fql_db_default.search(query=query_vec_default, k=1, vector_type='dense')
        assert len(results_default) == 1
        cprint(f"Default: Search result: {results_default[0]['id']}", "green")

        retrieved_default = fql_db_default.retrieve(ids=[res['id'] for res in results_default])
        assert len(retrieved_default) == 1
        cprint(f"Default: Retrieved content: {retrieved_default[0][1]}", "green")

        fql_db_default.save_index()
        del fql_db_default # Explicitly delete to close connection before potential reload

        # Reload and test
        fql_db_default_loaded = FqlDb(index_name=index_name_default, dimension=dimension_dense, db_name=db_name)
        assert fql_db_default_loaded.index.ntotal == 2
        results_default_loaded = fql_db_default_loaded.search(query=query_vec_default, k=1, vector_type='dense')
        assert len(results_default_loaded) == 1
        assert results_default_loaded[0]['id'] == results_default[0]['id']
        cprint("Default: Reload and search successful.", "green")
        del fql_db_default_loaded # Cleanup loaded instance

        cprint("--- Test 1 Passed ---", "green", attrs=["bold"])

        # === Test 2: FastEmbed Dense Mode ===
        if FASTEMBED_AVAILABLE:
            cprint("\n--- Test 2: FastEmbed Dense Mode ---", "yellow", attrs=["underline"])
            fql_db_fe_dense = FqlDb(index_name=index_name_fe_dense, use_fastembed=True, model_name=fe_model_dense, db_name=db_name, dimension=dimension_dense)

            fe_dense_data = [
                IndexData(id=201, content="FastEmbed dense text one", vector_type='dense'),
                IndexData(id=202, content="Another FastEmbed dense document", vector_type='dense'),
            ]
            fql_db_fe_dense.add(fe_dense_data, store=True)
            assert fql_db_fe_dense.index.ntotal == 2
            cprint("FE Dense: Added data.", "green")

            query_fe_dense = "query for dense fastembed"
            results_fe_dense = fql_db_fe_dense.search(query=query_fe_dense, k=1, vector_type='dense')
            assert len(results_fe_dense) == 1
            cprint(f"FE Dense: Search result ID: {results_fe_dense[0]['id']}", "green")

            retrieved_fe_dense = fql_db_fe_dense.retrieve(ids=[res['id'] for res in results_fe_dense])
            assert len(retrieved_fe_dense) == 1
            cprint(f"FE Dense: Retrieved content: {retrieved_fe_dense[0][1]}", "green")

            fql_db_fe_dense.save_index()
            del fql_db_fe_dense

            # Reload and test
            fql_db_fe_dense_loaded = FqlDb(index_name=index_name_fe_dense, use_fastembed=True, model_name=fe_model_dense, db_name=db_name, dimension=dimension_dense)
            assert fql_db_fe_dense_loaded.index.ntotal == 2
            results_fe_dense_loaded = fql_db_fe_dense_loaded.search(query=query_fe_dense, k=1, vector_type='dense')
            assert len(results_fe_dense_loaded) == 1
            assert results_fe_dense_loaded[0]['id'] == results_fe_dense[0]['id']
            cprint("FE Dense: Reload and search successful.", "green")
            del fql_db_fe_dense_loaded

            cprint("--- Test 2 Passed ---", "green", attrs=["bold"])
        else:
            cprint("\n--- Skipping Test 2: FastEmbed Dense Mode (FastEmbed not installed) ---", "yellow")


        # === Test 3: FastEmbed Sparse Mode ===
        if FASTEMBED_AVAILABLE:
            cprint("\n--- Test 3: FastEmbed Sparse Mode ---", "yellow", attrs=["underline"])
            # Dimension is needed for FAISS init, even if not used for sparse search itself
            fql_db_fe_sparse = FqlDb(index_name=index_name_fe_sparse, use_fastembed=True, model_name=fe_model_sparse, db_name=db_name, dimension=dimension_sparse)

            fe_sparse_data = [
                IndexData(id=301, content="Search for sparse vectors", vector_type='sparse'),
                IndexData(id=302, content="Another sparse document example", vector_type='sparse'),
            ]
            # Add generates vectors and stores in DB, FAISS index remains empty/unused for sparse
            fql_db_fe_sparse.add(fe_sparse_data, store=True)
            assert fql_db_fe_sparse.index.ntotal == 0 # FAISS index not used for sparse
            cprint("FE Sparse: Added data (stored in DB).", "green")

            query_fe_sparse = "query sparse"
            # Search retrieves from DB and calculates scores
            results_fe_sparse = fql_db_fe_sparse.search(query=query_fe_sparse, k=1, vector_type='sparse')
            assert len(results_fe_sparse) >= 1 # BM25 might return results even for less overlap
            cprint(f"FE Sparse: Search result ID: {results_fe_sparse[0]['id']}", "green")

            retrieved_fe_sparse = fql_db_fe_sparse.retrieve(ids=[res['id'] for res in results_fe_sparse])
            assert len(retrieved_fe_sparse) >= 1
            cprint(f"FE Sparse: Retrieved content: {retrieved_fe_sparse[0][1]}", "green")

            # No FAISS index to save/load for sparse in this setup
            cprint("FE Sparse: Save/Load not applicable to FAISS index for sparse.", "cyan")
            del fql_db_fe_sparse

            cprint("--- Test 3 Passed ---", "green", attrs=["bold"])
        else:
            cprint("\n--- Skipping Test 3: FastEmbed Sparse Mode (FastEmbed not installed) ---", "yellow")


        # === Test 4: FastEmbed Late Interaction Mode ===
        if FASTEMBED_AVAILABLE:
            cprint("\n--- Test 4: FastEmbed Late Interaction Mode ---", "yellow", attrs=["underline"])
            fql_db_fe_li = FqlDb(index_name=index_name_fe_li, use_fastembed=True, model_name=fe_model_li, db_name=db_name, dimension=dimension_li)

            fe_li_data = [
                IndexData(id=401, content="ColBERT late interaction model test", vector_type='late_interaction'),
                IndexData(id=402, content="Testing multi-vector representations", vector_type='late_interaction'),
            ]
            # Add generates vectors and stores in DB
            fql_db_fe_li.add(fe_li_data, store=True)
            assert fql_db_fe_li.index.ntotal == 0 # FAISS index not used
            cprint("FE Late Interaction: Added data (stored in DB).", "green")

            query_fe_li = "late interaction query"
            # Search retrieves from DB and calculates MaxSim scores
            results_fe_li = fql_db_fe_li.search(query=query_fe_li, k=1, vector_type='late_interaction')
            assert len(results_fe_li) == 1
            cprint(f"FE Late Interaction: Search result ID: {results_fe_li[0]['id']}", "green")

            retrieved_fe_li = fql_db_fe_li.retrieve(ids=[res['id'] for res in results_fe_li])
            assert len(retrieved_fe_li) == 1
            cprint(f"FE Late Interaction: Retrieved content: {retrieved_fe_li[0][1]}", "green")

            # No FAISS index to save/load
            cprint("FE Late Interaction: Save/Load not applicable to FAISS index.", "cyan")
            del fql_db_fe_li

            cprint("--- Test 4 Passed ---", "green", attrs=["bold"])
        else:
            cprint("\n--- Skipping Test 4: FastEmbed Late Interaction Mode (FastEmbed not installed) ---", "yellow")


        # === Test 5: FastEmbed Image Mode ===
        if FASTEMBED_AVAILABLE:
            cprint("\n--- Test 5: FastEmbed Image Mode ---", "yellow", attrs=["underline"])
            # Create dummy image files for testing
            try:
                from PIL import Image
                dummy_img_path1 = "test_img1.png"
                dummy_img_path2 = "test_img2.png"
                Image.new('RGB', (60, 30), color = 'red').save(dummy_img_path1)
                Image.new('RGB', (60, 30), color = 'blue').save(dummy_img_path2)
                cprint("Created dummy image files.", "cyan")
            except ImportError:
                cprint("PIL/Pillow not installed, cannot create dummy images for test.", "red")
                raise # Need Pillow for this test

            fql_db_fe_img = FqlDb(index_name=index_name_fe_img, use_fastembed=True, model_name=fe_model_img, db_name=db_name, dimension=dimension_img)

            fe_img_data = [
                IndexData(id=501, content=dummy_img_path1, vector_type='image', metadata={"color": "red"}),
                IndexData(id=502, content=dummy_img_path2, vector_type='image', metadata={"color": "blue"}),
            ]
            # Add generates image embeddings and adds to FAISS/DB
            fql_db_fe_img.add(fe_img_data, store=True)
            assert fql_db_fe_img.index.ntotal == 2
            cprint("FE Image: Added image data.", "green")

            query_fe_img = dummy_img_path1 # Search for the red image
            results_fe_img = fql_db_fe_img.search(query=query_fe_img, k=1, vector_type='image')
            assert len(results_fe_img) == 1
            assert results_fe_img[0]['id'] == 501 # Should find itself
            cprint(f"FE Image: Search result ID: {results_fe_img[0]['id']}", "green")

            retrieved_fe_img = fql_db_fe_img.retrieve(ids=[res['id'] for res in results_fe_img])
            assert len(retrieved_fe_img) == 1
            cprint(f"FE Image: Retrieved path: {retrieved_fe_img[0][1]}", "green")

            fql_db_fe_img.save_index()
            del fql_db_fe_img

            # Reload and test
            fql_db_fe_img_loaded = FqlDb(index_name=index_name_fe_img, use_fastembed=True, model_name=fe_model_img, db_name=db_name, dimension=dimension_img)
            assert fql_db_fe_img_loaded.index.ntotal == 2
            results_fe_img_loaded = fql_db_fe_img_loaded.search(query=query_fe_img, k=1, vector_type='image')
            assert len(results_fe_img_loaded) == 1
            assert results_fe_img_loaded[0]['id'] == results_fe_img[0]['id']
            cprint("FE Image: Reload and search successful.", "green")
            del fql_db_fe_img_loaded

            cprint("--- Test 5 Passed ---", "green", attrs=["bold"])
        else:
            cprint("\n--- Skipping Test 5: FastEmbed Image Mode (FastEmbed not installed) ---", "yellow")


        # === Test 6: FastEmbed Reranking ===
        if FASTEMBED_AVAILABLE:
            cprint("\n--- Test 6: FastEmbed Reranking ---", "yellow", attrs=["underline"])
            fql_db_fe_rerank = FqlDb(index_name=index_name_fe_rerank, use_fastembed=True, model_name=fe_model_dense, reranker_model_name=fe_reranker, db_name=db_name, dimension=dimension_rerank)

            fe_rerank_data = [
                IndexData(id=601, content="The quick brown fox jumps over the lazy dog.", vector_type='dense'),
                IndexData(id=602, content="A fast brown canine leaps above a sleepy canine.", vector_type='dense'), # Semantically similar, different words
                IndexData(id=603, content="Weather is nice today.", vector_type='dense'), # Less relevant
            ]
            fql_db_fe_rerank.add(fe_rerank_data, store=True)
            assert fql_db_fe_rerank.index.ntotal == 3
            cprint("FE Rerank: Added data.", "green")

            query_fe_rerank = "fast fox"
            # Search without reranking first
            results_no_rerank = fql_db_fe_rerank.search(query=query_fe_rerank, k=3, vector_type='dense', rerank=False)
            cprint(f"FE Rerank (Before): {[res['id'] for res in results_no_rerank]} Scores: {[f'{res['score']:.4f}' for res in results_no_rerank]}", "cyan")

            # Search with reranking
            results_reranked = fql_db_fe_rerank.search(query=query_fe_rerank, k=3, vector_type='dense', rerank=True)
            cprint(f"FE Rerank (After):  {[res['id'] for res in results_reranked]} Scores: {[f'{res['score']:.4f}' for res in results_reranked]}", "cyan")

            assert len(results_reranked) == 3
            # Expect reranker to potentially change the order, likely promoting 601/602
            assert results_reranked[0]['id'] in [601, 602] # Top result should be one of the relevant ones
            cprint("FE Rerank: Reranking executed.", "green")
            del fql_db_fe_rerank

            cprint("--- Test 6 Passed ---", "green", attrs=["bold"])
        else:
             cprint("\n--- Skipping Test 6: FastEmbed Reranking (FastEmbed not installed) ---", "yellow")


        cprint("\n========== All FqlDb tests passed! ==========", "green", attrs=["bold"])

    except Exception as e:
        cprint(f"\n!!!!!!!!!! Test failed: {e} !!!!!!!!!!!", "red", attrs=["bold"])
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise # Re-raise exception to make test runner aware of failure
    finally:
        # --- Final Cleanup ---
        cprint("\n--- Final Cleanup ---", "blue")
        # Ensure connections are closed if objects still exist due to error
        instances = [
            fql_db_default, fql_db_fe_dense, fql_db_fe_sparse,
            fql_db_fe_li, fql_db_fe_img, fql_db_fe_rerank
        ]
        for instance in instances:
            if instance and hasattr(instance, 'connection') and instance.connection:
                try:
                    instance.connection.close()
                    instance.connection = None # Prevent __del__ trying again
                    cprint(f"Closed connection for {instance.index_name}", "yellow")
                except Exception as e_close:
                     cprint(f"Error closing connection for {instance.index_name}: {e_close}", "red")

        # Explicitly delete objects before file cleanup
        del fql_db_default, fql_db_fe_dense, fql_db_fe_sparse, fql_db_fe_li, fql_db_fe_img, fql_db_fe_rerank

        cleanup_files(
            f"{index_name_default}.pkl", f"{index_name_fe_dense}.pkl",
            f"{index_name_fe_sparse}.pkl", f"{index_name_fe_li}.pkl",
            f"{index_name_fe_img}.pkl", f"{index_name_fe_rerank}.pkl",
            db_name,
            "test_img1.png", "test_img2.png" # Dummy image files
        )
        cprint("---------------------", "blue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FqlDb or print usage.")
    parser.add_argument("--usage", action="store_true", help="Print FqlDb usage instructions.")
    args = parser.parse_args()

    if args.usage:
        # Need to instantiate with valid dimension even for usage
        temp_db = None
        try:
            temp_db = FqlDb(index_name='temp_usage', dimension=1, db_name='temp_usage.db')
            temp_db.usage()
        except Exception as e:
             cprint(f"Error generating usage: {e}", "red")
        finally:
             if temp_db: del temp_db # Ensure connection closes via __del__
             cleanup_files('temp_usage.pkl', 'temp_usage.db')
    else:
        test_fql_db()
