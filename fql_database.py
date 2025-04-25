import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Any, Generator, Iterable
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse
from pydantic import BaseModel as PydanticBaseModel, Field, model_validator
import math

# --- Data Schema ---
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class IndexData(BaseModel):
    """Represents data to be indexed, including content and optional vector."""
    id: int
    content: Union[str, bytes] # Store image bytes or text
    metadata: Dict = Field(default_factory=dict)
    vector: Optional[np.ndarray] = None # Dense vector
    sparse_vector: Optional[Dict[str, Union[List[int], List[float]]]] = None # For sparse embeddings {indices: [], values: []}
    token_vectors: Optional[np.ndarray] = None # For late interaction models [num_tokens, dim]

    @model_validator(mode='before')
    @classmethod
    def check_content_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        content = values.get('content')
        if not isinstance(content, (str, bytes)):
            raise TypeError("Content must be either string (text) or bytes (image)")
        return values

# --- FastEmbed Integration ---
try:
    from fastembed import (
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionTextEmbedding,
        SparseEmbedding,
        DefaultEmbedding, # Base class
        CrossEncoder,
        Query, # Type hint
        Document # Type hint
    )
    from PIL import Image # Needed for image handling with fastembed
    import io
    _FASTEMBED_AVAILABLE = True
except ImportError:
    _FASTEMBED_AVAILABLE = False
    # Define dummy types for type hinting if fastembed is not installed
    TextEmbedding = type('TextEmbedding', (object,), {})
    SparseTextEmbedding = type('SparseTextEmbedding', (object,), {})
    ImageEmbedding = type('ImageEmbedding', (object,), {})
    LateInteractionTextEmbedding = type('LateInteractionTextEmbedding', (object,), {})
    SparseEmbedding = type('SparseEmbedding', (object,), {})
    DefaultEmbedding = type('DefaultEmbedding', (object,), {})
    CrossEncoder = type('CrossEncoder', (object,), {})
    Query = type('Query', (object,), {})
    Document = type('Document', (object,), {})
    Image = type('Image', (object,), {}) # Dummy PIL.Image
    cprint("Warning: 'fastembed' or 'Pillow' not found. FastEmbed features will be disabled.", "yellow")


# --- Helper Functions ---
def grouper(iterable: Iterable, n: int) -> Generator[List, None, None]:
    """Iterate over data in chunks or blocks."""
    it = iter(iterable)
    group = []
    while True:
        try:
            for _ in range(n):
                group.append(next(it))
            yield group
            group = []
        except StopIteration:
            if group:
                yield group
            break

def cleanup_test_files(*file_paths: str):
    """Removes specified test files."""
    files_removed = False
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                files_removed = True
        except Exception as e:
            cprint(f"Error removing file {file_path}: {e}", "red")
    if files_removed:
        cprint("Test files removed.", "yellow")

# --- Main Class ---
class FqlDb:
    """
    A class combining vector similarity search (FAISS or FastEmbed) with a SQLite database.

    Supports dense, sparse, late-interaction, and image embeddings via FastEmbed.
    """

    def __init__(
        self,
        index_name: str,
        db_name: str = "fql.db",
        dimension: Optional[int] = None,
        use_fastembed: bool = False,
        fastembed_model_name: Optional[str] = None, # Dense or Image model
        fastembed_sparse_model_name: Optional[str] = None,
        fastembed_late_interaction_model_name: Optional[str] = None,
        fastembed_reranker_model_name: Optional[str] = None,
        fastembed_batch_size: int = 32,
        fastembed_cache_dir: Optional[str] = None,
        fastembed_providers: Optional[List[str]] = None, # e.g., ["CUDAExecutionProvider"]
        faiss_index_type: str = "flat", # 'flat' or potentially others later
        overwrite: bool = False,
    ):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): Base name for the index files and database table.
            db_name (str, optional): Name of the SQLite database file. Defaults to "fql.db".
            dimension (Optional[int]): Dimension of the dense vectors. Required if not using FastEmbed
                                       or if FastEmbed model dimension cannot be inferred.
            use_fastembed (bool, optional): Whether to use FastEmbed for embeddings and reranking. Defaults to False.
            fastembed_model_name (Optional[str]): Name of the FastEmbed dense text or image model.
            fastembed_sparse_model_name (Optional[str]): Name of the FastEmbed sparse text model.
            fastembed_late_interaction_model_name (Optional[str]): Name of the FastEmbed late interaction model.
            fastembed_reranker_model_name (Optional[str]): Name of the FastEmbed reranker model.
            fastembed_batch_size (int, optional): Batch size for FastEmbed processing. Defaults to 32.
            fastembed_cache_dir (Optional[str], optional): Directory to cache FastEmbed models. Defaults to None.
            fastembed_providers (Optional[List[str]], optional): ONNX Runtime providers for FastEmbed (e.g., ["CUDAExecutionProvider"]). Defaults to None (CPU).
            faiss_index_type (str, optional): Type of FAISS index for dense vectors ('flat'). Defaults to "flat".
            overwrite (bool, optional): If True, overwrite existing index and database table. Defaults to False.
        """
        self.index_name = index_name
        self.db_name = db_name
        self.dimension = dimension
        self.use_fastembed = use_fastembed and _FASTEMBED_AVAILABLE
        self.fastembed_model_name = fastembed_model_name
        self.fastembed_sparse_model_name = fastembed_sparse_model_name
        self.fastembed_late_interaction_model_name = fastembed_late_interaction_model_name
        self.fastembed_reranker_model_name = fastembed_reranker_model_name
        self.fastembed_batch_size = fastembed_batch_size
        self.fastembed_cache_dir = fastembed_cache_dir
        self.fastembed_providers = fastembed_providers
        self.faiss_index_type = faiss_index_type
        self.overwrite = overwrite

        self.dense_model: Optional[Union[TextEmbedding, ImageEmbedding]] = None
        self.sparse_model: Optional[SparseTextEmbedding] = None
        self.late_interaction_model: Optional[LateInteractionTextEmbedding] = None
        self.reranker_model: Optional[CrossEncoder] = None

        self._validate_init_params()

        if self.use_fastembed:
            self._initialize_fastembed_models()
            if self.dense_model and not self.dimension:
                 # Try to infer dimension from dense model
                 try:
                     # Assuming models have a 'dim' or similar attribute after init
                     if hasattr(self.dense_model, 'dim'):
                         self.dimension = self.dense_model.dim
                     elif hasattr(self.dense_model, 'model') and hasattr(self.dense_model.model, 'dim'):
                          self.dimension = self.dense_model.model.dim
                     else:
                         # Attempt embedding a dummy item to get dimension
                         dummy_content = "test" if isinstance(self.dense_model, TextEmbedding) else self._create_dummy_image_bytes()
                         if dummy_content:
                             dummy_emb = next(self.dense_model.embed([dummy_content]))
                             self.dimension = dummy_emb.shape[-1]
                         else:
                             raise ValueError("Could not infer dimension from FastEmbed dense model.")
                     cprint(f"Inferred dense dimension {self.dimension} from FastEmbed model {self.fastembed_model_name}", "cyan")
                 except Exception as e:
                     cprint(f"Warning: Could not automatically infer dimension from FastEmbed model: {e}", "yellow")
                     raise ValueError("Dimension must be provided if it cannot be inferred from the FastEmbed model.") from e

        if not self.dimension and (not self.use_fastembed or not self.dense_model):
             raise ValueError("Vector dimension must be provided if not using a FastEmbed dense model.")

        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self._create_db_table() # Create table immediately

        self.index = self._load_or_build_index() # Handles dense vectors

        # Note: Sparse and Late Interaction data are stored in SQLite, not a separate FAISS index here.

    def _validate_init_params(self):
        """Validate initialization parameters."""
        if self.use_fastembed and not _FASTEMBED_AVAILABLE:
            cprint("Warning: use_fastembed=True but fastembed is not available. Disabling FastEmbed features.", "red")
            self.use_fastembed = False
        if self.use_fastembed and not any([self.fastembed_model_name, self.fastembed_sparse_model_name, self.fastembed_late_interaction_model_name]):
            cprint("Warning: use_fastembed=True but no FastEmbed model names provided.", "yellow")
        if not self.use_fastembed and not self.dimension:
            raise ValueError("Dimension must be provided when not using FastEmbed.")
        if self.faiss_index_type != "flat":
            cprint(f"Warning: Currently only 'flat' FAISS index type is fully supported. Using {self.faiss_index_type}.", "yellow")

    def _initialize_fastembed_models(self):
        """Initializes the specified FastEmbed models."""
        if not self.use_fastembed:
            return
        try:
            common_args = {
                "cache_dir": self.fastembed_cache_dir,
                "providers": self.fastembed_providers
            }
            # Filter out None values
            common_args = {k: v for k, v in common_args.items() if v is not None}

            if self.fastembed_model_name:
                 # Try TextEmbedding first, then ImageEmbedding
                 try:
                     self.dense_model = TextEmbedding(model_name=self.fastembed_model_name, **common_args)
                     cprint(f"Initialized FastEmbed TextEmbedding model: {self.fastembed_model_name}", "green")
                 except ValueError: # Model not found or not a text model
                     try:
                         self.dense_model = ImageEmbedding(model_name=self.fastembed_model_name, **common_args)
                         cprint(f"Initialized FastEmbed ImageEmbedding model: {self.fastembed_model_name}", "green")
                     except Exception as e:
                         cprint(f"Failed to initialize FastEmbed dense/image model '{self.fastembed_model_name}': {e}", "red")
                         self.dense_model = None # Ensure it's None if init fails

            if self.fastembed_sparse_model_name:
                self.sparse_model = SparseTextEmbedding(model_name=self.fastembed_sparse_model_name, **common_args)
                cprint(f"Initialized FastEmbed SparseTextEmbedding model: {self.fastembed_sparse_model_name}", "green")

            if self.fastembed_late_interaction_model_name:
                self.late_interaction_model = LateInteractionTextEmbedding(model_name=self.fastembed_late_interaction_model_name, **common_args)
                cprint(f"Initialized FastEmbed LateInteractionTextEmbedding model: {self.fastembed_late_interaction_model_name}", "green")

            if self.fastembed_reranker_model_name:
                self.reranker_model = CrossEncoder(model_name=self.fastembed_reranker_model_name, **common_args)
                cprint(f"Initialized FastEmbed CrossEncoder reranker model: {self.fastembed_reranker_model_name}", "green")

        except Exception as e:
            cprint(f"Error initializing FastEmbed models: {e}", "red")
            # Decide if this should be fatal or just disable fastembed
            self.use_fastembed = False
            cprint("Disabling FastEmbed features due to initialization error.", "yellow")

    def _create_dummy_image_bytes(self) -> Optional[bytes]:
        """Creates dummy image bytes for dimension inference."""
        try:
            img = Image.new('RGB', (60, 30), color = 'red')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        except Exception as e:
            cprint(f"Could not create dummy image: {e}", "yellow")
            return None

    def _create_db_table(self):
        """Creates the SQLite table if it doesn't exist or if overwrite is True."""
        with closing(self.connection.cursor()) as cur:
            if self.overwrite:
                cprint(f"Dropping existing table {self.index_name} (if exists)...", "yellow")
                cur.execute(f"DROP TABLE IF EXISTS {self.index_name}")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.index_name}(
                    id INTEGER PRIMARY KEY,
                    content BLOB, -- Store text as UTF-8 bytes or raw image bytes
                    metadata TEXT, -- Store metadata as JSON string
                    sparse_vector BLOB, -- Store pickled sparse vector dict
                    token_vectors BLOB -- Store pickled token vectors numpy array
                )"""
            )
        cprint(f"Ensured database table '{self.index_name}' exists.", "green")

    def _load_or_build_index(self) -> faiss.IndexIDMap2:
        """Loads the dense FAISS index or builds a new one."""
        index_file = f"{self.index_name}_dense.faiss"
        if os.path.exists(index_file) and not self.overwrite:
            try:
                self.index = self.load_index(index_file)
                # Verify dimension
                if self.index.d != self.dimension:
                     cprint(f"Warning: Loaded index dimension ({self.index.d}) differs from expected ({self.dimension}). Rebuilding.", "yellow")
                     self.index = self.build_index(self.dimension)
                else:
                    cprint(f"Dense index {self.index_name} loaded successfully from {index_file}.", "green")
            except Exception as e:
                cprint(f"Failed to load dense index {index_file}: {e}. Building a new one.", "yellow")
                self.index = self.build_index(self.dimension)
        else:
            if self.overwrite and os.path.exists(index_file):
                 cprint(f"Overwriting existing dense index file {index_file}.", "yellow")
            self.index = self.build_index(self.dimension)
            cprint(f"New dense index {self.index_name} created.", "green")
        return self.index

    def build_index(self, dimension: int) -> faiss.IndexIDMap2:
        """Builds a FAISS index for dense vectors."""
        if self.faiss_index_type == "flat":
            base_index = faiss.IndexFlatL2(dimension)
            cprint(f"Building FAISS IndexFlatL2 with dimension {dimension}.", "cyan")
        # Add other index types here if needed
        # elif self.faiss_index_type == "hnsw":
        #     # Example: base_index = faiss.IndexHNSWFlat(dimension, 32)
        #     cprint(f"Building FAISS IndexHNSWFlat with dimension {dimension}.", "cyan")
        else:
            cprint(f"Unsupported FAISS index type '{self.faiss_index_type}'. Defaulting to 'flat'.", "yellow")
            base_index = faiss.IndexFlatL2(dimension)

        index = faiss.IndexIDMap2(base_index)
        return index

    def _embed_batch(self, batch_content: List[Union[str, bytes]]) -> Tuple[Optional[List[np.ndarray]], Optional[List[Dict]], Optional[List[np.ndarray]]]:
        """Embeds a batch of content using available FastEmbed models."""
        dense_embeddings = None
        sparse_embeddings = None
        late_interaction_embeddings = None

        # Determine content type (assume homogeneous batch for simplicity)
        is_image_batch = isinstance(batch_content[0], bytes)

        try:
            # Dense Embeddings
            if self.dense_model:
                if is_image_batch:
                    if isinstance(self.dense_model, ImageEmbedding):
                        dense_embeddings = list(self.dense_model.embed(batch_content, batch_size=len(batch_content)))
                    else:
                        cprint("Warning: Dense model is not an ImageEmbedding model, skipping dense embedding for image batch.", "yellow")
                else: # Text batch
                    if isinstance(self.dense_model, TextEmbedding):
                        dense_embeddings = list(self.dense_model.embed(batch_content, batch_size=len(batch_content)))
                    else:
                        cprint("Warning: Dense model is not a TextEmbedding model, skipping dense embedding for text batch.", "yellow")

            # Sparse Embeddings (Text only)
            if self.sparse_model and not is_image_batch:
                sparse_results: List[SparseEmbedding] = list(self.sparse_model.embed(batch_content, batch_size=len(batch_content)))
                sparse_embeddings = [{"indices": emb.indices.tolist(), "values": emb.values.tolist()} for emb in sparse_results]
            elif self.sparse_model and is_image_batch:
                cprint("Warning: Sparse models only support text, skipping sparse embedding for image batch.", "yellow")

            # Late Interaction Embeddings (Text only)
            if self.late_interaction_model and not is_image_batch:
                late_interaction_embeddings = list(self.late_interaction_model.embed(batch_content, batch_size=len(batch_content)))
            elif self.late_interaction_model and is_image_batch:
                cprint("Warning: Late interaction models only support text, skipping late interaction embedding for image batch.", "yellow")

        except Exception as e:
            cprint(f"Error during FastEmbed batch processing: {e}", "red")
            # Decide how to handle partial failures - here we return what we have
            pass

        return dense_embeddings, sparse_embeddings, late_interaction_embeddings


    def add(self, data: List[IndexData]) -> None:
        """
        Adds data to the database and FAISS index. Handles embedding if use_fastembed is True.

        Args:
            data (List[IndexData]): A list of IndexData objects.
                                     If use_fastembed is True, vectors can be None, and content
                                     (str or image bytes) will be embedded.
                                     If use_fastembed is False, vectors must be provided.
        """
        if not data:
            return

        processed_data: List[IndexData] = []
        faiss_vectors = []
        faiss_ids = []

        if self.use_fastembed:
            cprint(f"Processing {len(data)} items with FastEmbed (batch size: {self.fastembed_batch_size})...", "cyan")
            original_ids = {item.id for item in data}
            if len(original_ids) != len(data):
                 cprint("Warning: Duplicate IDs found in input data. Behavior for duplicates depends on DB constraints.", "yellow")

            for batch in grouper(data, self.fastembed_batch_size):
                batch_ids = [item.id for item in batch]
                batch_content = [item.content for item in batch]
                batch_metadata = [item.metadata for item in batch]

                dense_embeds, sparse_embeds, li_embeds = self._embed_batch(batch_content)

                for i, item_id in enumerate(batch_ids):
                    new_data_item = IndexData(
                        id=item_id,
                        content=batch_content[i],
                        metadata=batch_metadata[i]
                    )
                    if dense_embeds and i < len(dense_embeds):
                        new_data_item.vector = dense_embeds[i]
                        faiss_vectors.append(dense_embeds[i])
                        faiss_ids.append(item_id)
                    elif self.dense_model: # If dense model exists but embedding failed for this item
                         cprint(f"Warning: Failed to generate dense embedding for item ID {item_id}.", "yellow")

                    if sparse_embeds and i < len(sparse_embeds):
                        new_data_item.sparse_vector = sparse_embeds[i]
                    if li_embeds and i < len(li_embeds):
                        new_data_item.token_vectors = li_embeds[i]

                    processed_data.append(new_data_item)
            cprint("FastEmbed processing complete.", "cyan")

        else: # Not using FastEmbed, expect vectors in input
            cprint(f"Processing {len(data)} items (manual vectors)...", "cyan")
            for item in data:
                if item.vector is None:
                    cprint(f"Warning: Skipping item ID {item.id} because use_fastembed is False and no vector provided.", "yellow")
                    continue
                if item.vector.shape[-1] != self.dimension:
                     cprint(f"Warning: Skipping item ID {item.id} due to dimension mismatch (expected {self.dimension}, got {item.vector.shape[-1]}).", "yellow")
                     continue
                processed_data.append(item)
                faiss_vectors.append(item.vector)
                faiss_ids.append(item.id)

        # Add dense vectors to FAISS
        if faiss_vectors:
            try:
                ids_np = np.array(faiss_ids, dtype=np.int64)
                vectors_np = np.array(faiss_vectors, dtype=np.float32).reshape(len(faiss_ids), -1) # Ensure 2D
                if vectors_np.shape[1] != self.dimension:
                     raise ValueError(f"Vector dimension mismatch before adding to FAISS. Expected {self.dimension}, got {vectors_np.shape[1]}")
                self.index.add_with_ids(vectors_np, ids_np)
                cprint(f"Added {len(faiss_ids)} dense vectors to FAISS index {self.index_name}.", "green")
            except Exception as e:
                cprint(f"Error adding vectors to FAISS index: {e}", "red")
                # Consider how to handle partial failure - rollback?

        # Store all processed data (including sparse/LI vectors) to SQLite
        if processed_data:
            self.store_to_db(processed_data)

    def save_index(self) -> None:
        """Saves the dense FAISS index to a file."""
        index_file = f"{self.index_name}_dense.faiss"
        try:
            chunk = faiss.serialize_index(self.index)
            with open(index_file, "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"Dense index {self.index_name} saved successfully to {index_file}.", "green")
        except Exception as e:
            cprint(f"Error saving dense index to {index_file}: {e}", "red")

    def load_index(self, index_path: str) -> faiss.Index:
        """Loads the dense FAISS index from a file."""
        try:
            with open(index_path, "rb") as f:
                index = faiss.deserialize_index(pickle.load(f))
            cprint(f"Dense index loaded successfully from {index_path}.", "green")
            return index
        except FileNotFoundError:
            cprint(f"Error: Dense index file not found at {index_path}.", "red")
            raise
        except Exception as e:
            cprint(f"Error loading dense index from {index_path}: {e}", "red")
            raise

    def store_to_db(self, data: List[IndexData]) -> None:
        """Stores processed data (including embeddings) to the SQLite database."""
        try:
            values = []
            for point in data:
                # Serialize complex types
                content_blob = point.content if isinstance(point.content, bytes) else point.content.encode('utf-8')
                metadata_str = json.dumps(point.metadata) if point.metadata else None
                sparse_blob = pickle.dumps(point.sparse_vector) if point.sparse_vector else None
                token_vectors_blob = pickle.dumps(point.token_vectors) if point.token_vectors is not None else None

                values.append((
                    point.id,
                    content_blob,
                    metadata_str,
                    sparse_blob,
                    token_vectors_blob
                ))

            with closing(self.connection.cursor()) as cur:
                # Use INSERT OR REPLACE to handle updates based on primary key (id)
                cur.executemany(
                    f"""INSERT OR REPLACE INTO {self.index_name} (id, content, metadata, sparse_vector, token_vectors)
                        VALUES (?, ?, ?, ?, ?)""", values
                )
            cprint(f"Stored/updated {len(data)} records in database table {self.index_name}.", "green")

        except Exception as e:
            cprint(f"Could not complete database operation: {e}", "red")
            # Consider more specific error handling or logging

    def search(
        self,
        query: Union[str, bytes, np.ndarray],
        k: int = 5,
        search_type: str = "dense", # 'dense', 'sparse', 'late_interaction', 'hybrid'
        rerank: bool = False,
        hybrid_weights: Optional[Dict[str, float]] = None, # e.g., {"dense": 0.6, "sparse": 0.4}
        late_interaction_candidates: int = 20 # Number of candidates to fetch for LI scoring
        ) -> List[Dict[str, Any]]:
        """
        Performs search based on the specified type.

        Args:
            query (Union[str, bytes, np.ndarray]): The query (text, image bytes, or precomputed vector).
            k (int, optional): Number of results to return. Defaults to 5.
            search_type (str, optional): Type of search ('dense', 'sparse', 'late_interaction', 'hybrid'). Defaults to 'dense'.
            rerank (bool, optional): Whether to apply reranking using FastEmbed CrossEncoder. Defaults to False.
            hybrid_weights (Optional[Dict[str, float]], optional): Weights for hybrid search components. Required if search_type='hybrid'. Defaults to None.
            late_interaction_candidates (int, optional): Number of dense candidates to retrieve before LI scoring. Defaults to 20.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the retrieved document ('id', 'content', 'metadata', 'score').
                                  Score interpretation depends on search_type and reranking.
        """
        if search_type == "dense":
            results = self._search_dense(query, k)
        elif search_type == "sparse":
             if not self.use_fastembed or not self.sparse_model:
                 cprint("Error: Sparse search requires use_fastembed=True and a sparse model.", "red")
                 return []
             results = self._search_sparse(query, k)
        elif search_type == "late_interaction":
             if not self.use_fastembed or not self.late_interaction_model:
                 cprint("Error: Late interaction search requires use_fastembed=True and a late interaction model.", "red")
                 return []
             results = self._search_late_interaction(query, k, candidate_k=late_interaction_candidates)
        elif search_type == "hybrid":
             results = self._search_hybrid(query, k, hybrid_weights)
        else:
            cprint(f"Error: Unknown search type '{search_type}'. Use 'dense', 'sparse', 'late_interaction', or 'hybrid'.", "red")
            return []

        # --- Reranking Step ---
        if rerank:
            if not self.use_fastembed or not self.reranker_model:
                cprint("Warning: Reranking requested but use_fastembed is False or no reranker model initialized. Skipping reranking.", "yellow")
            elif not results:
                 cprint("Warning: No initial results to rerank.", "yellow")
            else:
                results = self._rerank_results(query, results) # Assumes query is text for reranker

        return results


    def _search_dense(self, query: Union[str, bytes, np.ndarray], k: int) -> List[Dict[str, Any]]:
        """Performs dense vector search using FAISS."""
        query_vector = self._get_query_vector(query, "dense")
        if query_vector is None:
            return []

        try:
            distances, ids = self.search_index(query_vector.reshape(1, -1), k) # FAISS expects 2D array
            retrieved_docs = self.retrieve(ids)

            # Map retrieved data back to scores and format output
            id_to_doc = {doc[0]: doc for doc in retrieved_docs}
            results = []
            for i, doc_id in enumerate(ids):
                if doc_id in id_to_doc and doc_id != -1: # FAISS uses -1 for no result
                    doc_data = id_to_doc[doc_id]
                    results.append({
                        "id": doc_id,
                        "content": doc_data[1].decode('utf-8', errors='replace') if isinstance(doc_data[1], bytes) else doc_data[1], # Decode potential blob
                        "metadata": json.loads(doc_data[2]) if doc_data[2] else {},
                        "score": float(distances[i]) # L2 distance from FAISS
                    })
            # Sort by distance (lower is better)
            results.sort(key=lambda x: x["score"])
            return results

        except Exception as e:
            cprint(f"Error during dense search: {e}", "red")
            return []

    def _search_sparse(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Performs sparse vector search (manual dot product)."""
        if not isinstance(query, str):
             cprint("Error: Sparse search currently only supports text queries.", "red")
             return []
        if not self.sparse_model:
             cprint("Error: Sparse model not initialized.", "red")
             return []

        try:
            query_sparse_emb: SparseEmbedding = next(self.sparse_model.embed([query]))
            query_sparse_vec = {idx: val for idx, val in zip(query_sparse_emb.indices, query_sparse_emb.values)}

            candidate_ids = self._get_all_ids_from_db() # Get all IDs to score
            if not candidate_ids:
                return []

            scores = []
            for doc_id in candidate_ids:
                doc_data = self.retrieve_single(doc_id) # Fetch one by one
                if doc_data and doc_data['sparse_vector']:
                    doc_sparse_vec = doc_data['sparse_vector'] # Already a dict
                    score = 0.0
                    # Compute dot product
                    for idx, q_val in query_sparse_vec.items():
                        if idx in doc_sparse_vec['indices']:
                            doc_val_idx = doc_sparse_vec['indices'].index(idx)
                            score += q_val * doc_sparse_vec['values'][doc_val_idx]
                    if score > 0: # Only consider positive scores
                        scores.append({"id": doc_id, "score": score})

            # Sort by score (higher is better for dot product)
            scores.sort(key=lambda x: x["score"], reverse=True)
            top_k_ids = [item["id"] for item in scores[:k]]
            top_k_scores = {item["id"]: item["score"] for item in scores[:k]}

            retrieved_docs = self.retrieve(top_k_ids)
            results = []
            for doc_data in retrieved_docs:
                 doc_id = doc_data[0]
                 results.append({
                     "id": doc_id,
                     "content": doc_data[1].decode('utf-8', errors='replace') if isinstance(doc_data[1], bytes) else doc_data[1],
                     "metadata": json.loads(doc_data[2]) if doc_data[2] else {},
                     "score": float(top_k_scores.get(doc_id, 0.0)) # Dot product score
                 })
            # Ensure results are sorted by score descending
            results.sort(key=lambda x: x["score"], reverse=True)
            return results

        except Exception as e:
            cprint(f"Error during sparse search: {e}", "red")
            return []

    def _search_late_interaction(self, query: str, k: int, candidate_k: int) -> List[Dict[str, Any]]:
        """Performs late interaction search (ColBERT MaxSim)."""
        if not isinstance(query, str):
             cprint("Error: Late interaction search currently only supports text queries.", "red")
             return []
        if not self.late_interaction_model:
             cprint("Error: Late interaction model not initialized.", "red")
             return []

        try:
            # 1. Get query token embeddings
            query_token_embeds: np.ndarray = next(self.late_interaction_model.query_embed([query])) # [num_query_tokens, dim]

            # 2. Get candidate documents (e.g., using dense search)
            cprint(f"Fetching {candidate_k} candidates for Late Interaction scoring...", "cyan")
            candidate_results = self._search_dense(query, candidate_k)
            candidate_ids = [res["id"] for res in candidate_results]

            if not candidate_ids:
                cprint("No candidates found for Late Interaction scoring.", "yellow")
                return []

            # 3. Retrieve token embeddings for candidates
            doc_token_embeddings = {}
            retrieved_full_docs = self.retrieve_multiple_full(candidate_ids)
            for doc_id, doc_data in retrieved_full_docs.items():
                if doc_data and doc_data['token_vectors'] is not None:
                    doc_token_embeddings[doc_id] = doc_data['token_vectors'] # [num_doc_tokens, dim]
                else:
                     cprint(f"Warning: Missing token vectors for candidate ID {doc_id}", "yellow")


            if not doc_token_embeddings:
                 cprint("No token vectors found for any candidates.", "red")
                 return []

            # 4. Compute MaxSim scores
            scores = []
            for doc_id, doc_embeds in doc_token_embeddings.items():
                # Compute similarity matrix: [num_query_tokens, num_doc_tokens]
                sim_matrix = np.dot(query_token_embeds, doc_embeds.T)
                # MaxSim: Max over document tokens, Sum over query tokens
                max_sim_per_query_token = np.max(sim_matrix, axis=1)
                total_score = np.sum(max_sim_per_query_token)
                scores.append({"id": doc_id, "score": total_score})

            # 5. Sort by score and get top-k
            scores.sort(key=lambda x: x["score"], reverse=True)
            top_k_ids = [item["id"] for item in scores[:k]]
            top_k_scores = {item["id"]: item["score"] for item in scores[:k]}

            # 6. Format results
            final_results = []
            # Use already retrieved full docs info
            for doc_id in top_k_ids:
                 if doc_id in retrieved_full_docs:
                     doc_data = retrieved_full_docs[doc_id]
                     final_results.append({
                         "id": doc_id,
                         "content": doc_data['content'], # Already decoded
                         "metadata": doc_data['metadata'],
                         "score": float(top_k_scores.get(doc_id, 0.0)) # MaxSim score
                     })

            # Ensure final results are sorted
            final_results.sort(key=lambda x: x["score"], reverse=True)
            return final_results

        except Exception as e:
            cprint(f"Error during late interaction search: {e}", "red")
            return []

    def _search_hybrid(self, query: Union[str, bytes, np.ndarray], k: int, weights: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Performs hybrid search using Reciprocal Rank Fusion (RRF)."""
        if not weights or not math.isclose(sum(weights.values()), 1.0):
             cprint("Error: Hybrid search requires weights for 'dense' and/or 'sparse' that sum to 1.0.", "red")
             # Defaulting to dense search if weights are invalid
             cprint("Defaulting to dense search.", "yellow")
             return self._search_dense(query, k)

        rank_lists = []
        all_results_map = {} # Store full result details by ID

        # Perform Dense Search
        if weights.get("dense", 0.0) > 0:
            dense_results = self._search_dense(query, k * 2) # Fetch more candidates
            if dense_results:
                # Score for dense is L2 distance, lower is better. Rank is 1-based.
                # RRF needs ranks where lower is better.
                rank_lists.append([(res["id"], rank + 1) for rank, res in enumerate(dense_results)])
                for res in dense_results: all_results_map[res["id"]] = res
            else:
                 cprint("Warning: Dense search returned no results for hybrid.", "yellow")


        # Perform Sparse Search (only if query is text and sparse model exists)
        if weights.get("sparse", 0.0) > 0 and isinstance(query, str) and self.sparse_model:
            sparse_results = self._search_sparse(query, k * 2) # Fetch more candidates
            if sparse_results:
                # Score for sparse is dot product, higher is better.
                # Convert to rank where lower is better.
                rank_lists.append([(res["id"], rank + 1) for rank, res in enumerate(sparse_results)])
                for res in sparse_results:
                    if res["id"] not in all_results_map: all_results_map[res["id"]] = res # Add if not present
            else:
                 cprint("Warning: Sparse search returned no results for hybrid.", "yellow")
        elif weights.get("sparse", 0.0) > 0:
             cprint("Warning: Sparse weight provided but query is not text or sparse model missing. Skipping sparse search.", "yellow")

        # Perform Late Interaction Search (only if query is text and LI model exists)
        # Note: LI scores are not directly comparable, RRF might not be ideal.
        # Consider adding LI results *after* RRF or using a different fusion.
        # For now, we'll stick to Dense/Sparse RRF.
        if weights.get("late_interaction", 0.0) > 0:
             cprint("Warning: Hybrid search with 'late_interaction' weight is not directly supported via RRF due to score incompatibility. Skipping LI component.", "yellow")


        if not rank_lists:
            cprint("No results from any search component for hybrid search.", "yellow")
            return []

        # Apply RRF
        # RRF expects ranks where lower is better, which we've provided.
        # The RRF score itself is higher for better ranks.
        rrf_scores = self._reciprocal_rank_fusion(rank_lists) # Returns list of (id, rrf_score) sorted descending

        # Get top K results and format
        top_k_ids = [item[0] for item in rrf_scores[:k]]
        final_results = []
        for doc_id in top_k_ids:
            if doc_id in all_results_map:
                 res = all_results_map[doc_id]
                 # Find the RRF score for this ID
                 rrf_score = next((score for id, score in rrf_scores if id == doc_id), 0.0)
                 res["score"] = float(rrf_score) # Overwrite original score with RRF score
                 final_results.append(res)

        # Ensure final results are sorted by RRF score descending
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results

    def _reciprocal_rank_fusion(self, rank_lists: List[List[Tuple[int, int]]], k_const: int = 60) -> List[Tuple[int, float]]:
        """
        Performs Reciprocal Rank Fusion on multiple ranked lists.

        Args:
            rank_lists (List[List[Tuple[int, int]]]): A list of lists, where each inner list contains (id, rank) tuples.
                                                      Rank should be 1-based, lower is better.
            k_const (int, optional): Constant used in the RRF formula. Defaults to 60.

        Returns:
            List[Tuple[int, float]]: A list of (id, rrf_score) tuples, sorted by score descending.
        """
        rrf_scores: Dict[int, float] = {}
        all_ids = set(item[0] for rank_list in rank_lists for item in rank_list)

        for doc_id in all_ids:
            score = 0.0
            for rank_list in rank_lists:
                # Find the rank of doc_id in this list
                rank = next((r for id, r in rank_list if id == doc_id), None)
                if rank is not None:
                    score += 1.0 / (k_const + rank)
            rrf_scores[doc_id] = score

        # Sort by RRF score descending
        sorted_scores = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores


    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reranks search results using FastEmbed CrossEncoder."""
        if not isinstance(query, str):
            cprint("Warning: Reranking currently only supports text queries. Skipping.", "yellow")
            return results
        if not self.reranker_model:
             cprint("Error: Reranker model not initialized.", "red")
             return results

        try:
            passages = [res["content"] for res in results]
            # The CrossEncoder expects pairs of [query, passage]
            query_passage_pairs = [[query, passage] for passage in passages]

            cprint(f"Reranking {len(results)} results...", "cyan")
            reranked_scores: List[float] = self.reranker_model.predict(query_passage_pairs, batch_size=self.fastembed_batch_size) # type: ignore

            # Add reranked scores and sort
            for i, res in enumerate(results):
                res["rerank_score"] = float(reranked_scores[i])

            # Sort by rerank_score descending (higher is better)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            cprint("Reranking complete.", "green")
            return results

        except Exception as e:
            cprint(f"Error during reranking: {e}", "red")
            return results # Return original results on error


    def _get_query_vector(self, query: Union[str, bytes, np.ndarray], vector_type: str = "dense") -> Optional[np.ndarray]:
        """Generates or returns the query vector."""
        if isinstance(query, np.ndarray):
            # Assume precomputed vector
            if vector_type == "dense" and query.ndim == 1 and query.shape[0] == self.dimension:
                 return query
            elif vector_type == "dense" and query.ndim == 2 and query.shape[0] == 1 and query.shape[1] == self.dimension:
                 return query.flatten() # Use the 1D vector
            # Add checks for other vector types if needed
            else:
                 cprint(f"Error: Precomputed query vector has incorrect shape/dimensions for type '{vector_type}'. Expected ({self.dimension},) for dense.", "red")
                 return None
        elif self.use_fastembed:
            try:
                model_to_use: Optional[Union[TextEmbedding, ImageEmbedding]] = None
                is_image = isinstance(query, bytes)

                if vector_type == "dense":
                    model_to_use = self.dense_model
                # Add logic for other types if needed (sparse/LI query vectors handled in their search methods)

                if model_to_use is None:
                     cprint(f"Error: No appropriate FastEmbed model initialized for vector type '{vector_type}'.", "red")
                     return None

                # Check model compatibility with query type
                if is_image and not isinstance(model_to_use, ImageEmbedding):
                     cprint(f"Error: Query is an image, but the configured dense model '{self.fastembed_model_name}' is not an ImageEmbedding model.", "red")
                     return None
                if not is_image and not isinstance(model_to_use, TextEmbedding):
                     cprint(f"Error: Query is text, but the configured dense model '{self.fastembed_model_name}' is not a TextEmbedding model.", "red")
                     return None

                # Embed the query
                embedding = next(model_to_use.embed([query]))
                # Ensure correct dimension
                if embedding.shape[-1] != self.dimension:
                     cprint(f"Error: Embedded query dimension ({embedding.shape[-1]}) mismatch with index dimension ({self.dimension}).", "red")
                     return None
                return embedding

            except Exception as e:
                cprint(f"Error generating query embedding with FastEmbed: {e}", "red")
                return None
        else:
            cprint("Error: Query must be a precomputed numpy array when use_fastembed is False.", "red")
            return None

    def search_index(self, query_vector: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """
        Searches the dense FAISS index.

        Args:
            query_vector (np.ndarray): The 2D query vector (batch size 1).
            k (int, optional): Number of neighbors. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: Distances and IDs.
        """
        if query_vector.ndim != 2 or query_vector.shape[0] != 1:
             raise ValueError("FAISS search expects a 2D query_vector (batch size 1).")
        if query_vector.shape[1] != self.dimension:
             raise ValueError(f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.dimension}).")

        try:
            D, I = self.index.search(query_vector.astype(np.float32), k)
            distances = [float(d) for d in D[0]]
            ids = [int(i) for i in I[0] if i != -1] # Filter out -1 ids
            # Adjust distances list if -1 was filtered
            distances = distances[:len(ids)]
            return distances, ids
        except Exception as e:
             cprint(f"Error during FAISS search: {e}", "red")
             return [], []

    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """Retrieves basic data (id, content, metadata) from SQLite by IDs."""
        if not ids:
            return []
        # Ensure IDs are standard Python integers
        safe_ids = [int(i) for i in ids]
        if not safe_ids:
            return []

        rows = []
        try:
            with closing(self.connection.cursor()) as cur:
                placeholders = ','.join('?' * len(safe_ids))
                sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"
                cur.execute(sql, safe_ids)
                rows = cur.fetchall()
        except sqlite3.Error as e:
            cprint(f"Error during retrieve: {e}", "red")
            raise
        except Exception as e:
             cprint(f"An unexpected error occurred during retrieve: {e}", "red")
             raise
        return rows

    def retrieve_single(self, doc_id: int) -> Optional[Dict[str, Any]]:
         """Retrieves full data for a single ID, including pickled vectors."""
         try:
             with closing(self.connection.cursor()) as cur:
                 sql = f"SELECT id, content, metadata, sparse_vector, token_vectors FROM {self.index_name} WHERE id = ?"
                 cur.execute(sql, (doc_id,))
                 row = cur.fetchone()
                 if row:
                     content_val = row[1]
                     try:
                         # Try decoding as text, fallback to keeping as bytes (for image)
                         content_decoded = content_val.decode('utf-8')
                     except UnicodeDecodeError:
                         content_decoded = content_val # Keep as bytes

                     sparse_vec = pickle.loads(row[3]) if row[3] else None
                     token_vecs = pickle.loads(row[4]) if row[4] else None
                     return {
                         "id": row[0],
                         "content": content_decoded,
                         "metadata": json.loads(row[2]) if row[2] else {},
                         "sparse_vector": sparse_vec,
                         "token_vectors": token_vecs
                     }
                 else:
                     return None
         except sqlite3.Error as e:
             cprint(f"Error during retrieve_single (ID: {doc_id}): {e}", "red")
             return None
         except Exception as e:
              cprint(f"An unexpected error occurred during retrieve_single (ID: {doc_id}): {e}", "red")
              return None

    def retrieve_multiple_full(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
         """Retrieves full data for multiple IDs."""
         if not ids:
             return {}
         safe_ids = [int(i) for i in ids]
         if not safe_ids:
             return {}

         results = {}
         try:
             with closing(self.connection.cursor()) as cur:
                 placeholders = ','.join('?' * len(safe_ids))
                 sql = f"SELECT id, content, metadata, sparse_vector, token_vectors FROM {self.index_name} WHERE id IN ({placeholders})"
                 cur.execute(sql, safe_ids)
                 rows = cur.fetchall()
                 for row in rows:
                     doc_id = row[0]
                     content_val = row[1]
                     try:
                         content_decoded = content_val.decode('utf-8')
                     except UnicodeDecodeError:
                         content_decoded = content_val # Keep as bytes

                     sparse_vec = pickle.loads(row[3]) if row[3] else None
                     token_vecs = pickle.loads(row[4]) if row[4] else None
                     results[doc_id] = {
                         "id": doc_id,
                         "content": content_decoded,
                         "metadata": json.loads(row[2]) if row[2] else {},
                         "sparse_vector": sparse_vec,
                         "token_vectors": token_vecs
                     }
         except sqlite3.Error as e:
             cprint(f"Error during retrieve_multiple_full: {e}", "red")
         except Exception as e:
              cprint(f"An unexpected error occurred during retrieve_multiple_full: {e}", "red")
         return results

    def _get_all_ids_from_db(self) -> List[int]:
        """Retrieves all unique IDs from the database table."""
        ids = []
        try:
            with closing(self.connection.cursor()) as cur:
                cur.execute(f"SELECT id FROM {self.index_name}")
                rows = cur.fetchall()
                ids = [row[0] for row in rows]
        except sqlite3.Error as e:
            cprint(f"Error getting all IDs from DB: {e}", "red")
        return ids


    def __del__(self):
         """Closes the database connection when the object is deleted."""
         if hasattr(self, "connection") and self.connection:
             try:
                 self.connection.close()
                 cprint("Database connection closed.", "yellow")
             except Exception as e:
                 cprint(f"Error closing database connection: {e}", "red")

    def usage(self):
        """Prints usage instructions for the FqlDb class."""
        cprint("\n--- FqlDb Usage ---", "blue", attrs=["bold"])

        cprint("\nInitialization:", "green", attrs=["underline"])
        cprint("  # Manual vector management (default)", "cyan")
        cprint("  fql_db_manual = FqlDb(index_name='my_index', dimension=128, db_name='my_db.db')", "white")
        cprint("\n  # Using FastEmbed for dense text embeddings", "cyan")
        cprint("  fql_db_fast_text = FqlDb(index_name='fast_text', use_fastembed=True,", "white")
        cprint("                           fastembed_model_name='BAAI/bge-small-en-v1.5')", "white")
        cprint("\n  # Using FastEmbed for dense image embeddings", "cyan")
        cprint("  fql_db_fast_img = FqlDb(index_name='fast_img', use_fastembed=True,", "white")
        cprint("                          fastembed_model_name='Qdrant/clip-ViT-B-32-vision')", "white")
        cprint("\n  # Using FastEmbed for sparse text embeddings", "cyan")
        cprint("  fql_db_fast_sparse = FqlDb(index_name='fast_sparse', use_fastembed=True,", "white")
        cprint("                             fastembed_sparse_model_name='prithvida/Splade_PP_en_v1',", "white")
        cprint("                             dimension=128) # Dimension still needed for potential dense part", "white")
        cprint("\n  # Using FastEmbed for late interaction (ColBERT)", "cyan")
        cprint("  fql_db_fast_li = FqlDb(index_name='fast_li', use_fastembed=True,", "white")
        cprint("                         fastembed_late_interaction_model_name='colbert-ir/colbertv2.0',", "white")
        cprint("                         dimension=128) # Dimension still needed for potential dense part", "white")
        cprint("\n  # Using FastEmbed with reranker", "cyan")
        cprint("  fql_db_rerank = FqlDb(index_name='rerank_idx', use_fastembed=True,", "white")
        cprint("                        fastembed_model_name='BAAI/bge-small-en-v1.5',", "white")
        cprint("                        fastembed_reranker_model_name='BAAI/bge-reranker-base')", "white")
        cprint("\n  # Using FastEmbed with GPU (ensure correct providers/drivers)", "cyan")
        cprint("  fql_db_gpu = FqlDb(index_name='gpu_idx', use_fastembed=True,", "white")
        cprint("                     fastembed_model_name='BAAI/bge-small-en-v1.5',", "white")
        cprint("                     fastembed_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])", "white")


        cprint("\nAdding Data:", "green", attrs=["underline"])
        cprint("  # Manual: Provide pre-computed vectors", "cyan")
        cprint("  manual_data = [IndexData(id=1, content='text1', vector=np.array([0.1, 0.2]))]", "white")
        cprint("  fql_db_manual.add(manual_data)", "white")
        cprint("\n  # FastEmbed (Text): Provide text content", "cyan")
        cprint("  fast_text_data = [IndexData(id=1, content='text to embed')]", "white")
        cprint("  fql_db_fast_text.add(fast_text_data)", "white")
        cprint("\n  # FastEmbed (Image): Provide image content as bytes", "cyan")
        cprint("  with open('image.jpg', 'rb') as f: img_bytes = f.read()", "white")
        cprint("  fast_img_data = [IndexData(id=1, content=img_bytes)]", "white")
        cprint("  fql_db_fast_img.add(fast_img_data)", "white")

        cprint("\nSearching:", "green", attrs=["underline"])
        cprint("  # Dense search (manual vector)", "cyan")
        cprint("  results_manual = fql_db_manual.search(query=np.array([0.15, 0.25]), k=3)", "white")
        cprint("\n  # Dense search (FastEmbed text query)", "cyan")
        cprint("  results_fast_text = fql_db_fast_text.search(query='search query text', k=3)", "white")
        cprint("\n  # Dense search (FastEmbed image query)", "cyan")
        cprint("  results_fast_img = fql_db_fast_img.search(query=img_bytes, k=3)", "white")
        cprint("\n  # Sparse search (FastEmbed text query)", "cyan")
        cprint("  results_sparse = fql_db_fast_sparse.search(query='sparse query', k=3, search_type='sparse')", "white")
        cprint("\n  # Late Interaction search (FastEmbed text query)", "cyan")
        cprint("  results_li = fql_db_fast_li.search(query='late interaction query', k=3, search_type='late_interaction')", "white")
        cprint("\n  # Hybrid search (Dense + Sparse)", "cyan")
        cprint("  results_hybrid = fql_db_fast_sparse.search(query='hybrid query', k=5, search_type='hybrid',", "white")
        cprint("                                            hybrid_weights={'dense': 0.5, 'sparse': 0.5}) # Requires dense model too", "white")
        cprint("\n  # Search with Reranking (FastEmbed text query)", "cyan")
        cprint("  results_reranked = fql_db_rerank.search(query='query to rerank', k=10, rerank=True)", "white")


        cprint("\nSaving/Loading Dense Index:", "green", attrs=["underline"])
        cprint("  fql_db_manual.save_index()", "white")
        cprint("  # Index is loaded automatically on init if file exists", "white")

        cprint("\nCleanup:", "green", attrs=["underline"])
        cprint("  # Manually delete .db and .faiss files", "cyan")
        cprint("  cleanup_test_files('my_index_dense.faiss', 'my_db.db')", "white")
        cprint("\n--- End Usage ---", "blue", attrs=["bold"])


# --- Unit Tests ---
def test_fql_db_manual():
    """Tests FqlDb with manual vector management."""
    cprint("\n--- Starting FqlDb Manual Tests ---", "blue", attrs=["bold"])
    index_name = "test_manual_index"
    db_name = "test_manual.db"
    dimension = 2
    cleanup_test_files(f"{index_name}_dense.faiss", db_name)
    fql_db = None
    loaded_fql_db = None

    try:
        test_data = [
            IndexData(vector=np.array([1.0, 2.0], dtype=np.float32), id=1, content="Manual content 1", metadata={"src": "manual"}),
            IndexData(vector=np.array([3.0, 4.0], dtype=np.float32), id=2, content="Manual content 2", metadata={"src": "manual"}),
            IndexData(vector=np.array([1.5, 2.5], dtype=np.float32), id=3, content="Manual content 3", metadata={"src": "manual"}),
        ]
        fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name, overwrite=True)
        cprint("[Test Case] Initialization (Manual)", "cyan")

        fql_db.add(test_data)
        cprint("[Test Case] Add Data (Manual)", "cyan")
        assert fql_db.index.ntotal == 3, f"Expected 3 items in FAISS index, found {fql_db.index.ntotal}"

        query_vector = np.array([[1.1, 2.1]], dtype=np.float32)
        results = fql_db.search(query=query_vector[0], k=2, search_type="dense") # Pass 1D vector to search
        cprint("[Test Case] Dense Search (Manual)", "cyan")
        assert len(results) == 2, f"Expected 2 search results, got {len(results)}"
        assert results[0]["id"] == 1, f"Expected ID 1 to be closest, got {results[0]['id']}"
        assert results[1]["id"] == 3, f"Expected ID 3 to be second closest, got {results[1]['id']}"
        cprint(f"  Search results: {results}", "white")

        retrieved = fql_db.retrieve([res["id"] for res in results])
        cprint("[Test Case] Retrieve Data (Manual)", "cyan")
        assert len(retrieved) == 2, f"Expected 2 retrieved docs, got {len(retrieved)}"
        cprint(f"  Retrieved data: {retrieved}", "white")

        fql_db.save_index()
        cprint("[Test Case] Save Index (Manual)", "cyan")
        if fql_db.connection: fql_db.connection.close(); fql_db.connection = None

        loaded_fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name)
        cprint("[Test Case] Load Index (Manual)", "cyan")
        assert loaded_fql_db.index.ntotal == 3, "Loaded index should have 3 items"

        results_loaded = loaded_fql_db.search(query=query_vector[0], k=2, search_type="dense")
        cprint("[Test Case] Dense Search (Loaded Manual)", "cyan")
        assert len(results_loaded) == 2
        assert results_loaded[0]["id"] == 1
        cprint(f"  Search results: {results_loaded}", "white")

        cprint("\n--- FqlDb Manual Tests Passed! ---", "green", attrs=["bold"])

    except Exception as e:
        cprint(f"\n--- FqlDb Manual Test Failed: {e} ---", "red", attrs=["bold"])
        raise
    finally:
        if fql_db and fql_db.connection: fql_db.connection.close(); fql_db.connection = None
        if loaded_fql_db and loaded_fql_db.connection: loaded_fql_db.connection.close(); loaded_fql_db.connection = None
        del fql_db
        del loaded_fql_db
        cleanup_test_files(f"{index_name}_dense.faiss", db_name)

def test_fql_db_fastembed():
    """Tests FqlDb with FastEmbed features."""
    if not _FASTEMBED_AVAILABLE:
        cprint("\n--- Skipping FqlDb FastEmbed Tests (fastembed not installed) ---", "yellow", attrs=["bold"])
        return

    cprint("\n--- Starting FqlDb FastEmbed Tests ---", "blue", attrs=["bold"])
    index_name = "test_fast_index"
    db_name = "test_fast.db"
    # Use known models for testing
    dense_model = "BAAI/bge-small-en-v1.5" # dim 384
    sparse_model = "prithvida/Splade_PP_en_v1"
    li_model = "colbert-ir/colbertv2.0" # dim 128 (per token)
    rerank_model = "BAAI/bge-reranker-base"
    image_model = "Qdrant/clip-ViT-B-32-vision" # dim 512

    # --- Test Dense Text ---
    cprint("\n[Test Section] Dense Text Embeddings", "magenta", attrs=["underline"])
    cleanup_test_files(f"{index_name}_dense_text_dense.faiss", db_name)
    fql_db_text = None
    try:
        fql_db_text = FqlDb(
            index_name=f"{index_name}_dense_text",
            db_name=db_name,
            use_fastembed=True,
            fastembed_model_name=dense_model,
            overwrite=True
        )
        cprint("[Test Case] Initialization (FastEmbed Dense Text)", "cyan")
        assert fql_db_text.dimension == 384

        text_data = [
            IndexData(id=10, content="This is the first document."),
            IndexData(id=11, content="This document is the second one."),
            IndexData(id=12, content="And this is the third one."),
            IndexData(id=13, content="Is this the first document?"),
        ]
        fql_db_text.add(text_data)
        cprint("[Test Case] Add Data (FastEmbed Dense Text)", "cyan")
        assert fql_db_text.index.ntotal == 4

        query = "first document"
        results = fql_db_text.search(query=query, k=2, search_type="dense")
        cprint("[Test Case] Dense Search (FastEmbed Text Query)", "cyan")
        assert len(results) == 2
        assert results[0]["id"] == 10 or results[0]["id"] == 13 # Depending on exact similarity
        assert results[1]["id"] == 10 or results[1]["id"] == 13
        cprint(f"  Search results for '{query}': {results}", "white")

    except Exception as e:
        cprint(f"\n--- FqlDb Dense Text Test Failed: {e} ---", "red", attrs=["bold"])
        raise
    finally:
        if fql_db_text and fql_db_text.connection: fql_db_text.connection.close(); fql_db_text.connection = None
        del fql_db_text
        cleanup_test_files(f"{index_name}_dense_text_dense.faiss", db_name) # Clean db too

    # --- Test Sparse Text ---
    cprint("\n[Test Section] Sparse Text Embeddings", "magenta", attrs=["underline"])
    # Note: Sparse doesn't use FAISS index in this setup
    cleanup_test_files(f"{index_name}_sparse_text_dense.faiss", db_name) # Clean DB
    fql_db_sparse = None
    try:
        # Need a dimension for the dummy dense index even if not used for sparse search
        fql_db_sparse = FqlDb(
            index_name=f"{index_name}_sparse_text",
            db_name=db_name,
            use_fastembed=True,
            fastembed_sparse_model_name=sparse_model,
            dimension=1, # Dummy dimension for dense index
            overwrite=True
        )
        cprint("[Test Case] Initialization (FastEmbed Sparse Text)", "cyan")

        sparse_text_data = [
            IndexData(id=20, content="sparse vector example one"),
            IndexData(id=21, content="another example for sparse vectors"),
            IndexData(id=22, content="sparse is different from dense"),
        ]
        fql_db_sparse.add(sparse_text_data)
        cprint("[Test Case] Add Data (FastEmbed Sparse Text)", "cyan")
        # Check DB content
        test_retrieve = fql_db_sparse.retrieve_single(20)
        assert test_retrieve is not None and test_retrieve.get('sparse_vector') is not None, "Sparse vector not found in DB"
        cprint(f"  Retrieved sparse vector for ID 20: {test_retrieve['sparse_vector']['indices'][:5]}...", "white")


        query = "example sparse"
        results = fql_db_sparse.search(query=query, k=2, search_type="sparse")
        cprint("[Test Case] Sparse Search (FastEmbed Text Query)", "cyan")
        assert len(results) == 2, f"Expected 2 sparse results, got {len(results)}"
        assert results[0]["id"] == 21 or results[0]["id"] == 20 # Based on dot product score
        assert results[1]["id"] == 21 or results[1]["id"] == 20
        cprint(f"  Search results for '{query}': {results}", "white")

    except Exception as e:
        cprint(f"\n--- FqlDb Sparse Text Test Failed: {e} ---", "red", attrs=["bold"])
        raise
    finally:
        if fql_db_sparse and fql_db_sparse.connection: fql_db_sparse.connection.close(); fql_db_sparse.connection = None
        del fql_db_sparse
        cleanup_test_files(f"{index_name}_sparse_text_dense.faiss", db_name)

    # --- Test Late Interaction ---
    cprint("\n[Test Section] Late Interaction (ColBERT)", "magenta", attrs=["underline"])
    cleanup_test_files(f"{index_name}_li_text_dense.faiss", db_name) # Clean DB
    fql_db_li = None
    try:
        # Need dense model for candidate retrieval
        fql_db_li = FqlDb(
            index_name=f"{index_name}_li_text",
            db_name=db_name,
            use_fastembed=True,
            fastembed_model_name=dense_model, # For candidate retrieval
            fastembed_late_interaction_model_name=li_model,
            overwrite=True
        )
        cprint("[Test Case] Initialization (FastEmbed Late Interaction)", "cyan")
        assert fql_db_li.dimension == 384 # From dense model

        li_text_data = [
            IndexData(id=30, content="ColBERT model provides late interaction."),
            IndexData(id=31, content="Interaction happens during scoring."),
            IndexData(id=32, content="Dense models pool embeddings earlier."),
        ]
        fql_db_li.add(li_text_data)
        cprint("[Test Case] Add Data (FastEmbed Late Interaction)", "cyan")
        test_retrieve = fql_db_li.retrieve_single(30)
        assert test_retrieve is not None and test_retrieve.get('token_vectors') is not None, "Token vectors not found in DB"
        cprint(f"  Retrieved token vectors shape for ID 30: {test_retrieve['token_vectors'].shape}", "white")

        query = "late interaction scoring"
        results = fql_db_li.search(query=query, k=2, search_type="late_interaction", late_interaction_candidates=3)
        cprint("[Test Case] Late Interaction Search (FastEmbed Text Query)", "cyan")
        assert len(results) == 2, f"Expected 2 LI results, got {len(results)}"
        # Expect 30 and 31 to score highest due to "late interaction" and "scoring"
        assert results[0]["id"] == 30 or results[0]["id"] == 31
        assert results[1]["id"] == 30 or results[1]["id"] == 31
        cprint(f"  Search results for '{query}': {results}", "white")

    except Exception as e:
        cprint(f"\n--- FqlDb Late Interaction Test Failed: {e} ---", "red", attrs=["bold"])
        raise
    finally:
        if fql_db_li and fql_db_li.connection: fql_db_li.connection.close(); fql_db_li.connection = None
        del fql_db_li
        cleanup_test_files(f"{index_name}_li_text_dense.faiss", db_name)

    # --- Test Reranking ---
    cprint("\n[Test Section] Reranking", "magenta", attrs=["underline"])
    cleanup_test_files(f"{index_name}_rerank_dense.faiss", db_name) # Clean DB
    fql_db_rerank = None
    try:
        fql_db_rerank = FqlDb(
            index_name=f"{index_name}_rerank",
            db_name=db_name,
            use_fastembed=True,
            fastembed_model_name=dense_model,
            fastembed_reranker_model_name=rerank_model,
            overwrite=True
        )
        cprint("[Test Case] Initialization (FastEmbed with Reranker)", "cyan")

        rerank_text_data = [
            IndexData(id=40, content="The quick brown fox jumps over the lazy dog."),
            IndexData(id=41, content="A fast, dark-colored canine leaps above a sleepy canine."), # Semantically similar but different words
            IndexData(id=42, content="Weather is nice today."), # Less relevant
        ]
        fql_db_rerank.add(rerank_text_data)
        cprint("[Test Case] Add Data (FastEmbed Reranker)", "cyan")

        query = "Fast fox jump over dog"
        # Search without reranking first
        results_no_rerank = fql_db_rerank.search(query=query, k=3, search_type="dense", rerank=False)
        cprint("[Test Case] Dense Search (Before Reranking)", "cyan")
        cprint(f"  Search results for '{query}' (no rerank): {results_no_rerank}", "white")

        # Search with reranking
        results_reranked = fql_db_rerank.search(query=query, k=3, search_type="dense", rerank=True)
        cprint("[Test Case] Dense Search (With Reranking)", "cyan")
        assert len(results_reranked) == 3
        # Expect reranker to potentially boost ID 41 higher than dense search might
        assert results_reranked[0]["id"] == 40 or results_reranked[0]["id"] == 41
        assert "rerank_score" in results_reranked[0]
        cprint(f"  Search results for '{query}' (reranked): {results_reranked}", "white")
        # Check if order changed (possible, not guaranteed)
        if len(results_no_rerank) == len(results_reranked):
             original_order = [r['id'] for r in results_no_rerank]
             reranked_order = [r['id'] for r in results_reranked]
             if original_order != reranked_order:
                 cprint("  Reranking changed the order of results.", "cyan")
             else:
                 cprint("  Reranking did not change the order (or initial order was already optimal).", "cyan")


    except Exception as e:
        cprint(f"\n--- FqlDb Reranking Test Failed: {e} ---", "red", attrs=["bold"])
        raise
    finally:
        if fql_db_rerank and fql_db_rerank.connection: fql_db_rerank.connection.close(); fql_db_rerank.connection = None
        del fql_db_rerank
        cleanup_test_files(f"{index_name}_rerank_dense.faiss", db_name)

    # --- Test Image Embedding ---
    cprint("\n[Test Section] Image Embeddings", "magenta", attrs=["underline"])
    cleanup_test_files(f"{index_name}_image_dense.faiss", db_name) # Clean DB
    fql_db_img = None
    dummy_img_path = "dummy_test_image.png"
    try:
        # Create dummy image file
        dummy_bytes = FqlDb(index_name='_', dimension=1)._create_dummy_image_bytes() # Use helper
        if not dummy_bytes: raise RuntimeError("Failed to create dummy image for test")
        with open(dummy_img_path, "wb") as f:
            f.write(dummy_bytes)
        cprint(f"Created dummy image file: {dummy_img_path}", "cyan")

        fql_db_img = FqlDb(
            index_name=f"{index_name}_image",
            db_name=db_name,
            use_fastembed=True,
            fastembed_model_name=image_model, # CLIP vision model
            overwrite=True
        )
        cprint("[Test Case] Initialization (FastEmbed Image)", "cyan")
        assert fql_db_img.dimension == 512 # CLIP ViT-B/32 dimension

        img_data = [IndexData(id=50, content=dummy_bytes, metadata={"filename": dummy_img_path})]
        fql_db_img.add(img_data)
        cprint("[Test Case] Add Data (FastEmbed Image)", "cyan")
        assert fql_db_img.index.ntotal == 1

        # Search with the same image bytes
        results = fql_db_img.search(query=dummy_bytes, k=1, search_type="dense")
        cprint("[Test Case] Dense Search (FastEmbed Image Query)", "cyan")
        assert len(results) == 1, f"Expected 1 image result, got {len(results)}"
        assert results[0]["id"] == 50, f"Expected ID 50, got {results[0]['id']}"
        # L2 distance for identical vector should be close to 0
        assert results[0]["score"] < 1e-6, f"Expected near-zero distance for identical image, got {results[0]['score']}"
        cprint(f"  Search results for dummy image: {results}", "white")

    except Exception as e:
        cprint(f"\n--- FqlDb Image Embedding Test Failed: {e} ---", "red", attrs=["bold"])
        raise
    finally:
        if fql_db_img and fql_db_img.connection: fql_db_img.connection.close(); fql_db_img.connection = None
        del fql_db_img
        cleanup_test_files(f"{index_name}_image_dense.faiss", db_name, dummy_img_path)


    cprint("\n--- All FqlDb FastEmbed Tests Passed! ---", "green", attrs=["bold"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FqlDb or print usage.")
    parser.add_argument("--usage", action="store_true", help="Print FqlDb usage instructions.")
    parser.add_argument("--test", action="store_true", help="Run FqlDb tests.")
    args = parser.parse_args()

    if args.usage:
        # Need to instantiate with valid dimension even for usage
        temp_fql_db = None
        try:
            temp_fql_db = FqlDb(index_name='temp_usage', dimension=1, db_name='temp_usage.db', overwrite=True)
            temp_fql_db.usage()
        except Exception as e:
             cprint(f"Error generating usage: {e}", "red")
        finally:
            if temp_fql_db and temp_fql_db.connection: temp_fql_db.connection.close(); temp_fql_db.connection = None
            del temp_fql_db
            cleanup_test_files('temp_usage_dense.faiss', 'temp_usage.db')
    elif args.test:
        test_fql_db_manual()
        test_fql_db_fastembed()
    else:
        cprint("Please specify --usage or --test flag.", "yellow")
        parser.print_help()
