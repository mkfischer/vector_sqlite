import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Any, Literal, Iterable
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse
from pydantic import BaseModel as PydanticBaseModel, Field
from sentence_transformers import SentenceTransformer # Default embedding engine
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding, ImageEmbedding, SparseEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from PIL import Image # For image type checking

# --- Configuration ---
DEFAULT_DB_NAME = "fql.db"
DEFAULT_EMBEDDING_MODE = "sentence-transformers"
DEFAULT_ST_MODEL = "all-MiniLM-L6-v2"
DEFAULT_FE_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_FE_SPARSE_MODEL = "prithvida/Splade_PP_en_v1" # Example sparse model
DEFAULT_FE_RERANK_MODEL = "BAAI/bge-reranker-base" # Example reranker
DEFAULT_FE_LATE_MODEL = "colbert-ir/colbertv2.0" # Example late interaction model
DEFAULT_FE_IMAGE_MODEL = "Qdrant/clip-ViT-B-32-vision" # Example image model

# --- Data Models ---
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class IndexData(BaseModel):
    """Data structure for items to be indexed."""
    id: int
    content: Optional[str] = None # Text content
    image_path: Optional[str] = None # Path to image file
    metadata: Dict = Field(default_factory=dict)
    # Embeddings (populated internally or provided)
    vector: Optional[np.ndarray] = None # Dense vector
    sparse_indices: Optional[List[int]] = None
    sparse_values: Optional[List[float]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.content is None and self.image_path is None:
            raise ValueError("Either 'content' or 'image_path' must be provided.")
        if self.content is not None and self.image_path is not None:
            raise ValueError("Provide either 'content' or 'image_path', not both.")

    class Config:
        arbitrary_types_allowed = True # Allow np.ndarray


# --- Main Class ---
class FqlDb:
    """
    A class combining vector similarity search (FAISS) with a SQLite database,
    supporting both Sentence-Transformers (default) and FastEmbed backends.
    """

    def __init__(self,
                 index_name: str,
                 dimension: Optional[int] = None, # Made optional, determined by model
                 db_name: str = DEFAULT_DB_NAME,
                 embedding_mode: Literal["sentence-transformers", "fastembed"] = DEFAULT_EMBEDDING_MODE,
                 dense_model_name: Optional[str] = None,
                 sparse_model_name: Optional[str] = None,
                 reranker_model_name: Optional[str] = None,
                 late_interaction_model_name: Optional[str] = None,
                 image_model_name: Optional[str] = None,
                 rebuild_index: bool = False):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name for the index and database table.
            dimension (Optional[int]): The dimension of the dense vectors. If None, it's inferred from the model.
                                       Required if loading an existing index without a model.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            embedding_mode (Literal["sentence-transformers", "fastembed"], optional):
                The embedding backend to use. Defaults to "sentence-transformers".
            dense_model_name (Optional[str], optional): Name of the dense embedding model.
                Defaults depend on the embedding_mode.
            sparse_model_name (Optional[str], optional): Name of the sparse embedding model (FastEmbed only).
                Defaults to DEFAULT_FE_SPARSE_MODEL if mode is 'fastembed'.
            reranker_model_name (Optional[str], optional): Name of the reranker model (FastEmbed only).
                Defaults to DEFAULT_FE_RERANK_MODEL if mode is 'fastembed'.
            late_interaction_model_name (Optional[str], optional): Name of the late interaction model (FastEmbed only).
                Defaults to DEFAULT_FE_LATE_MODEL if mode is 'fastembed'.
            image_model_name (Optional[str], optional): Name of the image embedding model (FastEmbed only).
                Defaults to DEFAULT_FE_IMAGE_MODEL if mode is 'fastembed'.
            rebuild_index (bool, optional): If True, forces rebuilding the index even if files exist. Defaults to False.
        """
        self.index_name = index_name
        self.db_name = db_name
        self.embedding_mode = embedding_mode
        self.dense_model = None
        self.sparse_model = None
        self.reranker_model = None
        self.late_interaction_model = None
        self.image_model = None
        self.dimension = dimension # Store provided dimension initially

        cprint(f"Initializing FqlDb '{index_name}' with mode: '{embedding_mode}'", "cyan")

        # --- Initialize Models based on mode ---
        try:
            if self.embedding_mode == "sentence-transformers":
                model_name = dense_model_name or DEFAULT_ST_MODEL
                cprint(f"Loading SentenceTransformer model: {model_name}", "yellow")
                self.dense_model = SentenceTransformer(model_name)
                if self.dimension is None:
                    # Infer dimension from ST model
                    self.dimension = self.dense_model.get_sentence_embedding_dimension()
                    cprint(f"Inferred dimension from SentenceTransformer model: {self.dimension}", "yellow")
                elif self.dimension != self.dense_model.get_sentence_embedding_dimension():
                     cprint(f"Warning: Provided dimension {self.dimension} differs from SentenceTransformer model dimension {self.dense_model.get_sentence_embedding_dimension()}. Using provided dimension.", "red")

            elif self.embedding_mode == "fastembed":
                # Dense Model
                dense_model_name = dense_model_name or DEFAULT_FE_DENSE_MODEL
                cprint(f"Loading FastEmbed Dense model: {dense_model_name}", "yellow")
                self.dense_model = TextEmbedding(model_name=dense_model_name)
                # Infer dimension if not provided
                if self.dimension is None:
                   # Get dimension from one of the loaded models (dense is primary for FAISS)
                   # Need to access the underlying model info, fastembed doesn't have a simple get_dim method easily accessible
                   # Let's embed a dummy text to find out
                   try:
                       dummy_emb = list(self.dense_model.embed("test"))[0]
                       self.dimension = len(dummy_emb)
                       cprint(f"Inferred dimension from FastEmbed dense model: {self.dimension}", "yellow")
                   except Exception as e:
                       cprint(f"Could not infer dimension from FastEmbed model {dense_model_name}: {e}", "red")
                       raise ValueError("Dimension could not be inferred. Please provide it explicitly.") from e

                # Sparse Model (Optional)
                _sparse_model_name = sparse_model_name or DEFAULT_FE_SPARSE_MODEL
                if _sparse_model_name:
                    cprint(f"Loading FastEmbed Sparse model: {_sparse_model_name}", "yellow")
                    self.sparse_model = SparseTextEmbedding(model_name=_sparse_model_name)

                # Reranker Model (Optional)
                _reranker_model_name = reranker_model_name or DEFAULT_FE_RERANK_MODEL
                if _reranker_model_name:
                    cprint(f"Loading FastEmbed Reranker model: {_reranker_model_name}", "yellow")
                    self.reranker_model = TextCrossEncoder(model_name=_reranker_model_name)

                # Late Interaction Model (Optional)
                _late_model_name = late_interaction_model_name or DEFAULT_FE_LATE_MODEL
                if _late_model_name:
                     cprint(f"Loading FastEmbed Late Interaction model: {_late_model_name}", "yellow")
                     self.late_interaction_model = LateInteractionTextEmbedding(model_name=_late_model_name)

                # Image Model (Optional)
                _image_model_name = image_model_name or DEFAULT_FE_IMAGE_MODEL
                if _image_model_name:
                    cprint(f"Loading FastEmbed Image model: {_image_model_name}", "yellow")
                    self.image_model = ImageEmbedding(model_name=_image_model_name)
                    # Verify image model dimension matches if dimension was set by text model
                    try:
                        # Need a way to get image model dimension without embedding if possible
                        # For now, assume it matches or raise error later if adding images
                        pass # TODO: Add check if possible without dummy embedding
                    except Exception as e:
                        cprint(f"Could not verify dimension for FastEmbed image model {_image_model_name}: {e}", "red")


            else:
                raise ValueError(f"Unsupported embedding_mode: {self.embedding_mode}")

        except Exception as e:
            cprint(f"Error loading models for mode '{self.embedding_mode}': {e}", "red")
            raise

        if self.dimension is None:
             raise ValueError("Vector dimension must be provided or inferrable from the model.")

        # --- Initialize DB Connection and FAISS Index ---
        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self.index = self._load_or_build_index(rebuild_index)
        self._create_table_if_not_exists() # Ensure table exists

    def _load_or_build_index(self, rebuild: bool = False) -> faiss.IndexIDMap2:
        """
        Loads the index from file if it exists, otherwise builds a new index.

        Returns:
            faiss.IndexIDMap2: The loaded or newly built index.
        Loads the index from file if it exists, otherwise builds a new index.

        Args:
            rebuild (bool): If True, forces rebuilding the index.

        Returns:
            faiss.IndexIDMap2: The loaded or newly built index.
        """
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file) and not rebuild:
            try:
                cprint(f"Attempting to load index '{self.index_name}' from {index_file}", "yellow")
                index = self.load_index() # Use instance method
                # Verify dimension if possible (FAISS index doesn't store it directly in IndexIDMap2 easily)
                if index.ntotal > 0:
                     # Reconstruct one vector to check dimension
                     try:
                         reconstructed_vector = index.reconstruct(0)
                         loaded_dimension = reconstructed_vector.shape[0]
                         if loaded_dimension != self.dimension:
                             cprint(f"Warning: Loaded index dimension ({loaded_dimension}) differs from expected dimension ({self.dimension}). Rebuilding index.", "red")
                             return self._build_new_index()
                         cprint(f"Index '{self.index_name}' loaded successfully with {index.ntotal} vectors.", "green")
                         return index
                     except Exception as recon_e:
                          cprint(f"Could not verify dimension of loaded index '{self.index_name}': {recon_e}. Rebuilding index.", "red")
                          return self._build_new_index()
                else:
                    cprint(f"Index '{self.index_name}' loaded successfully (empty).", "green")
                    return index # Return empty loaded index
            except Exception as e:
                cprint(f"Failed to load index '{self.index_name}': {e}. Building a new one.", "red")
                return self._build_new_index()
        else:
            if rebuild and os.path.exists(index_file):
                cprint(f"Rebuilding index '{self.index_name}' as requested.", "yellow")
            elif not os.path.exists(index_file):
                 cprint(f"Index file {index_file} not found. Building a new one.", "yellow")
            return self._build_new_index()

    def _build_new_index(self) -> faiss.IndexIDMap2:
        """Builds and returns a new FAISS index."""
        cprint(f"Building new FAISS index '{self.index_name}' with dimension {self.dimension}.", "yellow")
        if self.dimension is None or self.dimension <= 0:
             raise ValueError("Cannot build index without a valid positive dimension.")
        flat_index = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIDMap2(flat_index)
        cprint(f"Index '{self.index_name}' created successfully.", "green")
        return index

    def _create_table_if_not_exists(self):
        """Creates the SQLite table if it doesn't exist."""
        try:
            with closing(self.connection.cursor()) as cur:
                # Store sparse vectors as JSON strings for simplicity
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(
                        id INTEGER PRIMARY KEY,
                        content TEXT,
                        image_path TEXT,
                        metadata TEXT,
                        sparse_indices TEXT,
                        sparse_values TEXT
                    )"""
                )
            # cprint(f"Ensured database table '{self.index_name}' exists.", "grey")
        except Exception as e:
            cprint(f"Error creating/checking database table '{self.index_name}': {e}", "red")
            raise

    # --- Embedding Generation ---

    def _get_dense_embeddings(self, texts: List[str]) -> Iterable[np.ndarray]:
        """Generates dense embeddings for a list of texts."""
        if not self.dense_model:
            raise RuntimeError("Dense embedding model not initialized.")
        if self.embedding_mode == "sentence-transformers":
            return self.dense_model.encode(texts, convert_to_numpy=True)
        elif self.embedding_mode == "fastembed":
            # FastEmbed's embed returns a generator
            return self.dense_model.embed(texts)
        else:
            raise RuntimeError(f"Unsupported embedding mode for dense embeddings: {self.embedding_mode}")

    def _get_sparse_embeddings(self, texts: List[str]) -> Optional[Iterable[SparseEmbedding]]:
        """Generates sparse embeddings for a list of texts (FastEmbed only)."""
        if self.embedding_mode != "fastembed":
            # cprint("Sparse embeddings only supported in 'fastembed' mode.", "yellow")
            return None
        if not self.sparse_model:
            # cprint("Sparse embedding model not loaded.", "yellow")
            return None
        return self.sparse_model.embed(texts)

    def _get_image_embeddings(self, image_paths: List[str]) -> Iterable[np.ndarray]:
        """Generates dense embeddings for a list of image paths (FastEmbed only)."""
        if self.embedding_mode != "fastembed":
            raise RuntimeError("Image embeddings only supported in 'fastembed' mode.")
        if not self.image_model:
            raise RuntimeError("Image embedding model not initialized.")
        # Validate image paths exist? Maybe too slow. FastEmbed might handle errors.
        return self.image_model.embed(image_paths)

    # --- Indexing and Storage ---

    def add(self, data: List[IndexData]) -> None:
        """
        Generates embeddings (if needed), adds data to the FAISS index, and stores metadata in SQLite.

        Args:
            data (List[IndexData]): A list of IndexData objects.
                                     Expects `content` OR `image_path` to be set.
                                     `vector`, `sparse_indices`, `sparse_values` can be pre-computed
                                     or will be generated if None.
        """
        if not data:
            return

        ids_to_add = []
        dense_vectors_to_add = []
        data_to_store = [] # Keep track of data for DB storage

        # Separate text and image data for potentially different embedding models/logic
        text_data_indices = [i for i, d in enumerate(data) if d.content is not None]
        image_data_indices = [i for i, d in enumerate(data) if d.image_path is not None]

        # --- Process Text Data ---
        if text_data_indices:
            texts = [data[i].content for i in text_data_indices] # type: ignore[misc]
            text_ids = [data[i].id for i in text_data_indices]

            # Generate Dense Embeddings
            dense_embeddings_iter = self._get_dense_embeddings(texts)
            dense_embeddings = list(dense_embeddings_iter) # Consume generator if FE

            # Generate Sparse Embeddings (Optional)
            sparse_embeddings_iter = self._get_sparse_embeddings(texts)
            sparse_embeddings = list(sparse_embeddings_iter) if sparse_embeddings_iter else [None] * len(texts)

            for i, original_index in enumerate(text_data_indices):
                item = data[original_index]
                item.vector = dense_embeddings[i] # Store generated dense vector
                if sparse_embeddings[i]:
                    item.sparse_indices = sparse_embeddings[i].indices.tolist() # type: ignore[union-attr]
                    item.sparse_values = sparse_embeddings[i].values.tolist() # type: ignore[union-attr]

                ids_to_add.append(item.id)
                dense_vectors_to_add.append(item.vector)
                data_to_store.append(item)

        # --- Process Image Data ---
        if image_data_indices:
            if self.embedding_mode != "fastembed":
                 raise RuntimeError("Image processing only supported in 'fastembed' mode.")
            if not self.image_model:
                 raise RuntimeError("Image embedding model not initialized for adding images.")

            image_paths = [data[i].image_path for i in image_data_indices] # type: ignore[misc]
            image_ids = [data[i].id for i in image_data_indices]

            # Generate Dense Image Embeddings
            image_embeddings_iter = self._get_image_embeddings(image_paths)
            image_embeddings = list(image_embeddings_iter) # Consume generator

            for i, original_index in enumerate(image_data_indices):
                item = data[original_index]
                item.vector = image_embeddings[i] # Store generated dense vector
                # Sparse embeddings are typically not generated for images in the same way
                item.sparse_indices = None
                item.sparse_values = None

                ids_to_add.append(item.id)
                dense_vectors_to_add.append(item.vector)
                data_to_store.append(item)

        # --- Add to FAISS Index ---
        if ids_to_add:
            final_ids = np.array(ids_to_add, dtype=np.int64)
            final_vectors = np.array(dense_vectors_to_add, dtype=np.float32)
            if final_vectors.shape[1] != self.dimension:
                 raise ValueError(f"Vector dimension mismatch: Expected {self.dimension}, got {final_vectors.shape[1]}")
            self.index.add_with_ids(final_vectors, final_ids)
            cprint(f"Added {len(final_ids)} vectors to FAISS index '{self.index_name}'.", "green")

            # --- Store in SQLite DB ---
            self.store_to_db(data_to_store)
        else:
            cprint("No valid data provided to add.", "yellow")


    def build_index(self, dimension: int) -> faiss.IndexIDMap2: # Keep for potential direct use, but prefer _build_new_index
        """
        Builds a FAISS index.

        Args:
            dimension (int): The dimension of the vectors.

        Returns:
            faiss.IndexIDMap2: The FAISS index.
        """
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        cprint(f"Building FAISS index '{self.index_name}' with dimension {dimension}.", "yellow")
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        return index

    # Keep add_to_index for backward compatibility or direct vector addition?
    # Let's deprecate it or make it internal, favoring the `add` method.
    # For now, just leave it but note it doesn't use the embedding models.
    def add_vectors_direct(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Directly adds pre-computed vectors and IDs to the FAISS index.
        Note: This bypasses the internal embedding models and DB storage logic. Use `add` for typical workflow.

        Args:
            vectors (np.ndarray): Numpy array of vectors (dtype=np.float32).
            ids (np.ndarray): Numpy array of corresponding IDs (dtype=np.int64).
        """
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("Number of vectors and IDs must match.")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: index requires {self.dimension}, got {vectors.shape[1]}.")
        self.index.add_with_ids(vectors, ids)
        cprint(f"Directly added {len(ids)} vectors to FAISS index '{self.index_name}'.", "green")


    def save_index(self) -> None:
        """
        Saves the FAISS index to a file.
        Saves the current FAISS index to a file named '{index_name}.pkl'.
        """
        index_file = f"{self.index_name}.pkl"
        try:
            cprint(f"Saving index '{self.index_name}' to {index_file}...", "yellow")
            chunk = faiss.serialize_index(self.index)
            with open(index_file, "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"Index '{self.index_name}' saved successfully.", "green")
        except Exception as e:
            cprint(f"Error saving index '{self.index_name}': {e}", "red")
            raise

    def load_index(self) -> faiss.IndexIDMap2:
        """
        Loads the FAISS index from file using the instance's index_name.

        Returns:
            faiss.Index: The loaded FAISS index.
        Loads the FAISS index from the file '{index_name}.pkl'.

        Returns:
            faiss.IndexIDMap2: The loaded FAISS index.
        """
        index_file = f"{self.index_name}.pkl"
        if not os.path.exists(index_file):
             raise FileNotFoundError(f"Index file not found: {index_file}")
        try:
            with open(index_file, "rb") as f:
                index_data = pickle.load(f)
                index = faiss.deserialize_index(index_data)
                # Ensure it's the correct type after deserialization
                if not isinstance(index, faiss.IndexIDMap2):
                     # This might happen if the saved index wasn't IndexIDMap2
                     # Or potentially if faiss versions differ significantly.
                     # Attempt conversion if it's just the flat index? Risky.
                     cprint(f"Warning: Loaded index is not of type IndexIDMap2. Type is {type(index)}. Attempting to use as is.", "red")
                     # Best practice would be to ensure saving/loading consistency or rebuild.
                return index # type: ignore # Assume it's compatible or user handles consequences
        except Exception as e:
            cprint(f"Error loading index from {index_file}: {e}", "red")
            raise


    def store_to_db(self, data: List[IndexData]) -> None:
        """
        Stores data to the SQLite database.

        Stores or updates data in the SQLite database table named '{index_name}'.

        Args:
            data (List[IndexData]): A list of IndexData objects to store.
                                     Assumes embeddings have been generated and attached if needed.
        """
        if not data:
            return
        try:
            values = []
            for point in data:
                # Serialize complex types like lists/dicts for storage
                sparse_indices_str = json.dumps(point.sparse_indices) if point.sparse_indices is not None else None
                sparse_values_str = json.dumps(point.sparse_values) if point.sparse_values is not None else None
                metadata_str = json.dumps(point.metadata) if point.metadata else None
                values.append((
                    point.id,
                    point.content,
                    point.image_path,
                    metadata_str,
                    sparse_indices_str,
                    sparse_values_str
                ))

            with closing(self.connection.cursor()) as cur:
                # Use INSERT OR REPLACE to handle updates based on primary key (id)
                cur.executemany(
                    f"""INSERT OR REPLACE INTO {self.index_name}
                       (id, content, image_path, metadata, sparse_indices, sparse_values)
                       VALUES (?,?,?,?,?,?)""", values
                )
            cprint(f"Stored/updated {len(data)} records in database table '{self.index_name}'.", "green")

        except Exception as e:
            cprint(f"Could not complete database operation for table '{self.index_name}': {e}", "red")
            raise

    def search(self,
               query: Union[str, np.ndarray, Image.Image],
               k: int = 3,
               search_type: Literal["text", "image"] = "text"
               ) -> Tuple[List[float], List[int]]:
        """
        Generates a query embedding and searches the FAISS index.

        Args:
            query (Union[str, np.ndarray, Image.Image]): The query (text, precomputed vector, or PIL Image).
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.
            search_type (Literal["text", "image"], optional): Type of search. Defaults to "text".

        Returns:
            Tuple[List[float], List[int]]: Distances and IDs of the nearest neighbors.
                                           Returns ([], []) if no results or error.
        """
        query_vector: Optional[np.ndarray] = None

        try:
            if isinstance(query, np.ndarray):
                query_vector = query.astype(np.float32) # Ensure correct type
                if query_vector.ndim == 1:
                     query_vector = np.expand_dims(query_vector, axis=0) # FAISS expects 2D array
            elif isinstance(query, str) and search_type == "text":
                if not self.dense_model: raise RuntimeError("Dense text model not initialized.")
                # Embed returns generator, need to consume it
                query_vector = list(self._get_dense_embeddings([query]))[0]
                query_vector = np.expand_dims(query_vector, axis=0)
            elif isinstance(query, Image.Image) and search_type == "image":
                 if self.embedding_mode != "fastembed": raise RuntimeError("Image search only in fastembed mode.")
                 if not self.image_model: raise RuntimeError("Image model not initialized.")
                 # FastEmbed ImageEmbedding expects paths, not PIL images directly.
                 # Need a temporary file or check if FE supports PIL images/bytes.
                 # For now, let's require image path for search or precomputed vector.
                 # Alternative: Modify to accept image path string directly.
                 raise NotImplementedError("Image search currently requires a precomputed vector or image path (not PIL Image).")
            elif isinstance(query, str) and search_type == "image": # Allow searching by image path
                 if self.embedding_mode != "fastembed": raise RuntimeError("Image search only in fastembed mode.")
                 if not self.image_model: raise RuntimeError("Image model not initialized.")
                 query_vector = list(self._get_image_embeddings([query]))[0]
                 query_vector = np.expand_dims(query_vector, axis=0)
            else:
                raise TypeError(f"Unsupported query type '{type(query)}' for search_type '{search_type}'.")

            if query_vector is None:
                 cprint("Could not generate or identify query vector.", "red")
                 return [], []

            if query_vector.shape[1] != self.dimension:
                 raise ValueError(f"Query vector dimension ({query_vector.shape[1]}) does not match index dimension ({self.dimension}).")

            # Perform FAISS search
            distances_np, ids_np = self.index.search(query_vector, k)

            # Process results
            if ids_np.size == 0 or ids_np[0][0] == -1: # Check for no results or invalid IDs
                return [], []

            distances = [float(d) for d in distances_np[0]]
            ids = [int(i) for i in ids_np[0] if i != -1] # Filter out -1 IDs if k > ntotal

            return distances[:len(ids)], ids # Ensure lists have same length

        except Exception as e:
            cprint(f"Error during search: {e}", "red")
            return [], [] # Return empty lists on error


    def retrieve(self, ids: List[int]) -> List[IndexData]:
        """
        Retrieves full IndexData objects from SQLite based on a list of IDs.

        Args:
            ids (List[int]): A list of IDs to retrieve.

        Returns:
            List[IndexData]: A list of IndexData objects. Returns empty list if no IDs match or error.
        """
        if not ids:
            return []

        results = []
        cur = None
        try:
            cur = self.connection.cursor()
            placeholders = ','.join('?' * len(ids))
            sql = f"SELECT id, content, image_path, metadata, sparse_indices, sparse_values FROM {self.index_name} WHERE id IN ({placeholders})"
            cur.execute(sql, ids)
            rows = cur.fetchall()

            # Map rows back to IndexData objects
            id_to_row = {row[0]: row for row in rows}

            # Return in the order of the input IDs, if found
            for item_id in ids:
                row = id_to_row.get(item_id)
                if row:
                    try:
                        metadata = json.loads(row[3]) if row[3] else {}
                        sparse_indices = json.loads(row[4]) if row[4] else None
                        sparse_values = json.loads(row[5]) if row[5] else None
                        results.append(IndexData(
                            id=row[0],
                            content=row[1],
                            image_path=row[2],
                            metadata=metadata,
                            sparse_indices=sparse_indices,
                            sparse_values=sparse_values
                            # Note: Dense vector is not stored in DB, only in FAISS
                        ))
                    except (json.JSONDecodeError, TypeError) as e:
                         cprint(f"Error parsing data for ID {row[0]}: {e}. Skipping.", "red")
                    except Exception as e_inner:
                         cprint(f"Unexpected error processing row for ID {row[0]}: {e_inner}. Skipping.", "red")


        except sqlite3.Error as e:
            cprint(f"Error during retrieve from table '{self.index_name}': {e}", "red")
            # Optionally re-raise or handle
        except Exception as e:
             cprint(f"An unexpected error occurred during retrieve: {e}", "red")
        finally:
            if cur:
                cur.close()
        return results

    # --- FastEmbed Specific Methods ---

    def rerank(self, query: str, results: List[IndexData], k: int = 5) -> Optional[List[Dict]]:
        """
        Reranks retrieved documents using a FastEmbed Cross-Encoder model.

        Args:
            query (str): The original query text.
            results (List[IndexData]): A list of IndexData objects (retrieved from DB).
            k (int, optional): The number of top results to return after reranking. Defaults to 5.

        Returns:
            Optional[List[Dict]]: A list of reranked documents with scores, or None if reranking is
                                  not supported or fails. Each dict contains 'id', 'content',
                                  'image_path', 'metadata', and 'rerank_score'.
        """
        if self.embedding_mode != "fastembed":
            cprint("Reranking is only supported in 'fastembed' mode.", "yellow")
            return None
        if not self.reranker_model:
            cprint("Reranker model not loaded. Cannot rerank.", "yellow")
            return None
        if not results:
            return []

        # Prepare passages for reranker (only works for text content)
        passages = [item.content for item in results if item.content is not None]
        original_indices = [i for i, item in enumerate(results) if item.content is not None] # Keep track of original items

        if not passages:
             cprint("No text content found in results to rerank.", "yellow")
             return []

        try:
            # Format for reranker: list of [query, passage] pairs
            query_passage_pairs = [[query, passage] for passage in passages]
            rerank_scores = self.reranker_model.predict(query_passage_pairs)

            # Combine results with scores and sort
            reranked_results_with_scores = []
            for i, score in enumerate(rerank_scores):
                original_item = results[original_indices[i]]
                reranked_results_with_scores.append({
                    "id": original_item.id,
                    "content": original_item.content,
                    "image_path": original_item.image_path,
                    "metadata": original_item.metadata,
                    "rerank_score": float(score) # Ensure standard float
                })

            # Sort by score descending
            reranked_results_with_scores.sort(key=lambda x: x["rerank_score"], reverse=True)

            return reranked_results_with_scores[:k]

        except Exception as e:
            cprint(f"Error during reranking: {e}", "red")
            return None


    def get_late_interaction_embeddings(self, texts: List[str], embedding_type: Literal["query", "document"] = "document") -> Optional[List[np.ndarray]]:
        """
        Generates late interaction embeddings (e.g., ColBERT) for texts using FastEmbed.

        Args:
            texts (List[str]): List of texts to embed.
            embedding_type (Literal["query", "document"]): Type of embedding to generate. Defaults to "document".

        Returns:
            Optional[List[np.ndarray]]: List of embeddings (each typically [num_tokens, dim]), or None if not supported/fails.
        """
        if self.embedding_mode != "fastembed":
            cprint("Late interaction embeddings only supported in 'fastembed' mode.", "yellow")
            return None
        if not self.late_interaction_model:
            cprint("Late interaction model not loaded.", "yellow")
            return None

        try:
            if embedding_type == "query":
                embeddings_iter = self.late_interaction_model.query_embed(texts)
            elif embedding_type == "document":
                embeddings_iter = self.late_interaction_model.embed(texts)
            else:
                raise ValueError("embedding_type must be 'query' or 'document'")

            return list(embeddings_iter) # Consume generator

        except Exception as e:
            cprint(f"Error generating late interaction embeddings: {e}", "red")
            return None


    # --- Housekeeping ---

    def __del__(self):
        Closes the database connection when the object is deleted.
        """
        if hasattr(self, "connection") and self.connection:
            try:
                self.connection.close()
                # cprint(f"Database connection '{self.db_name}' closed.", "grey")
            except Exception as e:
                # Might already be closed or invalid
                cprint(f"Error closing DB connection '{self.db_name}': {e}", "red")

    def usage(self):
        """Prints usage instructions for the FqlDb class."""
        cprint("--- FqlDb Usage ---", "cyan", attrs=["bold"])

        cprint("\nInitialization:", "green")
        cprint("  # Default mode (SentenceTransformers)", "white")
        cprint("  fql_db_st = FqlDb(index_name='st_index', dimension=384)", "white")
        cprint("  # FastEmbed mode (will infer dimension from default FE model)", "white")
        cprint("  fql_db_fe = FqlDb(index_name='fe_index', embedding_mode='fastembed')", "white")
        cprint("  # FastEmbed mode with specific models", "white")
        cprint("  fql_db_fe_custom = FqlDb(index_name='fe_custom', embedding_mode='fastembed',", "white")
        cprint("                         dense_model_name='BAAI/bge-large-en-v1.5',", "white")
        cprint("                         sparse_model_name='prithvida/Splade_PP_en_v1',", "white")
        cprint("                         reranker_model_name='BAAI/bge-reranker-base',", "white")
        cprint("                         image_model_name='Qdrant/clip-ViT-B-32-vision')", "white")

        cprint("\nAdding Data:", "green")
        cprint("  # Prepare data (text or image)", "white")
        cprint("  text_data = [IndexData(id=1, content='This is text'), IndexData(id=2, content='Another text')]", "white")
        cprint("  image_data = [IndexData(id=3, image_path='path/to/image1.jpg'), IndexData(id=4, image_path='path/to/image2.png')]", "white")
        cprint("  # Add data (embeddings generated automatically)", "white")
        cprint("  fql_db_fe.add(text_data)", "white")
        cprint("  fql_db_fe.add(image_data) # Only works in fastembed mode with image model", "white")

        cprint("\nSearching:", "green")
        cprint("  # Search text", "white")
        cprint("  distances, ids = fql_db_fe.search(query='search query text', k=5)", "white")
        cprint("  # Search with precomputed vector", "white")
        cprint("  query_vec = np.random.rand(1, fql_db_fe.dimension).astype(np.float32)", "white")
        cprint("  distances, ids = fql_db_fe.search(query=query_vec, k=3)", "white")
        cprint("  # Search using an image path (FastEmbed mode only)", "white")
        cprint("  distances, ids = fql_db_fe.search(query='path/to/query_image.jpg', k=3, search_type='image')", "white")


        cprint("\nRetrieving Data:", "green")
        cprint("  retrieved_items: List[IndexData] = fql_db_fe.retrieve(ids=ids)", "white")
        cprint("  for item in retrieved_items:", "white")
        cprint("      print(f\"ID: {item.id}, Content: {item.content}, Image: {item.image_path}\")", "white")
        cprint("      # Access sparse data if generated: item.sparse_indices, item.sparse_values", "white")


        cprint("\nReranking (FastEmbed mode only):", "green")
        cprint("  if fql_db_fe.reranker_model:", "white")
        cprint("      reranked_results = fql_db_fe.rerank(query='search query text', results=retrieved_items, k=5)", "white")
        cprint("      if reranked_results:", "white")
        cprint("          print(\"Reranked:\", reranked_results)", "white")

        cprint("\nLate Interaction Embeddings (FastEmbed mode only):", "green")
        cprint("  if fql_db_fe.late_interaction_model:", "white")
        cprint("      li_embeddings = fql_db_fe.get_late_interaction_embeddings(['text1', 'text2'])", "white")
        cprint("      if li_embeddings:", "white")
        cprint("          print(f\"Generated {len(li_embeddings)} late interaction embeddings.\")", "white")


        cprint("\nSaving/Loading Index:", "green")
        cprint("  fql_db_fe.save_index()", "white")
        cprint("  # Later...", "white")
        cprint("  loaded_db = FqlDb(index_name='fe_index', embedding_mode='fastembed') # Reloads automatically", "white")
        cprint("---------------------", "cyan", attrs=["bold"])


# --- Utility and Test Functions ---

def cleanup_test_files(index_name: str, db_name: str):
    """Removes test index and database files."""
    index_file = f"{index_name}.pkl"
    db_file = db_name
    files_removed = False
    cprint(f"\n--- Cleaning up test files for index '{index_name}' ---", "magenta")
    try:
        if os.path.exists(index_file):
            os.remove(index_file)
            cprint(f"Removed index file: {index_file}", "yellow")
            files_removed = True
        else:
            cprint(f"Index file not found (already clean?): {index_file}", "grey")
        if os.path.exists(db_file):
            os.remove(db_file)
            cprint(f"Removed database file: {db_file}", "yellow")
            files_removed = True
        else:
            cprint(f"Database file not found (already clean?): {db_file}", "grey")

        if not files_removed:
            cprint("No test files found to remove.", "grey")
        cprint("--- Cleanup finished ---", "magenta")

    except Exception as e:
        cprint(f"Error during test file cleanup for '{index_name}': {e}", "red")

def _create_dummy_image(path: str, size=(64, 64), color="red"):
    """Creates a small dummy image file."""
    try:
        img = Image.new('RGB', size, color=color)
        img.save(path)
        return True
    except Exception as e:
        cprint(f"Failed to create dummy image {path}: {e}", "red")
        return False

def test_fql_db_mode(mode: Literal["sentence-transformers", "fastembed"]):
    """Tests FqlDb functionality for a specific embedding mode."""
    cprint(f"\n===== Starting FqlDb Test: Mode = {mode} =====", "blue", attrs=["bold"])

    index_name = f"test_{mode}"
    db_name = f"test_{mode}.db"
    # Dimension will be inferred for fastembed, specify for sentence-transformers
    dimension = 384 if mode == "sentence-transformers" else None
    # Use smaller, faster models for testing if possible
    dense_model_st = "all-MiniLM-L6-v2" # 384 dim
    dense_model_fe = "BAAI/bge-small-en-v1.5" # 384 dim
    sparse_model_fe = "prithvida/Splade_PP_en_v1" # Example sparse
    rerank_model_fe = "BAAI/bge-reranker-base" # Example reranker
    image_model_fe = "Qdrant/clip-ViT-B-32-vision" # 512 dim - NOTE: Dimension mismatch potential

    # --- Ensure clean state ---
    cleanup_test_files(index_name, db_name)
    # Create dummy image files dir
    dummy_image_dir = "test_images"
    os.makedirs(dummy_image_dir, exist_ok=True)
    dummy_img1_path = os.path.join(dummy_image_dir, "img1.png")
    dummy_img2_path = os.path.join(dummy_image_dir, "img2.png")
    query_img_path = os.path.join(dummy_image_dir, "query_img.png")
    img1_created = _create_dummy_image(dummy_img1_path, color="blue")
    img2_created = _create_dummy_image(dummy_img2_path, color="green")
    query_img_created = _create_dummy_image(query_img_path, color="red")
    # --------------------------

    fql_db: Optional[FqlDb] = None
    loaded_fql_db: Optional[FqlDb] = None

    try:
        # --- Test Data ---
        test_text_data = [
            IndexData(id=1, content="This is the first test document.", metadata={"topic": "testing"}),
            IndexData(id=2, content="A second document for testing purposes.", metadata={"topic": "testing"}),
            IndexData(id=3, content="FastEmbed provides fast embeddings.", metadata={"topic": "fastembed"}),
        ]
        test_image_data = []
        if mode == "fastembed" and img1_created and img2_created:
             # Image model dim (512) differs from text (384). This will cause issues with a single FAISS index.
             # For testing, let's use a separate index for images or skip image tests if dims differ.
             # Let's skip adding images if dimensions mismatch for simplicity in this test.
             # A real application would need separate indexes or a multi-vector capable DB.
             img_dim = 512 # Dimension of clip-ViT-B-32-vision
             text_dim = 384 # Dimension of bge-small-en-v1.5
             if img_dim == text_dim:
                 test_image_data = [
                     IndexData(id=101, image_path=dummy_img1_path, metadata={"color": "blue"}),
                     IndexData(id=102, image_path=dummy_img2_path, metadata={"color": "green"}),
                 ]
             else:
                  cprint(f"Skipping image tests in mode '{mode}' due to dimension mismatch (Text: {text_dim}, Image: {img_dim})", "red")


        # --- Initialization ---
        cprint(f"\n[Test - {mode}] Initializing FqlDb...", "cyan")
        init_args = {
            "index_name": index_name,
            "db_name": db_name,
            "embedding_mode": mode,
            "dimension": dimension # Will be inferred if None and mode='fastembed'
        }
        if mode == "sentence-transformers":
            init_args["dense_model_name"] = dense_model_st
        else: # fastembed
            init_args["dense_model_name"] = dense_model_fe
            init_args["sparse_model_name"] = sparse_model_fe
            init_args["reranker_model_name"] = rerank_model_fe
            if not test_image_data: # Only load image model if we plan to test it
                 init_args["image_model_name"] = None
                 cprint("Image model loading skipped due to dimension mismatch or creation failure.", "yellow")
            else:
                 init_args["image_model_name"] = image_model_fe


        fql_db = FqlDb(**init_args) # type: ignore
        # Check inferred dimension if it was None
        if mode == 'fastembed' and dimension is None:
             assert fql_db.dimension is not None and fql_db.dimension > 0, "Dimension should have been inferred for fastembed"
             cprint(f"Dimension successfully inferred: {fql_db.dimension}", "green")
             # Update local dimension for consistency if needed for later checks
             dimension = fql_db.dimension # Use the inferred dimension going forward

        # --- Add & Store ---
        cprint(f"\n[Test - {mode}] Adding Text Data...", "cyan")
        fql_db.add(test_text_data)
        assert fql_db.index.ntotal == len(test_text_data), f"Expected {len(test_text_data)} vectors in index, found {fql_db.index.ntotal}"

        if test_image_data:
             cprint(f"\n[Test - {mode}] Adding Image Data...", "cyan")
             fql_db.add(test_image_data)
             expected_total = len(test_text_data) + len(test_image_data)
             assert fql_db.index.ntotal == expected_total, f"Expected {expected_total} vectors in index after images, found {fql_db.index.ntotal}"


        # --- Search & Retrieve (Text) ---
        cprint(f"\n[Test - {mode}] Searching Text...", "cyan")
        query_text = "information about testing documents"
        distances, ids = fql_db.search(query=query_text, k=2)
        cprint(f"  Query: '{query_text}'", "white")
        cprint(f"  Result IDs: {ids}", "white")
        cprint(f"  Result Distances: {distances}", "white")
        assert len(ids) <= 2, "Search should return at most k results"
        assert len(ids) == len(distances), "Distances and IDs length mismatch"
        if ids: # Proceed only if search returned results
             assert all(isinstance(i, int) for i in ids), "IDs should be integers"
             assert all(isinstance(d, float) for d in distances), "Distances should be floats"
             # Check if top results are plausible (ids 1 and 2 are about testing)
             assert set(ids).issubset({1, 2, 3}), f"Expected IDs from { {1,2,3} }, got {ids}"

             retrieved_data = fql_db.retrieve(ids)
             cprint(f"  Retrieved Data: {[item.id for item in retrieved_data]} - {[item.content[:20]+'...' if item.content else None for item in retrieved_data]}", "white")
             assert len(retrieved_data) == len(ids), "Retrieve count mismatch"
             retrieved_ids = sorted([item.id for item in retrieved_data])
             assert retrieved_ids == sorted(ids), "Retrieved IDs don't match search IDs"
             # Check sparse data storage if applicable
             if mode == "fastembed" and fql_db.sparse_model:
                 for item in retrieved_data:
                      if item.id in [t.id for t in test_text_data]: # Only check for text items
                           assert item.sparse_indices is not None, f"Sparse indices missing for ID {item.id}"
                           assert item.sparse_values is not None, f"Sparse values missing for ID {item.id}"
                           assert isinstance(item.sparse_indices, list), f"Sparse indices not list for ID {item.id}"
                           assert isinstance(item.sparse_values, list), f"Sparse values not list for ID {item.id}"
                 cprint("  Sparse data verified in retrieved items.", "green")

        else:
             cprint("  Search returned no results.", "yellow")


        # --- Search & Retrieve (Image, if applicable) ---
        if mode == "fastembed" and test_image_data and query_img_created:
            cprint(f"\n[Test - {mode}] Searching Image Path...", "cyan")
            query_image = query_img_path # Search using path
            img_distances, img_ids = fql_db.search(query=query_image, k=2, search_type="image")
            cprint(f"  Query Image Path: '{query_image}'", "white")
            cprint(f"  Result IDs: {img_ids}", "white")
            cprint(f"  Result Distances: {img_distances}", "white")
            assert len(img_ids) <= 2
            assert len(img_ids) == len(img_distances)
            if img_ids:
                assert all(isinstance(i, int) for i in img_ids)
                # Check if image IDs are retrieved (101, 102)
                assert set(img_ids).issubset({101, 102}), f"Expected image IDs from {{101, 102}}, got {img_ids}"
                img_retrieved = fql_db.retrieve(img_ids)
                cprint(f"  Retrieved Image Data: {[item.id for item in img_retrieved]} - {[item.image_path for item in img_retrieved]}", "white")
                assert len(img_retrieved) == len(img_ids)
            else:
                cprint("  Image search returned no results.", "yellow")


        # --- Rerank (FastEmbed mode only) ---
        if mode == "fastembed" and fql_db.reranker_model and ids:
            cprint(f"\n[Test - {mode}] Reranking...", "cyan")
            # Retrieve full data for reranking
            items_to_rerank = fql_db.retrieve(ids) # Use IDs from the text search
            reranked_results = fql_db.rerank(query=query_text, results=items_to_rerank, k=5)
            if reranked_results is not None:
                cprint(f"  Reranked Results (Top {len(reranked_results)}):", "white")
                for res in reranked_results:
                    cprint(f"    ID: {res['id']}, Score: {res['rerank_score']:.4f}, Content: {res['content'][:30]+'...'}", "white")
                assert len(reranked_results) <= 5
                assert len(reranked_results) <= len(items_to_rerank)
                # Check if scores are descending
                assert all(reranked_results[i]['rerank_score'] >= reranked_results[i+1]['rerank_score'] for i in range(len(reranked_results)-1)), "Reranked results are not sorted correctly."
            else:
                cprint("  Reranking failed or returned None.", "red")


        # --- Late Interaction (FastEmbed mode only) ---
        if mode == "fastembed" and fql_db.late_interaction_model:
             cprint(f"\n[Test - {mode}] Generating Late Interaction Embeddings...", "cyan")
             li_texts = ["example query", "example document"]
             li_query_embs = fql_db.get_late_interaction_embeddings(li_texts, embedding_type="query")
             li_doc_embs = fql_db.get_late_interaction_embeddings(li_texts, embedding_type="document")
             if li_query_embs and li_doc_embs:
                  assert len(li_query_embs) == len(li_texts)
                  assert len(li_doc_embs) == len(li_texts)
                  assert isinstance(li_query_embs[0], np.ndarray)
                  assert isinstance(li_doc_embs[0], np.ndarray)
                  # Check shape (e.g., [tokens, dim]) - dimensions vary
                  assert li_query_embs[0].ndim == 2 and li_query_embs[0].shape[0] > 0 and li_query_embs[0].shape[1] > 0
                  assert li_doc_embs[0].ndim == 2 and li_doc_embs[0].shape[0] > 0 and li_doc_embs[0].shape[1] > 0
                  cprint(f"  Successfully generated {len(li_query_embs)} query and {len(li_doc_embs)} document LI embeddings.", "green")
             else:
                  cprint("  Late interaction embedding generation failed or returned None.", "red")


        # --- Save & Load ---
        cprint(f"\n[Test - {mode}] Saving Index...", "cyan")
        fql_db.save_index()
        # Close connection before loading new instance
        if fql_db and fql_db.connection:
            fql_db.connection.close()
            fql_db.connection = None # Prevent __del__ trying again

        cprint(f"\n[Test - {mode}] Loading Index...", "cyan")
        # Reload - need to provide same config used for creation/inference
        loaded_fql_db = FqlDb(**init_args) # type: ignore
        expected_total = len(test_text_data) + len(test_image_data) # Recalculate expected
        assert loaded_fql_db.index.ntotal == expected_total, f"Loaded index has {loaded_fql_db.index.ntotal} vectors, expected {expected_total}"
        cprint("  Index loaded successfully after save.", "green")

        # --- Search & Retrieve (Loaded Index) ---
        cprint(f"\n[Test - {mode}] Searching Loaded Index (Text)...", "cyan")
        distances_loaded, ids_loaded = loaded_fql_db.search(query=query_text, k=2)
        cprint(f"  Query: '{query_text}'", "white")
        cprint(f"  Result IDs: {ids_loaded}", "white")
        assert sorted(ids_loaded) == sorted(ids), f"Search results differ after loading. Original: {ids}, Loaded: {ids_loaded}"

        retrieved_loaded = loaded_fql_db.retrieve(ids_loaded)
        cprint(f"  Retrieved Data: {[item.id for item in retrieved_loaded]}", "white")
        assert len(retrieved_loaded) == len(ids_loaded), "Retrieve count mismatch after load"
        assert sorted([item.id for item in retrieved_loaded]) == sorted(ids_loaded), "Retrieved IDs mismatch after load"

        cprint(f"\n===== FqlDb Test PASSED: Mode = {mode} =====", "green", attrs=["bold"])

    except Exception as e:
        cprint(f"\n===== FqlDb Test FAILED: Mode = {mode} =====", "red", attrs=["bold"])
        cprint(f"Error: {e}", "red")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise # Re-raise to fail the overall script if run in sequence

    finally:
        # --- Cleanup ---
        # Ensure connections are closed
        if fql_db and fql_db.connection:
            try: fql_db.connection.close()
            except: pass
        if loaded_fql_db and loaded_fql_db.connection:
            try: loaded_fql_db.connection.close()
            except: pass
        # Explicitly delete objects
        del fql_db
        del loaded_fql_db
        # Remove files
        cleanup_test_files(index_name, db_name)
        # Remove dummy images and dir
        try:
            if os.path.exists(dummy_img1_path): os.remove(dummy_img1_path)
            if os.path.exists(dummy_img2_path): os.remove(dummy_img2_path)
            if os.path.exists(query_img_path): os.remove(query_img_path)
            if os.path.exists(dummy_image_dir): os.rmdir(dummy_image_dir) # Only removes if empty
            cprint("Cleaned up dummy image files.", "yellow")
        except Exception as e:
            cprint(f"Error cleaning up dummy images: {e}", "red")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FqlDb tests or print usage instructions.")
    parser.add_argument("--usage", action="store_true", help="Print FqlDb usage instructions and exit.")
    parser.add_argument("--mode", choices=["sentence-transformers", "fastembed", "all"], default="all",
                        help="Specify which embedding mode to test ('all', 'sentence-transformers', 'fastembed'). Default is 'all'.")

    args = parser.parse_args()

    if args.usage:
        # Instantiate temporarily to call usage()
        try:
            # Need a valid dimension, even if temporary
            temp_db = FqlDb(index_name='usage_temp', dimension=10) # Use a dummy dimension
            temp_db.usage()
            del temp_db # Explicitly delete
            cleanup_test_files('usage_temp', DEFAULT_DB_NAME) # Clean up potentially created files
        except Exception as e:
             cprint(f"Error generating usage instructions: {e}", "red")
             cprint("Please ensure dependencies are installed.", "yellow")
    else:
        # Run tests based on the mode argument
        if args.mode == "all" or args.mode == "sentence-transformers":
            test_fql_db_mode("sentence-transformers")
        if args.mode == "all" or args.mode == "fastembed":
            # Add a check for fastembed dependencies? Pydantic should handle model loading errors.
            test_fql_db_mode("fastembed")
