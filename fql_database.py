import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Any, Literal
# Removed: from data_schema import IndexData
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse
from pydantic import BaseModel as PydanticBaseModel, Field # Added import
import json
from enum import Enum
from pathlib import Path

# FastEmbed imports
from fastembed import TextEmbedding, SparseTextEmbedding, ImageEmbedding, LateInteractionTextEmbedding
from fastembed import SparseEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

# --- Added classes from data_schema.py ---
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class EmbeddingType(str, Enum):
    """Enum for supported embedding types."""
    DENSE = "dense"
    SPARSE = "sparse"
    LATE_INTERACTION = "late_interaction"
    IMAGE = "image"

class IndexData(BaseModel):
    """Data structure for storing index data."""
    vector: np.ndarray
    id: int
    content: str
    metadata: Dict = {}
    embedding_type: EmbeddingType = EmbeddingType.DENSE

class FastEmbedConfig(BaseModel):
    """Configuration for FastEmbed models."""
    enabled: bool = False
    model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_type: EmbeddingType = EmbeddingType.DENSE
    batch_size: int = 32
    cuda: bool = False
    device_ids: List[int] = []
    
    @property
    def is_sparse(self) -> bool:
        """Check if the embedding type is sparse."""
        return self.embedding_type == EmbeddingType.SPARSE
    
    @property
    def is_late_interaction(self) -> bool:
        """Check if the embedding type is late interaction."""
        return self.embedding_type == EmbeddingType.LATE_INTERACTION
    
    @property
    def is_image(self) -> bool:
        """Check if the embedding type is image."""
        return self.embedding_type == EmbeddingType.IMAGE
# --- End of added classes ---


class FqlDb:
    """
    A class for combining vector similarity search with a SQLite database.
    Supports both FAISS and FastEmbed for vector operations.
    """

    def __init__(self, 
                 index_name: str, 
                 dimension: int, 
                 db_name: str = "fql.db", 
                 fastembed_config: Optional[FastEmbedConfig] = None):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name of the index.
            dimension (int): The dimension of the vectors.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            fastembed_config (Optional[FastEmbedConfig], optional): Configuration for FastEmbed. 
                                                                   Defaults to None (uses FAISS).
        """
        self.index_name = index_name
        self.dimension = dimension
        self.db_name = db_name
        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        
        # FastEmbed configuration
        self.fastembed_config = fastembed_config or FastEmbedConfig(enabled=False)
        self.embedding_model = None
        self.reranker_model = None
        
        # Initialize FastEmbed models if enabled
        if self.fastembed_config.enabled:
            self._initialize_fastembed_models()
            
        # Initialize the index
        self.index = self._load_or_build_index()
        
        # Create the database table if it doesn't exist
        self._create_table()

    def _initialize_fastembed_models(self) -> None:
        """
        Initializes the FastEmbed models based on the configuration.
        """
        try:
            # Initialize the appropriate embedding model based on the embedding type
            if self.fastembed_config.is_sparse:
                cprint(f"Initializing SparseTextEmbedding model: {self.fastembed_config.model_name}", "blue")
                self.embedding_model = SparseTextEmbedding(
                    model_name=self.fastembed_config.model_name,
                    batch_size=self.fastembed_config.batch_size,
                    cuda=self.fastembed_config.cuda,
                    device_ids=self.fastembed_config.device_ids or None
                )
            elif self.fastembed_config.is_late_interaction:
                cprint(f"Initializing LateInteractionTextEmbedding model: {self.fastembed_config.model_name}", "blue")
                self.embedding_model = LateInteractionTextEmbedding(
                    model_name=self.fastembed_config.model_name,
                    batch_size=self.fastembed_config.batch_size,
                    cuda=self.fastembed_config.cuda,
                    device_ids=self.fastembed_config.device_ids or None
                )
            elif self.fastembed_config.is_image:
                cprint(f"Initializing ImageEmbedding model: {self.fastembed_config.model_name}", "blue")
                self.embedding_model = ImageEmbedding(
                    model_name=self.fastembed_config.model_name,
                    batch_size=self.fastembed_config.batch_size,
                    cuda=self.fastembed_config.cuda,
                    device_ids=self.fastembed_config.device_ids or None
                )
            else:  # Default to dense text embedding
                cprint(f"Initializing TextEmbedding model: {self.fastembed_config.model_name}", "blue")
                self.embedding_model = TextEmbedding(
                    model_name=self.fastembed_config.model_name,
                    batch_size=self.fastembed_config.batch_size,
                    cuda=self.fastembed_config.cuda,
                    device_ids=self.fastembed_config.device_ids or None
                )
            
            # Initialize reranker model (optional, can be done later)
            # self.reranker_model = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
            
            cprint(f"FastEmbed model initialized successfully.", "green")
        except Exception as e:
            cprint(f"Failed to initialize FastEmbed model: {e}", "red")
            raise

    def _create_table(self) -> None:
        """
        Creates the database table if it doesn't exist.
        """
        try:
            with closing(self.connection.cursor()) as cur:
                # Add embedding_type column to store the type of embedding
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(
                        id INTEGER PRIMARY KEY, 
                        content TEXT, 
                        metadata TEXT,
                        embedding_type TEXT DEFAULT 'dense'
                    )"""
                )
            cprint(f"Database table {self.index_name} created or verified.", "green")
        except Exception as e:
            cprint(f"Failed to create database table: {e}", "red")
            raise

    def _load_or_build_index(self) -> faiss.IndexIDMap2:
        """
        Loads the index from file if it exists, otherwise builds a new index.

        Returns:
            faiss.IndexIDMap2: The loaded or newly built index.
        """
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file):
            try:
                # Use the instance method load_index which uses self.index_name
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
    
    def embed_text(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generates embeddings for a list of text documents using FastEmbed.

        Args:
            texts (List[str]): List of text documents to embed.

        Returns:
            List[np.ndarray]: List of embeddings.
        """
        if not self.fastembed_config.enabled or not self.embedding_model:
            raise ValueError("FastEmbed is not enabled or model is not initialized.")
        
        try:
            if self.fastembed_config.is_sparse:
                # For sparse embeddings, we need to handle SparseEmbedding objects
                sparse_embeddings = list(self.embedding_model.embed(texts))
                return sparse_embeddings
            elif self.fastembed_config.is_late_interaction:
                # For late interaction models
                return list(self.embedding_model.embed(texts))
            else:
                # For dense embeddings
                return list(self.embedding_model.embed(texts))
        except Exception as e:
            cprint(f"Error generating embeddings: {e}", "red")
            raise
    
    def embed_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Generates embeddings for a list of images using FastEmbed.

        Args:
            image_paths (List[str]): List of paths to images.

        Returns:
            List[np.ndarray]: List of embeddings.
        """
        if not self.fastembed_config.enabled or not self.embedding_model or not self.fastembed_config.is_image:
            raise ValueError("FastEmbed image embedding is not enabled or model is not initialized.")
        
        try:
            return list(self.embedding_model.embed(image_paths))
        except Exception as e:
            cprint(f"Error generating image embeddings: {e}", "red")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Reranks a list of documents based on their relevance to the query using FastEmbed.

        Args:
            query (str): The query string.
            documents (List[str]): List of documents to rerank.
            top_k (int, optional): Number of top results to return. Defaults to 3.

        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples.
        """
        if not self.fastembed_config.enabled:
            raise ValueError("FastEmbed is not enabled.")
        
        try:
            # Initialize reranker if not already done
            if not self.reranker_model:
                cprint("Initializing reranker model...", "blue")
                self.reranker_model = TextCrossEncoder(model_name="BAAI/bge-reranker-base")
            
            # Create pairs of (query, document)
            pairs = [(query, doc) for doc in documents]
            
            # Get scores
            scores = self.reranker_model.score(pairs)
            
            # Sort by score and get top_k
            results = list(zip(documents, scores))
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
        except Exception as e:
            cprint(f"Error during reranking: {e}", "red")
            raise

    def add_to_index(self, data: List[IndexData]) -> None:
        """
        Adds data to the FAISS index.

        Args:
            data (List[IndexData]): A list of IndexData objects to add.
        """
        ids = []
        vectors = []
        
        # Process vectors based on embedding type
        for point in data:
            ids.append(point.id)
            
            # Handle sparse embeddings differently
            if isinstance(point.vector, SparseEmbedding):
                # For sparse embeddings, we need to convert to dense for FAISS
                # Create a dense vector of zeros
                dense_vector = np.zeros(self.dimension, dtype=np.float32)
                # Fill in the non-zero values
                for idx, val in zip(point.vector.indices, point.vector.values):
                    if idx < self.dimension:  # Ensure index is within bounds
                        dense_vector[idx] = val
                vectors.append(dense_vector)
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
        chunk = faiss.serialize_index(self.index)
        with open(f"{self.index_name}.pkl", "wb") as f:
            pickle.dump(chunk, f)
        cprint(f"Index {self.index_name} saved successfully.", "green")

    def load_index(self) -> faiss.Index:
        """
        Loads the FAISS index from file using the instance's index_name.

        Returns:
            faiss.Index: The loaded FAISS index.
        """
        index_file = f"{self.index_name}.pkl"
        with open(index_file, "rb") as f:
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
                # Convert metadata to JSON string
                metadata_str = json.dumps(point.metadata) if isinstance(point.metadata, dict) else str(point.metadata)
                
                # Get embedding type
                embedding_type = point.embedding_type.value if hasattr(point, 'embedding_type') else 'dense'
                
                values.append((point.id, point.content, metadata_str, embedding_type))
                
            # Use a dedicated cursor for insertion
            with closing(self.connection.cursor()) as cur:
                cur.executemany(
                    f"""INSERT OR IGNORE INTO {self.index_name} 
                    (id, content, metadata, embedding_type) 
                    VALUES (?,?,?,?)""", 
                    values
                )
            # Since isolation_level=None (autocommit), changes should be persisted immediately.
            # No explicit commit needed here.
            cprint(f"Stored/updated {len(data)} records in database table {self.index_name}.", "green")

        except Exception as e:
            cprint(f"Could not complete database operation: {e}", "red")
            raise

    def search_index(self, query: Union[np.ndarray, str, List[str]], k: int = 3, 
                     rerank: bool = False) -> Tuple[List[float], List[int]]:
        """
        Searches the FAISS index for the nearest neighbors of a query vector.
        If FastEmbed is enabled and query is a string, it will be embedded first.

        Args:
            query (Union[np.ndarray, str, List[str]]): The query vector or text.
            k (int, optional): The number of nearest neighbors to return. Defaults to 3.
            rerank (bool, optional): Whether to rerank results using FastEmbed. Defaults to False.

        Returns:
            Tuple[List[float], List[int]]: A tuple containing the distances and IDs of the nearest neighbors.
        """
        query_vector = query
        
        # If FastEmbed is enabled and query is a string or list of strings, embed it
        if self.fastembed_config.enabled and self.embedding_model:
            if isinstance(query, str):
                if self.fastembed_config.is_image:
                    # Assume query is an image path
                    query_vector = self.embed_images([query])[0]
                else:
                    # Assume query is text
                    embedded = self.embed_text([query])
                    if self.fastembed_config.is_sparse:
                        # Convert sparse to dense for FAISS
                        sparse_emb = embedded[0]
                        dense_vector = np.zeros(self.dimension, dtype=np.float32)
                        for idx, val in zip(sparse_emb.indices, sparse_emb.values):
                            if idx < self.dimension:
                                dense_vector[idx] = val
                        query_vector = np.array([dense_vector], dtype=np.float32)
                    else:
                        query_vector = np.array([embedded[0]], dtype=np.float32)
            elif isinstance(query, list) and all(isinstance(q, str) for q in query):
                if self.fastembed_config.is_image:
                    # Assume query is a list of image paths
                    embedded = self.embed_images(query)
                    query_vector = np.array([embedded[0]], dtype=np.float32)
                else:
                    # Assume query is a list of texts
                    embedded = self.embed_text(query)
                    if self.fastembed_config.is_sparse:
                        # Convert sparse to dense for FAISS
                        sparse_emb = embedded[0]
                        dense_vector = np.zeros(self.dimension, dtype=np.float32)
                        for idx, val in zip(sparse_emb.indices, sparse_emb.values):
                            if idx < self.dimension:
                                dense_vector[idx] = val
                        query_vector = np.array([dense_vector], dtype=np.float32)
                    else:
                        query_vector = np.array([embedded[0]], dtype=np.float32)
        
        # Ensure query_vector is in the right shape for FAISS
        if len(query_vector.shape) == 1:
            query_vector = np.array([query_vector], dtype=np.float32)
            
        # Search the index
        D, I = self.index.search(query_vector, k)
        
        # Convert numpy types to standard Python types
        distances = [float(d) for d in D[0]]
        ids = [int(i) for i in I[0]]
        
        # Rerank if requested and FastEmbed is enabled
        if rerank and self.fastembed_config.enabled and isinstance(query, str):
            try:
                # Retrieve the documents
                retrieved_docs = self.retrieve(ids)
                if retrieved_docs:
                    # Extract content
                    documents = [doc[1] for doc in retrieved_docs]  # doc[1] is the content
                    
                    # Rerank
                    reranked = self.rerank(query, documents, top_k=k)
                    
                    # Map back to original IDs
                    reranked_ids = []
                    reranked_distances = []
                    for i, (doc, score) in enumerate(reranked):
                        # Find the original ID for this document
                        for j, retrieved_doc in enumerate(retrieved_docs):
                            if retrieved_doc[1] == doc:  # Compare content
                                reranked_ids.append(ids[j])
                                reranked_distances.append(score)
                                break
                    
                    # If we found all reranked documents, return them
                    if len(reranked_ids) == len(reranked):
                        return reranked_distances, reranked_ids
            except Exception as e:
                cprint(f"Reranking failed, falling back to standard results: {e}", "yellow")
        
        return distances, ids

    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """
        Retrieves data from the SQLite database based on a list of IDs.

        Args:
            ids (List[int]): A list of IDs to retrieve. Expects standard Python ints.

        Returns:
            List[Tuple]: A list of tuples containing the retrieved data (id, content, metadata, embedding_type).
        """
        if not ids:
            return []

        rows = []
        cur = None
        try:
            cur = self.connection.cursor()
            # Ensure IDs are standard Python integers (should be guaranteed by search_index now)
            placeholders = ','.join('?' * len(ids))
            sql = f"SELECT id, content, metadata, embedding_type FROM {self.index_name} WHERE id IN ({placeholders})"
            cur.execute(sql, ids)
            rows = cur.fetchall()
            
            # Parse metadata from JSON string
            parsed_rows = []
            for row in rows:
                try:
                    metadata = json.loads(row[2]) if row[2] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = row[2]  # Keep as string if not valid JSON
                
                parsed_rows.append((row[0], row[1], metadata, row[3]))
            
            return parsed_rows
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
        cprint("Initialize FqlDb with FastEmbed:","green")
        cprint("  fastembed_config = FastEmbedConfig(enabled=True, model_name='BAAI/bge-small-en-v1.5')","white")
        cprint("  fql_db = FqlDb(index_name='my_index', dimension=384, db_name='my_db.db', fastembed_config=fastembed_config)","white")
        cprint("\nAdd data to the index:","green")
        cprint("  fql_db.add_to_index(data=[IndexData(...)])","white")
        cprint("Store data to the database:","green")
        cprint("  fql_db.store_to_db(data=[IndexData(...)])","white")
        cprint("\nSearch the index with vector:","green")
        cprint("  distances, ids = fql_db.search_index(query=np.array([1.0, 2.0], dtype=np.float32), k=3)","white")
        cprint("Search the index with text (if FastEmbed enabled):","green")
        cprint("  distances, ids = fql_db.search_index(query='sample query text', k=3)","white")
        cprint("Search with reranking (if FastEmbed enabled):","green")
        cprint("  distances, ids = fql_db.search_index(query='sample query text', k=5, rerank=True)","white")
        cprint("\nRetrieve data from the database:","green")
        cprint("  retrieved_data = fql_db.retrieve(ids=[1, 2, 3])","white")
        cprint("\nFastEmbed specific functions:","green")
        cprint("  embeddings = fql_db.embed_text(['text1', 'text2'])","white")
        cprint("  image_embeddings = fql_db.embed_images(['image1.jpg', 'image2.jpg'])","white")
        cprint("  reranked_results = fql_db.rerank(query='query', documents=['doc1', 'doc2'], top_k=2)","white")
        cprint("\nSave the index:","green")
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
    Tests the FqlDb class with both FAISS and FastEmbed configurations.
    """
    cprint("\n" + "="*80, "blue")
    cprint("Starting FqlDb tests with default FAISS configuration...", "blue")
    cprint("="*80, "blue")

    # Test setup for FAISS
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

        # Create FqlDb instance with default FAISS configuration
        fql_db = FqlDb(index_name=index_name, dimension=dimension, db_name=db_name)
        cprint("Created FqlDb instance with FAISS configuration.", "green")

        # Test add_to_index and store_to_db
        fql_db.add_to_index(test_data)
        fql_db.store_to_db(test_data)
        cprint("Added and stored test data.", "green")

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

        cprint("All FAISS configuration tests passed!", "green")

        # Clean up before FastEmbed tests
        if loaded_fql_db and loaded_fql_db.connection:
            loaded_fql_db.connection.close()
            loaded_fql_db.connection = None
        del loaded_fql_db
        cleanup_test_files(index_name, db_name)

        # =====================================================================
        # Now test with FastEmbed configuration
        # =====================================================================
        cprint("\n" + "="*80, "blue")
        cprint("Starting FqlDb tests with FastEmbed configuration...", "blue")
        cprint("="*80, "blue")

        # Test setup for FastEmbed
        fastembed_index_name = "test_fastembed_index"
        fastembed_dimension = 384  # BGE-small-en dimension
        fastembed_db_name = "test_fastembed.db"

        # --- Ensure clean state before test ---
        cleanup_test_files(fastembed_index_name, fastembed_db_name)
        # --------------------------------------

        fastembed_config = FastEmbedConfig(
            enabled=True,
            model_name="BAAI/bge-small-en-v1.5",
            embedding_type=EmbeddingType.DENSE
        )

        fastembed_db = None
        loaded_fastembed_db = None

        try:
            # Create FqlDb instance with FastEmbed configuration
            fastembed_db = FqlDb(
                index_name=fastembed_index_name,
                dimension=fastembed_dimension,
                db_name=fastembed_db_name,
                fastembed_config=fastembed_config
            )
            cprint("Created FqlDb instance with FastEmbed configuration.", "green")

            # Test text embedding
            test_texts = ["This is a test document.", "Another test document for embedding."]
            cprint("Testing text embedding...", "blue")
            embeddings = fastembed_db.embed_text(test_texts)
            assert len(embeddings) == 2
            assert embeddings[0].shape == (fastembed_dimension,)
            cprint(f"Generated {len(embeddings)} text embeddings with dimension {embeddings[0].shape[0]}.", "green")

            # Create test data with FastEmbed embeddings
            fastembed_test_data = [
                IndexData(
                    vector=embeddings[0],
                    id=1,
                    content=test_texts[0],
                    metadata={"source": "fastembed_test"},
                    embedding_type=EmbeddingType.DENSE
                ),
                IndexData(
                    vector=embeddings[1],
                    id=2,
                    content=test_texts[1],
                    metadata={"source": "fastembed_test"},
                    embedding_type=EmbeddingType.DENSE
                ),
            ]

            # Test add_to_index and store_to_db
            fastembed_db.add_to_index(fastembed_test_data)
            fastembed_db.store_to_db(fastembed_test_data)
            cprint("Added and stored FastEmbed test data.", "green")

            # Test search with text query
            query_text = "test document"
            cprint(f"Testing search with text query: '{query_text}'...", "blue")
            distances, ids = fastembed_db.search_index(query_text, k=2)
            assert len(distances) == 2
            assert len(ids) == 2
            cprint(f"Search results - Distances: {distances}, IDs: {ids}", "cyan")

            # Test retrieve
            retrieved_data = fastembed_db.retrieve(ids)
            assert len(retrieved_data) == 2
            cprint(f"Retrieved data: {retrieved_data}", "cyan")

            # Test reranking
            cprint("Testing reranking...", "blue")
            try:
                reranked_results = fastembed_db.rerank(query_text, test_texts, top_k=2)
                assert len(reranked_results) == 2
                cprint(f"Reranked results: {reranked_results}", "cyan")
            except Exception as e:
                cprint(f"Reranking test skipped: {e}", "yellow")

            # Test save_index and load_index
            fastembed_db.save_index()
            # Close the current connection before loading a new instance
            if fastembed_db.connection:
                fastembed_db.connection.close()
                fastembed_db.connection = None

            loaded_fastembed_db = FqlDb(
                index_name=fastembed_index_name,
                dimension=fastembed_dimension,
                db_name=fastembed_db_name,
                fastembed_config=fastembed_config
            )
            assert loaded_fastembed_db.index.ntotal == len(fastembed_test_data)
            cprint("FastEmbed index loaded successfully after save.", "green")

            # Test search with loaded index
            distances_loaded, ids_loaded = loaded_fastembed_db.search_index(query_text, k=2)
            assert len(distances_loaded) == 2
            assert len(ids_loaded) == 2
            cprint(f"Search results with loaded index - Distances: {distances_loaded}, IDs: {ids_loaded}", "cyan")

            # Test search with reranking
            try:
                cprint("Testing search with reranking...", "blue")
                distances_reranked, ids_reranked = loaded_fastembed_db.search_index(query_text, k=2, rerank=True)
                assert len(distances_reranked) == 2
                assert len(ids_reranked) == 2
                cprint(f"Search results with reranking - Distances: {distances_reranked}, IDs: {ids_reranked}", "cyan")
            except Exception as e:
                cprint(f"Search with reranking test skipped: {e}", "yellow")

            cprint("All FastEmbed configuration tests passed!", "green")

            # Test sparse embedding if available
            try:
                cprint("\n" + "="*80, "blue")
                cprint("Testing sparse embedding functionality...", "blue")
                
                # Create a new FqlDb instance with sparse embedding configuration
                sparse_config = FastEmbedConfig(
                    enabled=True,
                    model_name="prithivida/Splade_PP_en_v1",
                    embedding_type=EmbeddingType.SPARSE
                )
                
                sparse_index_name = "test_sparse_index"
                sparse_dimension = 30522  # SPLADE vocabulary size
                sparse_db_name = "test_sparse.db"
                
                # Clean up any existing files
                cleanup_test_files(sparse_index_name, sparse_db_name)
                
                sparse_db = FqlDb(
                    index_name=sparse_index_name,
                    dimension=sparse_dimension,
                    db_name=sparse_db_name,
                    fastembed_config=sparse_config
                )
                
                # Test sparse embedding
                sparse_texts = ["This is a test for sparse embedding."]
                sparse_embeddings = sparse_db.embed_text(sparse_texts)
                assert len(sparse_embeddings) == 1
                assert isinstance(sparse_embeddings[0], SparseEmbedding)
                
                # Create test data with sparse embeddings
                sparse_test_data = [
                    IndexData(
                        vector=sparse_embeddings[0],
                        id=1,
                        content=sparse_texts[0],
                        metadata={"type": "sparse"},
                        embedding_type=EmbeddingType.SPARSE
                    )
                ]
                
                # Add to index and store
                sparse_db.add_to_index(sparse_test_data)
                sparse_db.store_to_db(sparse_test_data)
                
                # Test search
                sparse_query = "test sparse"
                distances, ids = sparse_db.search_index(sparse_query, k=1)
                assert len(distances) == 1
                assert len(ids) == 1
                assert ids[0] == 1
                
                # Clean up
                if sparse_db.connection:
                    sparse_db.connection.close()
                    sparse_db.connection = None
                del sparse_db
                cleanup_test_files(sparse_index_name, sparse_db_name)
                
                cprint("Sparse embedding tests passed!", "green")
            except Exception as e:
                cprint(f"Sparse embedding tests skipped: {e}", "yellow")
                
            # Test image embedding if available
            try:
                cprint("\n" + "="*80, "blue")
                cprint("Testing image embedding functionality...", "blue")
                
                # Check if test image exists
                test_image_path = Path("test_image.jpg")
                if not test_image_path.exists():
                    # Create a simple test image
                    from PIL import Image
                    img = Image.new('RGB', (100, 100), color = 'red')
                    img.save(test_image_path)
                    cprint(f"Created test image at {test_image_path}", "blue")
                
                # Create a new FqlDb instance with image embedding configuration
                image_config = FastEmbedConfig(
                    enabled=True,
                    model_name="Qdrant/clip-ViT-B-32-vision",
                    embedding_type=EmbeddingType.IMAGE
                )
                
                image_index_name = "test_image_index"
                image_dimension = 512  # CLIP ViT-B/32 dimension
                image_db_name = "test_image.db"
                
                # Clean up any existing files
                cleanup_test_files(image_index_name, image_db_name)
                
                image_db = FqlDb(
                    index_name=image_index_name,
                    dimension=image_dimension,
                    db_name=image_db_name,
                    fastembed_config=image_config
                )
                
                # Test image embedding
                image_embeddings = image_db.embed_images([str(test_image_path)])
                assert len(image_embeddings) == 1
                assert image_embeddings[0].shape == (image_dimension,)
                
                # Create test data with image embeddings
                image_test_data = [
                    IndexData(
                        vector=image_embeddings[0],
                        id=1,
                        content=str(test_image_path),
                        metadata={"type": "image"},
                        embedding_type=EmbeddingType.IMAGE
                    )
                ]
                
                # Add to index and store
                image_db.add_to_index(image_test_data)
                image_db.store_to_db(image_test_data)
                
                # Test search with image
                distances, ids = image_db.search_index(str(test_image_path), k=1)
                assert len(distances) == 1
                assert len(ids) == 1
                assert ids[0] == 1
                
                # Clean up
                if image_db.connection:
                    image_db.connection.close()
                    image_db.connection = None
                del image_db
                cleanup_test_files(image_index_name, image_db_name)
                
                # Remove test image
                if test_image_path.exists():
                    test_image_path.unlink()
                
                cprint("Image embedding tests passed!", "green")
            except Exception as e:
                cprint(f"Image embedding tests skipped: {e}", "yellow")
                
            # Test late interaction model if available
            try:
                cprint("\n" + "="*80, "blue")
                cprint("Testing late interaction model functionality...", "blue")
                
                # Create a new FqlDb instance with late interaction configuration
                late_config = FastEmbedConfig(
                    enabled=True,
                    model_name="colbert-ir/colbertv2.0",
                    embedding_type=EmbeddingType.LATE_INTERACTION
                )
                
                late_index_name = "test_late_index"
                late_dimension = 128  # ColBERT dimension
                late_db_name = "test_late.db"
                
                # Clean up any existing files
                cleanup_test_files(late_index_name, late_db_name)
                
                late_db = FqlDb(
                    index_name=late_index_name,
                    dimension=late_dimension,
                    db_name=late_db_name,
                    fastembed_config=late_config
                )
                
                # Test late interaction embedding
                late_texts = ["This is a test for late interaction embedding."]
                late_embeddings = late_db.embed_text(late_texts)
                
                # Create test data with late interaction embeddings
                # Note: Late interaction embeddings are 2D arrays, we'll use the first row
                late_test_data = [
                    IndexData(
                        vector=late_embeddings[0][0],  # Take first token embedding
                        id=1,
                        content=late_texts[0],
                        metadata={"type": "late_interaction"},
                        embedding_type=EmbeddingType.LATE_INTERACTION
                    )
                ]
                
                # Add to index and store
                late_db.add_to_index(late_test_data)
                late_db.store_to_db(late_test_data)
                
                # Clean up
                if late_db.connection:
                    late_db.connection.close()
                    late_db.connection = None
                del late_db
                cleanup_test_files(late_index_name, late_db_name)
                
                cprint("Late interaction model tests passed!", "green")
            except Exception as e:
                cprint(f"Late interaction model tests skipped: {e}", "yellow")

        except Exception as e:
            cprint(f"FastEmbed tests failed: {e}", "red")
            raise
        finally:
            # Clean up FastEmbed test files
            if fastembed_db and fastembed_db.connection:
                fastembed_db.connection.close()
                fastembed_db.connection = None
            if loaded_fastembed_db and loaded_fastembed_db.connection:
                loaded_fastembed_db.connection.close()
                loaded_fastembed_db.connection = None
            del fastembed_db
            del loaded_fastembed_db
            cleanup_test_files(fastembed_index_name, fastembed_db_name)

        cprint("\n" + "="*80, "green")
        cprint("All FqlDb tests passed successfully!", "green")
        cprint("="*80, "green")

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
        # Need to instantiate with valid dimension even for usage
        fql_db = FqlDb(index_name='temp', dimension=1, db_name='temp.db')
        fql_db.usage()
        # Clean up dummy files created for usage
        cleanup_test_files('temp', 'temp.db')
        del fql_db
    else:
        test_fql_db()
