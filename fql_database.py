import faiss
import numpy as np
import sqlite3
from typing import Optional, List, Dict, Tuple, Union, Any, Literal
from data_schema import IndexData
from contextlib import closing
import pickle
import os
from termcolor import cprint
import argparse
import math
from PIL import Image # Needed for image size check

# Attempt to import fastembed components, handle ImportError if not installed
try:
    from fastembed import TextEmbedding, ImageEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding, DefaultEmbedding
    from fastembed.rerank import TextRerank
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None
    ImageEmbedding = None
    TextRerank = None
    DefaultEmbedding = None # Define DefaultEmbedding even if unavailable for type hints

def grouper(iterable: list, n: int):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

class FqlDb:
    """
    A class combining vector similarity search (FAISS or FastEmbed) with a SQLite database.

    Supports optional internal embedding generation using FastEmbed for text and images,
    and optional re-ranking using FastEmbed's rerankers.

    Note: Sparse and Late Interaction models from FastEmbed are not directly supported
    with the default FAISS backend due to architectural differences. Use Qdrant for
    native support of those features.
    """

    def __init__(self,
                 index_name: str,
                 db_name: str = "fql.db",
                 dimension: Optional[int] = None,
                 use_fastembed: bool = False,
                 fastembed_model_name: Optional[str] = None,
                 embedding_type: Literal['text', 'image'] = 'text',
                 reranker_model_name: Optional[str] = None,
                 batch_size: int = 32,
                 cache_dir: Optional[str] = None):
        """
        Initializes the FqlDb object.

        Args:
            index_name (str): The name for the index files and database table.
            db_name (str, optional): The name of the SQLite database file. Defaults to "fql.db".
            dimension (int, optional): The dimension of the vectors. Required if use_fastembed is False.
                                      If use_fastembed is True, this is inferred from the model.
            use_fastembed (bool, optional): Whether to use FastEmbed for internal embedding generation. Defaults to False.
            fastembed_model_name (str, optional): The name of the FastEmbed model to use (e.g., 'BAAI/bge-small-en-v1.5').
                                                 Required if use_fastembed is True.
            embedding_type (Literal['text', 'image'], optional): The type of embedding model to load ('text' or 'image').
                                                               Defaults to 'text'.
            reranker_model_name (str, optional): The name of the FastEmbed reranker model to use (e.g., 'BAAI/bge-reranker-base').
                                                Defaults to None (no reranking).
            batch_size (int, optional): Batch size for FastEmbed processing. Defaults to 32.
            cache_dir (str, optional): Directory to cache downloaded FastEmbed models. Defaults to None (uses FastEmbed default).

        Raises:
            ValueError: If use_fastembed is False and dimension is not provided.
            ValueError: If use_fastembed is True and fastembed_model_name is not provided.
            ImportError: If use_fastembed or reranker_model_name is specified but 'fastembed' is not installed.
            TypeError: If an invalid embedding_type is provided.
        """
        self.index_name = index_name
        self.db_name = db_name
        self.use_fastembed = use_fastembed
        self.embedding_model = None
        self.reranker_model = None
        self.embedding_type = embedding_type
        self.batch_size = batch_size
        self._dimension = dimension # Store provided dimension

        if not FASTEMBED_AVAILABLE and (use_fastembed or reranker_model_name):
            raise ImportError("FastEmbed is not installed. Please install it with `pip install fastembed` or `pip install fastembed[image]` or `pip install fastembed[rerank]` to use this feature.")

        if use_fastembed:
            if not fastembed_model_name:
                # Default to a standard text model if none provided
                fastembed_model_name = "BAAI/bge-small-en-v1.5"
                cprint(f"No FastEmbed model name provided, defaulting to {fastembed_model_name}", "yellow")

            try:
                if embedding_type == 'text':
                    self.embedding_model = TextEmbedding(model_name=fastembed_model_name, cache_dir=cache_dir)
                elif embedding_type == 'image':
                    self.embedding_model = ImageEmbedding(model_name=fastembed_model_name, cache_dir=cache_dir)
                else:
                    raise TypeError(f"Unsupported embedding_type: {embedding_type}. Choose 'text' or 'image'.")

                # Infer dimension from the loaded model
                # Accessing dimension might differ slightly based on fastembed version/model type
                # Using a general approach that should work for dense models
                if hasattr(self.embedding_model, 'dim'):
                     self._dimension = self.embedding_model.dim
                elif hasattr(self.embedding_model, 'model') and hasattr(self.embedding_model.model, 'model') and hasattr(self.embedding_model.model.model, 'config') and hasattr(self.embedding_model.model.model.config, 'hidden_size'):
                     # Fallback for some structures
                     self._dimension = self.embedding_model.model.model.config.hidden_size
                else:
                     # Attempt to get dim via list_supported_models as last resort
                     models_list = TextEmbedding.list_supported_models() if embedding_type == 'text' else ImageEmbedding.list_supported_models()
                     model_info = next((m for m in models_list if m['model'] == fastembed_model_name), None)
                     if model_info and 'dim' in model_info:
                         self._dimension = model_info['dim']
                     else:
                         raise ValueError(f"Could not automatically determine dimension for FastEmbed model {fastembed_model_name}. Please provide it manually.")

                cprint(f"FastEmbed {embedding_type} model '{fastembed_model_name}' loaded. Dimension: {self._dimension}", "green")

            except Exception as e:
                cprint(f"Failed to load FastEmbed model '{fastembed_model_name}': {e}", "red")
                raise

        elif self._dimension is None:
            raise ValueError("Vector dimension must be provided if use_fastembed is False.")

        if reranker_model_name:
            try:
                self.reranker_model = TextRerank(model_name=reranker_model_name, cache_dir=cache_dir)
                cprint(f"FastEmbed reranker model '{reranker_model_name}' loaded.", "green")
            except Exception as e:
                cprint(f"Failed to load FastEmbed reranker model '{reranker_model_name}': {e}", "red")
                raise

        # Now use self.dimension (either provided or inferred)
        self.dimension = self._dimension
        self.connection = sqlite3.Connection(self.db_name, isolation_level=None)
        self.index = self._load_or_build_index()


    def _load_or_build_index(self) -> faiss.IndexIDMap2:
        """
        Loads the FAISS index from file if it exists, otherwise builds a new index.
        Uses the instance's dimension.
        """
        index_file = f"{self.index_name}.pkl"
        if os.path.exists(index_file):
            try:
                loaded_index = self._load_index_internal()
                # Verify dimension match
                if loaded_index.d != self.dimension:
                     cprint(f"Warning: Loaded index dimension ({loaded_index.d}) does not match expected dimension ({self.dimension}). Rebuilding index.", "yellow")
                     index = self._build_index_internal(self.dimension)
                else:
                     index = loaded_index
                     cprint(f"Index '{self.index_name}' loaded successfully (Dimension: {index.d}).", "green")
            except Exception as e:
                cprint(f"Failed to load index '{self.index_name}': {e}. Building a new one.", "yellow")
                index = self._build_index_internal(self.dimension)
                cprint(f"Index '{self.index_name}' created successfully (Dimension: {self.dimension}).", "green")
        else:
            index = self._build_index_internal(self.dimension)
            cprint(f"Index '{self.index_name}' created successfully (Dimension: {self.dimension}).", "green")
        return index

    def _build_index_internal(self, dimension: int) -> faiss.IndexIDMap2:
        """Internal method to build a FAISS index."""
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        return index

    def _add_vectors_to_index(self, data: List[IndexData]) -> None:
        """Internal method to add vector data to the FAISS index."""
        ids = []
        vectors = []
        valid_data_count = 0
        for point in data:
            if point.vector is not None and point.vector.shape == (self.dimension,):
                 ids.append(point.id)
                 vectors.append(point.vector)
                 valid_data_count += 1
            else:
                 cprint(f"Skipping data point with id {point.id} due to missing or incorrect vector dimension.", "yellow")

        if not vectors:
             cprint("No valid vectors provided to add to the index.", "yellow")
             return

        ids_np = np.array(ids, dtype=np.int64)
        vectors_np = np.array(vectors, dtype=np.float32)

        try:
            self.index.add_with_ids(vectors_np, ids_np)
            cprint(f"Added {valid_data_count} vectors to index '{self.index_name}'.", "green")
        except Exception as e:
             cprint(f"Error adding vectors to FAISS index: {e}", "red")
             # Potentially re-raise or handle more gracefully depending on desired behavior
             raise


    def save_index(self) -> None:
        """Saves the FAISS index to a file."""
        index_file = f"{self.index_name}.pkl"
        try:
            chunk = faiss.serialize_index(self.index)
            with open(index_file, "wb") as f:
                pickle.dump(chunk, f)
            cprint(f"Index '{self.index_name}' saved successfully to {index_file}.", "green")
        except Exception as e:
            cprint(f"Error saving index '{self.index_name}' to {index_file}: {e}", "red")
            raise

    def _load_index_internal(self) -> faiss.Index:
        """Internal method to load the FAISS index from file."""
        index_file = f"{self.index_name}.pkl"
        try:
            with open(index_file, "rb") as f:
                index = faiss.deserialize_index(pickle.load(f))
            return index
        except FileNotFoundError:
            cprint(f"Index file '{index_file}' not found.", "red")
            raise
        except Exception as e:
            cprint(f"Error loading index '{self.index_name}' from {index_file}: {e}", "red")
            raise

    def _store_content_to_db(self, data: List[IndexData]) -> None:
        """Internal method to store content and metadata to the SQLite database."""
        if not data:
            cprint("No data provided to store in the database.", "yellow")
            return

        try:
            values = []
            for point in data:
                # Ensure metadata is stored as a string (e.g., JSON)
                metadata_str = str(point.metadata) if point.metadata is not None else None
                values.append((point.id, point.content, metadata_str))

            with closing(self.connection.cursor()) as cur:
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(id INTEGER PRIMARY KEY, content TEXT, metadata TEXT)"""
                )
                cur.executemany(
                    f"""INSERT OR REPLACE INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)""", values
                )
            cprint(f"Stored/updated {len(data)} records in database table '{self.index_name}'.", "green")

        except sqlite3.Error as e:
            cprint(f"SQLite error during store operation: {e}", "red")
            raise
        except Exception as e:
            cprint(f"Unexpected error during store operation: {e}", "red")
            raise

    def add(self,
            contents: List[Any],
            ids: List[int],
            metadata: Optional[List[Optional[Dict[str, Any]]]] = None) -> None:
        """
        Adds content to the database and index. Generates embeddings if use_fastembed is True.

        Args:
            contents (List[Any]): A list of content items (text strings or image file paths).
            ids (List[int]): A list of unique integer IDs corresponding to the content items.
            metadata (Optional[List[Optional[Dict[str, Any]]]], optional): A list of metadata dictionaries,
                                                                        one for each content item. Defaults to None.

        Raises:
            ValueError: If lists of contents, ids, and metadata (if provided) have different lengths.
            ValueError: If use_fastembed is True but the embedding model is not initialized.
            FileNotFoundError: If embedding_type is 'image' and an image path is invalid.
            Exception: If embedding generation fails.
        """
        if not len(contents) == len(ids):
            raise ValueError("Length of 'contents' and 'ids' lists must be the same.")
        if metadata is not None and not len(contents) == len(metadata):
            raise ValueError("Length of 'contents' and 'metadata' lists must be the same.")

        if metadata is None:
            metadata = [{} for _ in contents] # Ensure metadata list exists

        index_data_list: List[IndexData] = []

        if self.use_fastembed:
            if not self.embedding_model:
                raise ValueError("use_fastembed is True, but the embedding model is not initialized.")

            cprint(f"Generating embeddings for {len(contents)} items using FastEmbed...", "cyan")
            all_vectors = []
            processed_count = 0
            try:
                # Process in batches
                for batch_contents in grouper(contents, self.batch_size):
                    if self.embedding_type == 'image':
                        # Check image paths and basic validity before embedding
                        valid_batch_contents = []
                        for item in batch_contents:
                            if isinstance(item, str) and os.path.exists(item):
                                try:
                                    # Try opening image to catch basic errors
                                    with Image.open(item) as img:
                                        img.verify() # Verify image data
                                    valid_batch_contents.append(item)
                                except FileNotFoundError:
                                     cprint(f"Image file not found: {item}. Skipping.", "yellow")
                                except Exception as img_err:
                                     cprint(f"Error processing image {item}: {img_err}. Skipping.", "yellow")
                            else:
                                cprint(f"Invalid image input: {item}. Skipping.", "yellow")
                        if not valid_batch_contents:
                            continue # Skip empty batches
                        batch_vectors = list(self.embedding_model.embed(valid_batch_contents))
                        all_vectors.extend(batch_vectors)
                        processed_count += len(valid_batch_contents)
                    else: # 'text'
                        # Assume text content is valid strings
                        batch_vectors = list(self.embedding_model.embed(batch_contents))
                        all_vectors.extend(batch_vectors)
                        processed_count += len(batch_contents)

                cprint(f"Generated {len(all_vectors)} embeddings.", "green")

                # Create IndexData objects - Need to align vectors with original ids/metadata
                # This assumes the embedding model returns vectors in the same order as input
                # and skips items that failed validation (especially for images)
                current_vector_idx = 0
                for i, content_item in enumerate(contents):
                    # Determine if this item was successfully processed
                    should_have_vector = True
                    if self.embedding_type == 'image':
                         if not (isinstance(content_item, str) and os.path.exists(content_item)):
                             should_have_vector = False
                         else:
                              try:
                                  with Image.open(content_item) as img:
                                      img.verify()
                              except:
                                  should_have_vector = False

                    if should_have_vector and current_vector_idx < len(all_vectors):
                         vector = all_vectors[current_vector_idx]
                         index_data_list.append(IndexData(
                             id=ids[i],
                             content=content_item, # Store original content (path for image)
                             vector=vector,
                             metadata=metadata[i]
                         ))
                         current_vector_idx += 1
                    # else: item was skipped, don't create IndexData

            except Exception as e:
                cprint(f"Error during FastEmbed embedding generation: {e}", "red")
                raise # Re-raise to indicate failure

        else:
            # If not using fastembed, vectors must be provided externally beforehand.
            # This method assumes content is added, but vectors need separate handling.
            # Let's modify this: if not use_fastembed, this method ONLY stores content.
            # The user must call _add_vectors_to_index separately with IndexData containing vectors.
            cprint("use_fastembed is False. Storing content only. Vectors must be added separately using _add_vectors_to_index.", "yellow")
            for i, content_item in enumerate(contents):
                 index_data_list.append(IndexData(
                     id=ids[i],
                     content=content_item,
                     vector=None, # No vector generated here
                     metadata=metadata[i]
                 ))

        # Add vectors to index (only if they were generated)
        if self.use_fastembed and index_data_list:
             self._add_vectors_to_index(index_data_list)

        # Store content and metadata to DB
        if index_data_list:
             self._store_content_to_db(index_data_list)


    def _search_index_internal(self, query_vector: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """
        Internal method to search the FAISS index.

        Args:
            query_vector (np.ndarray): The query vector (should be 2D, e.g., [[...]]).
            k (int, optional): The number of nearest neighbors. Defaults to 3.

        Returns:
            Tuple[List[float], List[int]]: Distances and IDs of neighbors.
        """
        if query_vector.shape != (1, self.dimension):
             cprint(f"Query vector has incorrect shape {query_vector.shape}. Expected (1, {self.dimension}). Reshaping.", "yellow")
             try:
                 query_vector = query_vector.reshape(1, self.dimension)
             except ValueError as e:
                 cprint(f"Cannot reshape query vector: {e}", "red")
                 return [], [] # Return empty results on error

        try:
            D, I = self.index.search(query_vector, k)
            # Convert numpy types to standard Python types
            distances = [float(d) for d in D[0]]
            ids = [int(i) for i in I[0] if i != -1] # Filter out -1 which indicates fewer than k results
            # Adjust distances list if IDs were filtered
            distances = distances[:len(ids)]
            return distances, ids
        except Exception as e:
            cprint(f"Error during FAISS search: {e}", "red")
            return [], []


    def search(self,
               query: Any,
               k: int = 3,
               rerank: bool = False,
               rerank_k: int = 10) -> List[Dict[str, Any]]:
        """
        Searches for content similar to the query.

        Args:
            query (Any): The query content (text string or image file path).
            k (int, optional): The number of final results to return. Defaults to 3.
            rerank (bool, optional): Whether to perform re-ranking using FastEmbed reranker. Defaults to False.
            rerank_k (int, optional): The number of initial candidates to fetch for re-ranking. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the retrieved 'id',
                                  'content', 'metadata', and 'score' (similarity or rerank score).
                                  Sorted by score (descending for similarity, descending for rerank).

        Raises:
            ValueError: If use_fastembed is True but the embedding model is not initialized.
            ValueError: If rerank is True but the reranker model is not initialized.
            FileNotFoundError: If embedding_type is 'image' and the query image path is invalid.
            Exception: If embedding or reranking fails.
        """
        if not self.use_fastembed:
            raise ValueError("Cannot use `search` method with raw query when `use_fastembed` is False. Use `_search_index_internal` with a pre-computed vector.")
        if not self.embedding_model:
            raise ValueError("use_fastembed is True, but the embedding model is not initialized.")
        if rerank and not self.reranker_model:
             raise ValueError("rerank is True, but the reranker model is not initialized.")

        query_vector_np = None
        try:
            cprint(f"Generating query embedding for: {query}", "cyan")
            if self.embedding_type == 'image':
                if isinstance(query, str) and os.path.exists(query):
                     try:
                         with Image.open(query) as img:
                             img.verify()
                         query_vector_np = next(self.embedding_model.embed([query])) # Get first item from generator
                     except FileNotFoundError:
                          raise FileNotFoundError(f"Query image file not found: {query}")
                     except Exception as img_err:
                          raise ValueError(f"Error processing query image {query}: {img_err}")
                else:
                     raise ValueError(f"Invalid query image input: {query}")
            else: # 'text'
                 if isinstance(query, str):
                     query_vector_np = next(self.embedding_model.embed([query]))
                 else:
                      raise ValueError(f"Invalid query text input: {query}")

            if query_vector_np is None:
                 raise ValueError("Failed to generate query embedding.")

            query_vector_faiss = query_vector_np.reshape(1, -1).astype('float32')
            cprint("Query embedding generated.", "green")

        except Exception as e:
            cprint(f"Error generating query embedding: {e}", "red")
            raise

        # Determine number of candidates to fetch
        search_k = rerank_k if rerank else k

        # Perform initial search
        distances, ids = self._search_index_internal(query_vector_faiss, k=search_k)
        if not ids:
            cprint("Initial search returned no results.", "yellow")
            return []

        # Retrieve content for candidates
        retrieved_docs_tuples = self.retrieve(ids)
        if not retrieved_docs_tuples:
             cprint("Could not retrieve content for initial search results.", "yellow")
             return []

        # Create a mapping from id to retrieved data for easier lookup
        retrieved_docs_map = {row[0]: {'id': row[0], 'content': row[1], 'metadata': row[2]} for row in retrieved_docs_tuples}

        # Add initial similarity scores (convert L2 distance to similarity if needed, 1 / (1 + L2))
        results_with_scores = []
        for db_id, dist in zip(ids, distances):
             if db_id in retrieved_docs_map:
                  doc_info = retrieved_docs_map[db_id]
                  doc_info['score'] = 1.0 / (1.0 + dist) # Example similarity score
                  results_with_scores.append(doc_info)


        if rerank and self.reranker_model:
            cprint(f"Reranking top {len(results_with_scores)} candidates...", "cyan")
            # Prepare documents for reranker [(query, document_content), ...]
            # Ensure document content is text for reranking
            rerank_pairs = []
            valid_results_for_rerank = []
            for doc in results_with_scores:
                 if isinstance(doc['content'], str):
                      rerank_pairs.append((query, doc['content']))
                      valid_results_for_rerank.append(doc)
                 else:
                      cprint(f"Skipping item id {doc['id']} from reranking as content is not text.", "yellow")

            if not rerank_pairs:
                 cprint("No valid text documents found for reranking.", "yellow")
                 # Return initial results sorted by similarity, limited to k
                 results_with_scores.sort(key=lambda x: x['score'], reverse=True)
                 return results_with_scores[:k]

            try:
                # Get reranking scores
                rerank_scores = self.reranker_model.rank(query=query, docs=[p[1] for p in rerank_pairs], raw_scores=False) # Get scores between 0 and 1

                # Combine scores with documents
                reranked_results = []
                for i, score_info in enumerate(rerank_scores):
                     # score_info is {'score': float, 'doc_id': int} where doc_id is the index in the input list
                     original_doc_index = score_info['doc_id']
                     doc_info = valid_results_for_rerank[original_doc_index]
                     doc_info['score'] = score_info['score'] # Update score with rerank score
                     reranked_results.append(doc_info)

                # Sort by new rerank score
                reranked_results.sort(key=lambda x: x['score'], reverse=True)
                cprint("Reranking complete.", "green")
                return reranked_results[:k] # Return top k reranked results

            except Exception as e:
                cprint(f"Error during reranking: {e}", "red")
                # Fallback to returning initial similarity results if reranking fails
                results_with_scores.sort(key=lambda x: x['score'], reverse=True)
                return results_with_scores[:k]

        else:
            # If not reranking, just sort initial results by similarity score and return top k
            results_with_scores.sort(key=lambda x: x['score'], reverse=True)
            return results_with_scores[:k]


    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """
        Retrieves data from the SQLite database based on a list of IDs.

        Args:
            ids (List[int]): A list of IDs to retrieve. Expects standard Python ints.

        Returns:
            List[Tuple]: A list of tuples containing the retrieved (id, content, metadata).
        """
        if not ids:
            cprint("Retrieve called with empty ID list.", "yellow")
            return []

        rows = []
        cur = None
        try:
            cur = self.connection.cursor()
            # Ensure IDs are standard Python integers
            placeholders = ','.join('?' * len(ids))
            sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"
            cur.execute(sql, ids)
            rows = cur.fetchall()
            if len(rows) != len(ids):
                 found_ids = {row[0] for row in rows}
                 missing_ids = [i for i in ids if i not in found_ids]
                 cprint(f"Warning: Could not retrieve all requested IDs. Missing: {missing_ids}", "yellow")
        except sqlite3.Error as e:
            cprint(f"SQLite error during retrieve: {e}", "red")
            raise # Re-raise the exception
        except Exception as e:
             cprint(f"An unexpected error occurred during retrieve: {e}", "red")
             raise
        finally:
            if cur:
                cur.close()
        return rows

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

        cprint("\nInitialization:", "green")
        cprint("  # Default (External Embeddings):", "cyan")
        cprint("  fql_db_default = FqlDb(index_name='my_index', dimension=384, db_name='my_db.db')", "white")
        cprint("\n  # Using FastEmbed (Text):", "cyan")
        cprint("  fql_db_fast_text = FqlDb(index_name='fast_text_index', use_fastembed=True,", "white")
        cprint("                           fastembed_model_name='BAAI/bge-small-en-v1.5', db_name='fast_text.db')", "white")
        cprint("\n  # Using FastEmbed (Image):", "cyan")
        cprint("  fql_db_fast_image = FqlDb(index_name='fast_image_index', use_fastembed=True,", "white")
        cprint("                            fastembed_model_name='Qdrant/clip-ViT-B-32-vision', embedding_type='image', db_name='fast_image.db')", "white")
        cprint("\n  # Using FastEmbed with Reranking:", "cyan")
        cprint("  fql_db_rerank = FqlDb(index_name='rerank_index', use_fastembed=True,", "white")
        cprint("                        fastembed_model_name='BAAI/bge-small-en-v1.5',", "white")
        cprint("                        reranker_model_name='BAAI/bge-reranker-base', db_name='rerank.db')", "white")

        cprint("\nAdding Data:", "green")
        cprint("  # If use_fastembed=True:", "cyan")
        cprint("  contents = ['Some text', 'Another text'] # Or ['img1.jpg', 'img2.png'] for images", "white")
        cprint("  ids = [1, 2]", "white")
        cprint("  metadata = [{'source': 'doc1'}, {'source': 'doc2'}]", "white")
        cprint("  fql_db_fast_text.add(contents=contents, ids=ids, metadata=metadata)", "white")
        cprint("\n  # If use_fastembed=False (Add content first, then vectors separately):", "cyan")
        cprint("  fql_db_default.add(contents=contents, ids=ids, metadata=metadata) # Stores content only", "white")
        cprint("  # --- Generate vectors externally ---", "grey")
        cprint("  vectors = [np.array(...), np.array(...)]", "white")
        cprint("  index_data = [IndexData(id=ids[i], content=contents[i], vector=vectors[i], metadata=metadata[i]) for i in range(len(ids))]", "white")
        cprint("  fql_db_default._add_vectors_to_index(data=index_data) # Internal method for adding pre-computed vectors", "white")


        cprint("\nSearching:", "green")
        cprint("  # If use_fastembed=True:", "cyan")
        cprint("  query = 'Search query text' # Or 'query_img.jpg' for images", "white")
        cprint("  results = fql_db_fast_text.search(query=query, k=5)", "white")
        cprint("  results_reranked = fql_db_rerank.search(query=query, k=5, rerank=True, rerank_k=20)", "white")
        cprint("\n  # If use_fastembed=False (Search with pre-computed vector):", "cyan")
        cprint("  query_vector = np.array([[...]], dtype=np.float32)", "white")
        cprint("  distances, ids = fql_db_default._search_index_internal(query_vector=query_vector, k=5)", "white")
        cprint("  retrieved_data = fql_db_default.retrieve(ids=ids)", "white")

        cprint("\nRetrieving by ID:", "green")
        cprint("  retrieved_data = fql_db_default.retrieve(ids=[1, 2, 3])", "white")

        cprint("\nSaving Index:", "green")
        cprint("  fql_db_default.save_index()", "white")

        cprint("\nNotes:", "yellow")
        cprint("  - Sparse and Late Interaction models require different backends (like Qdrant).", "grey")
        cprint("  - Ensure 'fastembed' (and relevant extras like [image], [rerank]) are installed if `use_fastembed=True`.", "grey")
        cprint("---------------------", "blue", attrs=["bold"])


def cleanup_test_files(*names):
    """Removes specified files if they exist."""
    files_removed = False
    for name in names:
        try:
            if os.path.exists(name):
                os.remove(name)
                files_removed = True
        except Exception as e:
            cprint(f"Error removing file {name}: {e}", "red")
    if files_removed:
        cprint(f"Cleaned up test files: {', '.join(names)}", "yellow")


# --- Test Helper Functions ---
def create_dummy_image(path, size=(64, 64), color="red"):
    """Creates a small dummy image file."""
    try:
        img = Image.new('RGB', size, color=color)
        img.save(path)
        return True
    except Exception as e:
        cprint(f"Failed to create dummy image {path}: {e}", "red")
        return False

# --- Updated Test Function ---
def test_fql_db():
    """Tests the FqlDb class with default and FastEmbed configurations."""
    cprint("\n--- Starting FqlDb Tests ---", "blue", attrs=["bold"])

    # --- Test Case 1: Default Configuration (External Embeddings) ---
    cprint("\n[Test Case 1: Default (External Embeddings)]", "cyan")
    index_name_def = "test_index_default"
    db_name_def = "test_default.db"
    dimension_def = 2
    fql_db_def = None
    loaded_fql_db_def = None
    cleanup_test_files(f"{index_name_def}.pkl", db_name_def)

    try:
        test_data_def = [
            IndexData(id=1, content="Default content 1", vector=np.array([1.0, 2.0], dtype=np.float32), metadata={"key": "val1"}),
            IndexData(id=2, content="Default content 2", vector=np.array([3.0, 4.0], dtype=np.float32), metadata={"key": "val2"}),
            IndexData(id=3, content="Default content 3", vector=np.array([5.0, 6.0], dtype=np.float32), metadata={"key": "val3"}),
        ]
        fql_db_def = FqlDb(index_name=index_name_def, dimension=dimension_def, db_name=db_name_def)

        # Test adding content and vectors separately
        fql_db_def.add(contents=[d.content for d in test_data_def], ids=[d.id for d in test_data_def], metadata=[d.metadata for d in test_data_def])
        fql_db_def._add_vectors_to_index(test_data_def) # Add vectors using internal method

        assert fql_db_def.index.ntotal == len(test_data_def)
        cprint("Data added successfully (Default).", "green")

        # Test search and retrieve
        query_vector_def = np.array([[2.0, 3.0]], dtype=np.float32)
        distances, ids = fql_db_def._search_index_internal(query_vector_def, k=2)
        assert len(ids) == 2, f"Expected 2 IDs, got {len(ids)}"
        ids.sort()
        assert ids == [1, 2]
        cprint(f"Search successful (Default): IDs={ids}, Distances={distances}", "green")

        retrieved_data = fql_db_def.retrieve(ids)
        assert len(retrieved_data) == 2, f"Expected 2 retrieved items, got {len(retrieved_data)}"
        retrieved_ids = sorted([row[0] for row in retrieved_data])
        assert retrieved_ids == ids
        cprint(f"Retrieve successful (Default): {retrieved_data}", "green")

        # Test save and load
        fql_db_def.save_index()
        if fql_db_def.connection: fql_db_def.connection.close(); fql_db_def.connection = None

        loaded_fql_db_def = FqlDb(index_name=index_name_def, dimension=dimension_def, db_name=db_name_def)
        assert loaded_fql_db_def.index.ntotal == len(test_data_def)
        cprint("Save/Load successful (Default).", "green")

        # Test search with loaded index
        distances_load, ids_load = loaded_fql_db_def._search_index_internal(query_vector_def, k=2)
        ids_load.sort()
        assert ids_load == ids
        cprint("Search after load successful (Default).", "green")

        cprint("[Test Case 1 Passed]", "green", attrs=["bold"])

    except Exception as e:
        cprint(f"[Test Case 1 Failed]: {e}", "red", attrs=["bold"])
        raise
    finally:
        if fql_db_def and fql_db_def.connection: fql_db_def.connection.close()
        if loaded_fql_db_def and loaded_fql_db_def.connection: loaded_fql_db_def.connection.close()
        del fql_db_def
        del loaded_fql_db_def
        cleanup_test_files(f"{index_name_def}.pkl", db_name_def)

    # --- Test Case 2: FastEmbed Text ---
    if FASTEMBED_AVAILABLE:
        cprint("\n[Test Case 2: FastEmbed Text]", "cyan")
        index_name_ft = "test_index_fast_text"
        db_name_ft = "test_fast_text.db"
        model_name_ft = "BAAI/bge-small-en-v1.5" # Ensure this model is supported by your fastembed version
        fql_db_ft = None
        loaded_fql_db_ft = None
        cleanup_test_files(f"{index_name_ft}.pkl", db_name_ft)

        try:
            test_contents_ft = ["This is the first document.", "This is the second document.", "A third piece of text."]
            test_ids_ft = [10, 11, 12]
            test_metadata_ft = [{"topic": "A"}, {"topic": "B"}, {"topic": "A"}]

            fql_db_ft = FqlDb(index_name=index_name_ft, db_name=db_name_ft,
                              use_fastembed=True, fastembed_model_name=model_name_ft, embedding_type='text')

            fql_db_ft.add(contents=test_contents_ft, ids=test_ids_ft, metadata=test_metadata_ft)
            assert fql_db_ft.index.ntotal == len(test_contents_ft)
            cprint("Data added successfully (FastEmbed Text).", "green")

            # Test search
            query_ft = "A document about text"
            search_results_ft = fql_db_ft.search(query=query_ft, k=2)
            assert len(search_results_ft) == 2, f"Expected 2 search results, got {len(search_results_ft)}"
            assert all('id' in r and 'score' in r for r in search_results_ft)
            cprint(f"Search successful (FastEmbed Text): Top result ID={search_results_ft[0]['id']}, Score={search_results_ft[0]['score']:.4f}", "green")

            # Test save and load
            fql_db_ft.save_index()
            if fql_db_ft.connection: fql_db_ft.connection.close(); fql_db_ft.connection = None

            loaded_fql_db_ft = FqlDb(index_name=index_name_ft, db_name=db_name_ft,
                                     use_fastembed=True, fastembed_model_name=model_name_ft, embedding_type='text')
            assert loaded_fql_db_ft.index.ntotal == len(test_contents_ft)
            cprint("Save/Load successful (FastEmbed Text).", "green")

            # Test search with loaded index
            search_results_load_ft = loaded_fql_db_ft.search(query=query_ft, k=2)
            assert len(search_results_load_ft) == 2
            assert search_results_load_ft[0]['id'] == search_results_ft[0]['id'] # Check if top result is consistent
            cprint("Search after load successful (FastEmbed Text).", "green")

            cprint("[Test Case 2 Passed]", "green", attrs=["bold"])

        except Exception as e:
            cprint(f"[Test Case 2 Failed]: {e}", "red", attrs=["bold"])
            raise
        finally:
            if fql_db_ft and fql_db_ft.connection: fql_db_ft.connection.close()
            if loaded_fql_db_ft and loaded_fql_db_ft.connection: loaded_fql_db_ft.connection.close()
            del fql_db_ft
            del loaded_fql_db_ft
            cleanup_test_files(f"{index_name_ft}.pkl", db_name_ft)
    else:
        cprint("\nSkipping FastEmbed tests as the library is not installed.", "yellow")

    # --- Test Case 3: FastEmbed Image ---
    if FASTEMBED_AVAILABLE and ImageEmbedding:
        cprint("\n[Test Case 3: FastEmbed Image]", "cyan")
        index_name_fi = "test_index_fast_image"
        db_name_fi = "test_fast_image.db"
        model_name_fi = "Qdrant/clip-ViT-B-32-vision" # Ensure this model is supported
        img1_path = "dummy_image1.png"
        img2_path = "dummy_image2.jpg"
        img3_path = "dummy_image3.png"
        dummy_files = [img1_path, img2_path, img3_path]
        fql_db_fi = None
        loaded_fql_db_fi = None
        cleanup_test_files(f"{index_name_fi}.pkl", db_name_fi, *dummy_files) # Clean up images too

        # Create dummy images
        img1_created = create_dummy_image(img1_path, color="blue")
        img2_created = create_dummy_image(img2_path, color="green")
        img3_created = create_dummy_image(img3_path, color="blue") # Similar to img1

        if not (img1_created and img2_created and img3_created):
             cprint("Failed to create dummy images, skipping Image test.", "red")
        else:
            try:
                test_contents_fi = [img1_path, img2_path, img3_path]
                test_ids_fi = [20, 21, 22]
                test_metadata_fi = [{"color": "blue"}, {"color": "green"}, {"color": "blue"}]

                fql_db_fi = FqlDb(index_name=index_name_fi, db_name=db_name_fi,
                                  use_fastembed=True, fastembed_model_name=model_name_fi, embedding_type='image')

                fql_db_fi.add(contents=test_contents_fi, ids=test_ids_fi, metadata=test_metadata_fi)
                assert fql_db_fi.index.ntotal == len(test_contents_fi)
                cprint("Data added successfully (FastEmbed Image).", "green")

                # Test search (query with an image similar to the first/third)
                query_fi = img1_path
                search_results_fi = fql_db_fi.search(query=query_fi, k=2)
                assert len(search_results_fi) == 2
                # Expect img1 (id 20) and img3 (id 22) to be the top results due to color similarity
                result_ids = {r['id'] for r in search_results_fi}
                assert 20 in result_ids and 22 in result_ids
                cprint(f"Search successful (FastEmbed Image): Top result IDs={result_ids}", "green")

                # Test save and load
                fql_db_fi.save_index()
                if fql_db_fi.connection: fql_db_fi.connection.close(); fql_db_fi.connection = None

                loaded_fql_db_fi = FqlDb(index_name=index_name_fi, db_name=db_name_fi,
                                         use_fastembed=True, fastembed_model_name=model_name_fi, embedding_type='image')
                assert loaded_fql_db_fi.index.ntotal == len(test_contents_fi)
                cprint("Save/Load successful (FastEmbed Image).", "green")

                # Test search with loaded index
                search_results_load_fi = loaded_fql_db_fi.search(query=query_fi, k=2)
                assert len(search_results_load_fi) == 2
                result_ids_load = {r['id'] for r in search_results_load_fi}
                assert 20 in result_ids_load and 22 in result_ids_load
                cprint("Search after load successful (FastEmbed Image).", "green")

                cprint("[Test Case 3 Passed]", "green", attrs=["bold"])

            except Exception as e:
                cprint(f"[Test Case 3 Failed]: {e}", "red", attrs=["bold"])
                raise
            finally:
                if fql_db_fi and fql_db_fi.connection: fql_db_fi.connection.close()
                if loaded_fql_db_fi and loaded_fql_db_fi.connection: loaded_fql_db_fi.connection.close()
                del fql_db_fi
                del loaded_fql_db_fi
                cleanup_test_files(f"{index_name_fi}.pkl", db_name_fi, *dummy_files) # Cleanup images
    else:
        cprint("\nSkipping FastEmbed Image tests.", "yellow")


    # --- Test Case 4: FastEmbed Text with Reranking ---
    if FASTEMBED_AVAILABLE and TextRerank:
        cprint("\n[Test Case 4: FastEmbed Text with Reranking]", "cyan")
        index_name_rr = "test_index_rerank"
        db_name_rr = "test_rerank.db"
        model_name_rr = "BAAI/bge-small-en-v1.5"
        reranker_name_rr = "BAAI/bge-reranker-base" # Ensure this model is supported
        fql_db_rr = None
        loaded_fql_db_rr = None
        cleanup_test_files(f"{index_name_rr}.pkl", db_name_rr)

        try:
            # More nuanced content for reranking
            test_contents_rr = [
                "The quick brown fox jumps over the lazy dog.", # Target 1
                "A fast, dark-colored fox leaps above a sleepy canine.", # Similar meaning
                "Weather forecast predicts sunshine tomorrow.", # Irrelevant
                "The lazy dog slept under the tree.", # Related entities, different action
                "A quick brown rabbit hops quickly." # Different animal
            ]
            test_ids_rr = [30, 31, 32, 33, 34]
            test_metadata_rr = [{"id": 30}, {"id": 31}, {"id": 32}, {"id": 33}, {"id": 34}]

            fql_db_rr = FqlDb(index_name=index_name_rr, db_name=db_name_rr,
                              use_fastembed=True, fastembed_model_name=model_name_rr,
                              reranker_model_name=reranker_name_rr, embedding_type='text')

            fql_db_rr.add(contents=test_contents_rr, ids=test_ids_rr, metadata=test_metadata_rr)
            assert fql_db_rr.index.ntotal == len(test_contents_rr)
            cprint("Data added successfully (Rerank).", "green")

            # Test search without reranking
            query_rr = "fox jumping over dog"
            search_results_no_rerank = fql_db_rr.search(query=query_rr, k=3, rerank=False)
            cprint(f"Search results (No Rerank): {[r['id'] for r in search_results_no_rerank]}", "cyan")
            # Likely results based on dense similarity: 30, 31, 33

            # Test search WITH reranking
            search_results_rerank = fql_db_rr.search(query=query_rr, k=3, rerank=True, rerank_k=5) # Rerank top 5
            assert len(search_results_rerank) == 3
            assert all('id' in r and 'score' in r for r in search_results_rerank)
            # Expect reranker to prioritize 30 and 31 higher than 33
            reranked_ids = [r['id'] for r in search_results_rerank]
            cprint(f"Search results (Reranked): {reranked_ids}", "cyan")
            assert reranked_ids[0] in [30, 31] # Top result should be one of the most relevant
            assert reranked_ids[1] in [30, 31]
            assert 33 not in reranked_ids[:2] # Less likely to be top 2 after reranking

            cprint("Search with reranking successful.", "green")

            # Test save/load (optional for rerank, mainly tests index/db persistence)
            fql_db_rr.save_index()
            if fql_db_rr.connection: fql_db_rr.connection.close(); fql_db_rr.connection = None
            loaded_fql_db_rr = FqlDb(index_name=index_name_rr, db_name=db_name_rr,
                                     use_fastembed=True, fastembed_model_name=model_name_rr,
                                     reranker_model_name=reranker_name_rr, embedding_type='text')
            assert loaded_fql_db_rr.index.ntotal == len(test_contents_rr)
            search_results_load_rerank = loaded_fql_db_rr.search(query=query_rr, k=3, rerank=True, rerank_k=5)
            assert [r['id'] for r in search_results_load_rerank] == reranked_ids
            cprint("Save/Load and search after load successful (Rerank).", "green")


            cprint("[Test Case 4 Passed]", "green", attrs=["bold"])

        except Exception as e:
            cprint(f"[Test Case 4 Failed]: {e}", "red", attrs=["bold"])
            raise
        finally:
            if fql_db_rr and fql_db_rr.connection: fql_db_rr.connection.close()
            if loaded_fql_db_rr and loaded_fql_db_rr.connection: loaded_fql_db_rr.connection.close()
            del fql_db_rr
            del loaded_fql_db_rr
            cleanup_test_files(f"{index_name_rr}.pkl", db_name_rr)
    else:
        cprint("\nSkipping FastEmbed Reranking tests.", "yellow")


    cprint("\n--- All FqlDb tests completed ---", "blue", attrs=["bold"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test FqlDb or print usage.")
    parser.add_argument("--usage", action="store_true", help="Print FqlDb usage instructions.")
    args = parser.parse_args()

    # Dummy instantiation for usage printing - outside try/except
    # Needs dimension if use_fastembed=False (default)
    fql_db_usage = FqlDb(index_name='temp_usage', dimension=1, db_name='temp_usage.db')

    if args.usage:
        fql_db_usage.usage()
    else:
        try:
            test_fql_db()
        except Exception as e:
             cprint(f"\n--- Test Suite Failed: {e} ---", "red", attrs=["bold"])
             # Optionally exit with non-zero status
             # sys.exit(1)

    # Cleanup dummy usage files
    del fql_db_usage # Ensure __del__ is called if needed
    cleanup_test_files("temp_usage.pkl", "temp_usage.db")
