import faiss
import numpy as np
import sqlite3
from typing import List, Dict, Tuple, Optional, Generator, Any
from contextlib import closing
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import unittest
from termcolor import colored
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ValidationError
import os # Added for tearDown
import ast # Added for literal_eval

# Note: The use of a class structure here follows the user request,
# despite potentially conflicting with conventions discouraging OOP.

# --- Data Schema (Moved outside FqlDb class) ---
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True
        # Pydantic v2 compatibility if needed:
        # protected_namespaces = ()

class IndexData(BaseModel):
    vector: np.ndarray
    id: int
    content: str
    metadata: Dict = {}

# --- FqlDb Class ---
class FqlDb:

    def __init__(self, index_name: str = "fql_index", dimension: Optional[int] = None, db_name: str = "fql.db", model_name: str = "all-MiniLM-L6-v2", step_size: int = 3):
        self.index_name = index_name
        self.db_name = db_name
        self.model = SentenceTransformer(model_name)
        # Infer dimension from the model if not provided
        self.dimension = dimension if dimension is not None else self.model.get_sentence_embedding_dimension()
        self.step_size = step_size
        self._initialize_resources()
        self._next_id = self._get_max_id() + 1

    def _initialize_resources(self):
        """Initializes or loads FAISS index and SQLite connection."""
        self.connection = sqlite3.connect(self.db_name, isolation_level=None)
        sqlite3.register_adapter(np.int64, int)
        try:
            self.index = self.load_index()
            # Ensure table exists even if index loaded
            self._create_table_if_not_exists()
            print(colored(f"Loaded existing index '{self.index_name}' with {self.index.ntotal} vectors.", "yellow"))
        except FileNotFoundError:
            print(colored(f"No existing index found. Creating new index '{self.index_name}'.", "yellow"))
            self.index = self.build_index(dimension=self.dimension)
            self._create_table_if_not_exists()
        except Exception as e:
            print(colored(f"Error initializing resources: {e}", "red"))
            # Fallback to creating new resources
            self.index = self.build_index(dimension=self.dimension)
            self._create_table_if_not_exists()


    def _create_table_if_not_exists(self):
        """Creates the SQLite table if it doesn't exist."""
        try:
            with self.connection:
                self.connection.execute(
                    f"""CREATE TABLE IF NOT EXISTS {self.index_name}(id INTEGER PRIMARY KEY, content TEXT, metadata TEXT)"""
                )
        except Exception as e:
            print(colored(f"Error creating table '{self.index_name}': {e}", "red"))


    def _get_max_id(self) -> int:
        """Gets the maximum ID currently in the database table."""
        try:
            with closing(self.connection.cursor()) as cur:
                # Ensure table exists before querying
                self._create_table_if_not_exists()
                cur.execute(f"SELECT MAX(id) FROM {self.index_name}")
                result = cur.fetchone()
                return result[0] if result and result[0] is not None else -1
        except Exception as e:
            # Handle case where table might not exist yet or other SQL errors
            print(colored(f"Could not retrieve max ID, starting from 0. Error: {e}", "yellow"))
            return -1


    def build_index(self, dimension: int) -> faiss.IndexIDMap2:
        """Builds a new FAISS index."""
        flat_index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap2(flat_index)
        return index

    def add_to_index(self, data: List[IndexData]) -> None:
        """Adds data points (vector + id) to the FAISS index."""
        if not data:
            return
        ids = np.array([point.id for point in data], dtype=np.int64)
        vectors = np.array([point.vector for point in data], dtype=np.float32)
        self.index.add_with_ids(vectors, ids)

    def save_index(self) -> None:
        """Saves the current FAISS index to a file."""
        try:
            chunk = faiss.serialize_index(self.index)
            with open(f"{self.index_name}.pkl", "wb") as f:
                pickle.dump(chunk, f)
            print(colored(f"Index saved to {self.index_name}.pkl", "green"))
        except Exception as e:
            print(colored(f"Error saving index: {e}", "red"))


    def load_index(self) -> faiss.IndexIDMap2:
        """Loads a FAISS index from a file."""
        # FileNotFoundError will be caught by _initialize_resources
        with open(f"{self.index_name}.pkl", "rb") as f:
            index = faiss.deserialize_index(pickle.load(f))
        # Ensure the loaded index is of the expected type
        if not isinstance(index, faiss.IndexIDMap2):
             raise TypeError("Loaded index is not of type IndexIDMap2")
        return index


    def store_to_db(self, data: List[IndexData]) -> None:
        """Stores metadata (content, metadata dict) to the SQLite database."""
        if not data:
            return
        try:
            values = [(point.id, point.content, str(point.metadata)) for point in data]
            with self.connection:
                # Table creation is handled in initialization
                self.connection.executemany(
                    f"INSERT INTO {self.index_name} (id, content, metadata) VALUES (?,?,?)", values
                )
        except sqlite3.Error as e:
            print(colored(f"SQLite error during store: {e}", "red"))
        except Exception as e:
            print(colored(f"Unexpected error during store: {e}", "red"))


    def search_index(self, query_vector: np.ndarray, k: int = 3) -> Tuple[List[float], List[int]]:
        """Performs a search on the FAISS index."""
        if self.index.ntotal == 0:
             print(colored("Warning: Searching an empty index.", "yellow"))
             return [], []
        # Ensure query_vector is 2D
        if query_vector.ndim == 1:
            query_vector = np.array([query_vector], dtype=np.float32)
        elif query_vector.dtype != np.float32:
             query_vector = query_vector.astype(np.float32)

        distances, ids = self.index.search(query_vector, k)
        return list(distances[0]), list(ids[0])


    def retrieve(self, ids: List[int]) -> List[Tuple]:
        """Retrieves metadata from SQLite based on IDs."""
        if not ids:
            return []
        rows = []
        try:
            with closing(self.connection.cursor()) as cur:
                # Use parameterized query for IN clause
                placeholders = ','.join('?' * len(ids))
                sql = f"SELECT id, content, metadata FROM {self.index_name} WHERE id IN ({placeholders})"
                cur.execute(sql, ids)
                rows = cur.fetchall()
        except sqlite3.Error as e:
             print(colored(f"SQLite error during retrieve: {e}", "red"))
        except Exception as e:
            print(colored(f"Unexpected error during retrieve: {e}", "red"))

        # Ensure results are in the same order as input IDs
        ordered_rows_dict = {row[0]: row for row in rows}
        ordered_results = [ordered_rows_dict.get(id) for id in ids if id in ordered_rows_dict]
        return ordered_results


    @staticmethod
    def grouper(iterable: list, n: int) -> Generator[list, None, None]:
        """Yields successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    def add_documents(self, documents: List[str]):
        """Encodes, indexes, and stores a list of text documents."""
        print(colored(f"Adding {len(documents)} documents...", "cyan"))
        for batch in tqdm(list(self.grouper(documents, self.step_size)), desc="Processing batches"):
            # Calculate embeddings
            embeddings = self.model.encode(batch)
            all_points = []
            for i in range(len(batch)):
                current_id = self._next_id
                point = IndexData(vector=embeddings[i], content=batch[i], id=current_id)
                self._next_id += 1
                all_points.append(point)

            # Add to index and database
            self.add_to_index(data=all_points)
            self.store_to_db(data=all_points)
        print(colored("Document addition complete.", "green"))
        # Note: Saving the index is now an explicit step, call self.save_index() separately if needed.


    def search(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        """Encodes a text query, searches the index, and retrieves results."""
        if not query_text:
            return []
        query_embedding = self.model.encode([query_text])
        distances, ids = self.search_index(query_vector=query_embedding, k=k)
        retrieved_data = self.retrieve(ids=ids)

        results = []
        for i, row in enumerate(retrieved_data):
            if row:
                # Safely evaluate metadata string back to dict
                metadata_dict = {}
                try:
                    # Use ast.literal_eval for safer evaluation than eval()
                    metadata_dict = ast.literal_eval(row[2]) if row[2] else {}
                except (ValueError, SyntaxError):
                     print(colored(f"Warning: Could not parse metadata for ID {row[0]}: {row[2]}", "yellow"))
                     metadata_dict = {"raw_metadata": row[2]} # Keep raw string if parsing fails

                # Ensure index i is valid for distances list
                if i < len(distances):
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": metadata_dict,
                        "distance": distances[i] # Corresponding distance
                    })
                else:
                    # Handle cases where retrieve might return fewer items than k if some IDs weren't found
                    print(colored(f"Warning: Mismatch between retrieved data and distances for ID {row[0]}. Skipping distance.", "yellow"))
                    results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": metadata_dict,
                        "distance": None # Indicate missing distance
                    })
        return results


    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
            print(colored("Database connection closed.", "yellow"))


# --- Unit Tests ---
class TestFqlDb(unittest.TestCase):
    def setUp(self):
        self.index_name = "test_index"
        self.db_name = "test.db"
        # Clean up any old test files before starting
        self._cleanup_files()
        self.db = FqlDb(index_name=self.index_name, db_name=self.db_name, step_size=2) # Smaller step for testing
        self.data_chunks = [
            "The weather is lovely today.",
            "It's so sunny outside!",
            "He drove to the stadium.",
            "Mayank is a software developer.",
            "I don't think it works.",
            "why do you want to know about my profession?",
            "what is your salary?",
            "have conviction in the power of goodness.",
            "Mayank works as a software developer."
        ]
        self.db.add_documents(self.data_chunks)
        # Explicitly save after adding documents for testing load/search
        self.db.save_index()
        # Close and reopen to simulate loading
        self.db.close()
        self.db = FqlDb(index_name=self.index_name, db_name=self.db_name)


    def test_search_and_retrieve(self):
        query = "Who is Mayank?"
        results = self.db.search(query_text=query, k=2)

        print(colored("\n--- Test: Search and Retrieve ---", "blue"))
        print(colored("Query:", "cyan"), query)
        print(colored("Results:", "cyan"))
        for res in results:
             # Corrected line: removed unnecessary backslash and nested f-string
             print(f"  ID: {colored(res['id'], 'yellow')}, Dist: {colored(f'{res[\"distance\"]:.4f}', 'magenta')}, Content: {colored(res['content'], 'white')}")

        self.assertGreater(len(results), 0, "Should return at least one result")
        self.assertLessEqual(len(results), 2, "Should return at most k=2 results")
        # Check if the content is relevant (simple check)
        self.assertTrue(any("Mayank" in res['content'] for res in results), "Expected relevant content not found")
        print(colored("Status: PASSED", "green"))

    def test_empty_search(self):
        query = "skdjhflskdjhflksdjhf" # Unlikely to match anything
        results = self.db.search(query_text=query, k=1)
        print(colored("\n--- Test: Empty Search Result ---", "blue"))
        print(colored("Query:", "cyan"), query)
        print(colored("Results:", "cyan"), results)
        # Depending on the index and data, it might still return the closest, even if distant.
        # self.assertEqual(len(results), 0) # This might fail, FAISS always returns k results
        self.assertLessEqual(len(results), 1)
        print(colored("Status: PASSED (by returning <= k results)", "green"))


    def _cleanup_files(self):
        """Utility to remove test files."""
        for f in [f"{self.db_name}", f"{self.index_name}.pkl"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError as e:
                    print(colored(f"Error removing file {f}: {e}", "red"))

    def tearDown(self):
        self.db.close()
        self._cleanup_files()
        print(colored("Test environment cleaned up.", "yellow"))

if __name__ == '__main__':
    # Provides more verbose output
    # Added import for ast
    import ast
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)

