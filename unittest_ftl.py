import unittest
import os
import shutil
import tempfile
import numpy as np
from termcolor import cprint, colored
import sqlite3

# Assuming ftl_db.py is in the same directory or accessible via PYTHONPATH
from ftl_db import FtlDb, IndexData, _print_error, _print_success, _print_info

# --- Configuration ---
TEST_MODEL = "all-MiniLM-L6-v2" # Use a standard small model for testing
# Get dimension dynamically - safer than hardcoding
try:
    from sentence_transformers import SentenceTransformer
    temp_model = SentenceTransformer(TEST_MODEL)
    TEST_DIMENSION = temp_model.get_sentence_embedding_dimension()
    del temp_model # Clean up the temporary model instance
except ImportError:
    print("Warning: sentence_transformers not installed. Using default dimension 384. Install for accurate testing.")
    TEST_DIMENSION = 384
except Exception as e:
    print(f"Warning: Could not load model '{TEST_MODEL}' to get dimension. Using default 384. Error: {e}")
    TEST_DIMENSION = 384


TEST_DATA = [
    "This is the first test sentence.",
    "Here is another sentence for testing.",
    "Vector databases are interesting.",
    "Testing the FtlDb implementation.",
    "Hoping these tests pass successfully.",
]

# --- Custom Test Runner for Colored Output ---

class ColorTextTestResult(unittest.TextTestResult):
    """A test result class that prints colored messages."""
    def addSuccess(self, test):
        super().addSuccess(test)
        cprint(f"PASS: {test.id()}", "green")

    def addError(self, test, err):
        super().addError(test, err)
        cprint(f"ERROR: {test.id()}", "red", attrs=["bold"])
        # Optionally print error details
        # self.stream.writeln(self._exc_info_to_string(err, test))

    def addFailure(self, test, err):
        super().addFailure(test, err)
        cprint(f"FAIL: {test.id()}", "yellow", attrs=["bold"])
        # Optionally print failure details
        # self.stream.writeln(self._exc_info_to_string(err, test))

class ColorTextTestRunner(unittest.TextTestRunner):
    """A test runner that uses the colored result class."""
    resultclass = ColorTextTestResult

    def run(self, test):
        cprint("\n--- Starting Test Suite ---", "cyan", attrs=["bold"])
        result = super().run(test)
        cprint("--- Test Suite Finished ---", "cyan", attrs=["bold"])
        # Print summary
        if result.wasSuccessful():
             _print_success(f"All {result.testsRun} tests passed!")
        else:
            _print_error(f"Test Suite Failed: {len(result.failures)} failures, {len(result.errors)} errors.")
        return result


# --- Test Class ---

class TestFtlDb(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.index_name = "test_index"
        self.db_path = os.path.join(self.test_dir, f"{self.index_name}.db")
        self.index_path = os.path.join(self.test_dir, f"{self.index_name}.pkl")
        # Ensure clean state before each test
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        _print_info(f"\nRunning test: {self.id()}")
        _print_info(f"Using temp directory: {self.test_dir}")


    def tearDown(self):
        """Clean up the temporary directory."""
        # Attempt to close any open FtlDb instances gracefully if they exist
        # This is tricky as the instance might be local to the test method.
        # Relying on __del__ or context manager is better practice in the tests themselves.
        if hasattr(self, 'db') and self.db:
             try:
                 self.db.close()
             except Exception:
                 pass # Ignore errors during cleanup closing
        shutil.rmtree(self.test_dir)
        _print_info(f"Cleaned up temp directory: {self.test_dir}")

    def test_01_initialization_new(self):
        """Test creating a new FtlDb instance."""
        with FtlDb(index_name=self.index_name,
                   dimension=TEST_DIMENSION,
                   db_path=self.db_path,
                   index_path=self.index_path,
                   model_name=TEST_MODEL) as db:
            self.assertEqual(db.index_name, self.index_name)
            self.assertEqual(db.dimension, TEST_DIMENSION)
            self.assertIsNotNone(db.index)
            self.assertEqual(db.index.ntotal, 0)
            self.assertIsNotNone(db.connection)
            self.assertTrue(os.path.exists(self.db_path))
            # Index file might not exist until save is called, which happens on __exit__
        self.assertTrue(os.path.exists(self.index_path)) # Check after context exit

    def test_02_initialization_rebuild(self):
        """Test the rebuild=True flag."""
        # Create initial files
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add(["initial data"])
        self.assertTrue(os.path.exists(self.db_path))
        self.assertTrue(os.path.exists(self.index_path))
        self.assertGreater(db.index.ntotal, 0)

        # Re-initialize with rebuild=True
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL, rebuild=True) as db_rebuilt:
            self.assertEqual(db_rebuilt.index.ntotal, 0) # Index should be empty
            # Check DB is also empty
            with db_rebuilt._db_cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.index_name}")
                count = cur.fetchone()[0]
            self.assertEqual(count, 0)
        self.assertTrue(os.path.exists(self.db_path)) # Files should still exist but be empty/new
        self.assertTrue(os.path.exists(self.index_path))


    def test_03_add_single_item(self):
        """Test adding a single item."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            item = "A single test item."
            added_ids = db.add([item])
            self.assertEqual(len(added_ids), 1)
            self.assertEqual(added_ids[0], 0) # First ID should be 0
            self.assertEqual(db.index.ntotal, 1)
            self.assertEqual(db._next_id, 1)

            # Verify in DB
            with db._db_cursor() as cur:
                cur.execute(f"SELECT id, content FROM {self.index_name} WHERE id = ?", (added_ids[0],))
                res = cur.fetchone()
            self.assertIsNotNone(res)
            self.assertEqual(res[0], added_ids[0])
            self.assertEqual(res[1], item)

    def test_04_add_multiple_items_batch(self):
        """Test adding multiple items using batching."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            added_ids = db.add(TEST_DATA, batch_size=2)
            self.assertEqual(len(added_ids), len(TEST_DATA))
            self.assertListEqual(added_ids, list(range(len(TEST_DATA))))
            self.assertEqual(db.index.ntotal, len(TEST_DATA))
            self.assertEqual(db._next_id, len(TEST_DATA))

            # Verify count in DB
            with db._db_cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.index_name}")
                count = cur.fetchone()[0]
            self.assertEqual(count, len(TEST_DATA))

    def test_05_search_basic(self):
        """Test basic search functionality."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add(TEST_DATA, batch_size=3)
            query = "What is FtlDb?"
            k = 2
            results = db.search(query, k=k)

            self.assertEqual(len(results), k)
            for res_id, content, distance, metadata in results:
                self.assertIsInstance(res_id, int)
                self.assertIsInstance(content, str)
                self.assertIsInstance(distance, float)
                self.assertIsInstance(metadata, dict)
                self.assertIn(content, TEST_DATA) # Check if content is from original data
                self.assertGreaterEqual(distance, 0.0) # Distances should be non-negative

    def test_06_search_exact_match(self):
        """Test searching for an item already in the index."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add(TEST_DATA)
            query = TEST_DATA[2] # "Vector databases are interesting."
            k = 1
            results = db.search(query, k=k)

            self.assertEqual(len(results), k)
            res_id, content, distance, metadata = results[0]
            self.assertEqual(content, query)
            # Distance for an exact match should be very close to 0
            self.assertLess(distance, 1e-5)

    def test_07_search_empty_index(self):
        """Test searching when the index is empty."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            results = db.search("query on empty", k=1)
            self.assertEqual(len(results), 0)

    def test_08_search_no_results(self):
        """Test search where k is larger than index size."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add([TEST_DATA[0]]) # Add only one item
            k = 5
            results = db.search("test query", k=k)
            # Should return only the available item(s)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][1], TEST_DATA[0])


    def test_09_save_and_load(self):
        """Test saving the index and loading it back."""
        # Phase 1: Create, add data, and save (implicitly via context manager)
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add(TEST_DATA)
            self.assertEqual(db.index.ntotal, len(TEST_DATA))
        # db is now closed, index saved

        # Phase 2: Load the saved index and verify
        with FtlDb(index_name=self.index_name, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db_loaded:
            self.assertEqual(db_loaded.dimension, TEST_DIMENSION) # Dimension should be loaded
            self.assertEqual(db_loaded.index.ntotal, len(TEST_DATA))
            self.assertEqual(db_loaded._next_id, len(TEST_DATA)) # next_id should be restored

            # Perform a search to ensure data integrity
            query = TEST_DATA[1]
            results = db_loaded.search(query, k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][1], query)
            self.assertLess(results[0][2], 1e-5) # Distance should be near zero

    def test_10_add_invalid_data(self):
        """Test adding data that is not a list of strings."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            with self.assertRaises(ValueError):
                db.add("this is not a list") # type: ignore
            with self.assertRaises(ValueError):
                db.add([1, 2, 3]) # type: ignore

    def test_11_search_invalid_query(self):
        """Test searching with an invalid query type."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add([TEST_DATA[0]])
            with self.assertRaises(ValueError):
                db.search(None, k=1) # type: ignore
            with self.assertRaises(ValueError):
                db.search("", k=1) # Empty string

    def test_12_search_invalid_k(self):
        """Test searching with an invalid k value."""
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add([TEST_DATA[0]])
            with self.assertRaises(ValueError):
                db.search("query", k=0)
            with self.assertRaises(ValueError):
                db.search("query", k=-1)
            with self.assertRaises(ValueError):
                db.search("query", k="abc") # type: ignore

    def test_13_context_manager(self):
        """Test that the context manager saves and closes."""
        db_instance = FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL)
        with db_instance as db:
            db.add([TEST_DATA[0]])
            self.assertIsNotNone(db.connection) # Connection should be open inside 'with'
            self.assertTrue(os.path.exists(self.db_path))
            # Index file might not exist yet

        # After 'with' block:
        self.assertTrue(os.path.exists(self.index_path)) # Index should have been saved
        self.assertIsNone(db_instance.connection) # Connection should be closed

        # Try using the closed connection (should fail)
        with self.assertRaises(ConnectionError):
             with db_instance._db_cursor() as cur:
                 cur.execute("SELECT 1") # type: ignore

    def test_14_dimension_mismatch_error(self):
        """Test error handling for dimension mismatch during init."""
        wrong_dimension = TEST_DIMENSION + 1
        with self.assertRaises(ValueError):
             FtlDb(index_name=self.index_name,
                   dimension=wrong_dimension, # Provide wrong dimension
                   db_path=self.db_path,
                   index_path=self.index_path,
                   model_name=TEST_MODEL)

    def test_15_load_dimension_mismatch(self):
        """Test loading an index with a different dimension than the model expects."""
         # Phase 1: Create index with correct dimension
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add([TEST_DATA[0]])
        # db closed, index saved

        # Phase 2: Attempt to load with a different dimension specified (should use loaded dimension)
        wrong_dimension = TEST_DIMENSION + 1
        # Expecting FtlDb to print an error but override the wrong dimension with the loaded one
        print("\n--- Expecting Dimension Conflict Warning ---")
        with FtlDb(index_name=self.index_name,
                   dimension=wrong_dimension, # Provide wrong dimension
                   db_path=self.db_path,
                   index_path=self.index_path,
                   model_name=TEST_MODEL) as db_loaded:
            self.assertEqual(db_loaded.dimension, TEST_DIMENSION) # Should revert to loaded dimension
            self.assertEqual(db_loaded.index.ntotal, 1)
        print("--- End Expected Warning ---")

    def test_16_metadata_handling(self):
        """Test adding and retrieving data with metadata (implicitly tested via search)."""
        # The current FtlDb implementation doesn't allow adding custom metadata directly via `add`.
        # Metadata is added as an empty dict by default in IndexData.
        # We test that the search returns the metadata column correctly.
        with FtlDb(index_name=self.index_name, dimension=TEST_DIMENSION, db_path=self.db_path, index_path=self.index_path, model_name=TEST_MODEL) as db:
            db.add([TEST_DATA[0]])
            results = db.search(TEST_DATA[0], k=1)
            self.assertEqual(len(results), 1)
            res_id, content, distance, metadata = results[0]
            self.assertIsInstance(metadata, dict)
            # Default metadata is an empty dict, represented as '{}' in DB
            # The eval in search should convert it back to {}
            self.assertEqual(metadata, {})

            # Manually insert data with metadata to test retrieval
            test_meta = {"source": "manual_test", "value": 123}
            manual_id = 999
            vector = db.model.encode(["manual metadata test"])[0]
            with db._db_cursor() as cur:
                 cur.execute(f"INSERT INTO {self.index_name} (id, content, metadata) VALUES (?, ?, ?)",
                             (manual_id, "manual metadata test", str(test_meta)))
            # Add vector to index separately (not ideal, but needed for test)
            db.index.add_with_ids(np.array([vector]).astype(np.float32), np.array([manual_id]))

            # Search for the manually added item
            results_manual = db.search("manual metadata test", k=1)
            found = False
            for res_id, content, distance, metadata in results_manual:
                if res_id == manual_id:
                    self.assertEqual(content, "manual metadata test")
                    self.assertEqual(metadata, test_meta)
                    found = True
                    break
            self.assertTrue(found, "Manually added item with metadata not found in search results.")


# --- Main Execution ---

if __name__ == '__main__':
    suite = unittest.TestSuite()
    # Run tests in a specific order if needed (e.g., test creation before loading)
    test_loader = unittest.TestLoader()
    test_names = sorted([name for name in dir(TestFtlDb) if name.startswith('test_')])
    for test_name in test_names:
        suite.addTest(TestFtlDb(test_name))

    # Use the custom runner
    runner = ColorTextTestRunner(verbosity=0) # Set verbosity=0 to suppress default dots
    runner.run(suite)
