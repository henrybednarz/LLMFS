import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Optional


class FileVectorDatabase:
    def __init__(self, expected_size: int = 1_000_000):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.d = 384  # Dimensionality
        self.m = 32  # Connectedness
        self.expected_size = expected_size

        self.index = faiss.IndexHNSWFlat(self.d, self.m)

        self.file_paths = []
        self.file_metadata = []
        self.is_trained = False

    def normalize_embeddings(self, embeddings: np.ndarray):
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _extract_text_from_file(self, file_path: str):
        content = ""
        return content

    def add_file(self, file_path: str, custom_metadata: Optional[Dict] = None):
        path_obj = Path(file_path)
        stat = path_obj.stat()
        file_path = str(path_obj.resolve())

        if not os.path.exists(file_path):
            return False

        file_name = path_obj.name
        extension = path_obj.suffix.lower()
        parent_name = path_obj.parent
        file_size = os.path.getsize(file_path)
        created_date = stat.st_ctime
        modified_date = stat.st_mtime
        content_summary = self._extract_text_from_file(file_path)

        text_for_encoding = f"""File named '{file_name}' is a {extension} document located in the {parent_name} directory, created in {created_date} and last modified in {modified_date} with a size of {file_size}. The file contains: {content_summary}"""

        embedding = self.model.encode([text_for_encoding])
        normalized_embedding = self.normalize_embeddings(embedding)
        self.index.add(normalized_embedding.astype('float32'))
        self.file_paths.append(file_path)
        return True

    def search(self, query: str, top_k: int = 5):
        if len(self.file_paths) == 0:
            return []

        query_embedding = self.model.encode([query])
        normalized_query = self.normalize_embeddings(query_embedding)

        scores, indices = self.index.search(normalized_query.astype('float32'), top_k)

        query_scores = scores[0]  # Shape: (top_k,)
        query_indices = indices[0]  # Shape: (top_k,)

        results = []
        for score, idx in zip(query_scores, query_indices):
            if idx >= 0 and idx < len(self.file_paths):
                results.append(self.file_paths[idx])

        return results


def scan_directory(root_path: str, max_files: Optional[int] = None) -> List[str]:

    file_paths = []
    for root, dirs, files in os.walk(root_path):

        dirs[:] = [d for d in dirs if not d.startswith('.')]

        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

                if max_files and len(file_paths) >= max_files:
                    return file_paths

    return file_paths


# Example usage
if __name__ == "__main__":
    # Configuration
    DIRECTORY_TO_SCAN = "./testdir"
    DATABASE_PATH = "./file_database"
    MAX_FILES = 10000  # Limit for testing, remove for full scan

    # Step 1: Scan for files
    all_files = scan_directory(DIRECTORY_TO_SCAN, max_files=MAX_FILES)

    if not all_files:
        exit(1)

    db = FileVectorDatabase(expected_size=len(all_files))
    files = scan_directory("./testdir")
    for f in files:
        db.add_file(f)

    test_queries = [
        "python script",
        "configuration file",
        "documentation",
        "image files"
    ]

    for query in test_queries:
        print(f"-----QUERY: {query}-----")
        results = db.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"{result}")
        print("----------------")

