from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def get_vector(file_path):
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
    content_summary = ""

    text_for_encoding = f"""File named '{file_name}' is a {extension} document located in the {parent_name} directory, created in {created_date} and last modified in {modified_date} with a size of {file_size}. The file contains: {content_summary}"""
    vector = model.encode([text_for_encoding])
    print(vector)


get_vector("./testdir/coding/Numpy Vectorization.py")