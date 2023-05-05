# The original data can be found at:
# https://nlp.stanford.edu/data/glove.840B.300d.zip

from os.path import exists
from os import makedirs, remove

from . import get_glove_data_home
from ._base import RemoteFileMetadata

from pathlib import Path
import zipfile  # use tarfile for tgz file
import urllib.request
import datetime

ARCHIVE = RemoteFileMetadata(
    filename="glove.840B.300d.zip",
    url="https://nlp.stanford.edu/data/glove.840B.300d.zip",
)


def fetch_glove_word_vectors(*, data_home):
    data_home = get_glove_data_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)

    zip_path = Path(f'{data_home}/{ARCHIVE.filename}')
    urllib.request.urlretrieve(ARCHIVE.url, zip_path)
    with zipfile.ZipFile(zip_path) as embedding_zip:
        embedding_zip.extractall(path="datasets")
        for info in embedding_zip.infolist():
            print(f"Filename: {info.filename}")
            print(f"Modified: {datetime.datetime(*info.date_time)}")
            print(f"Normal size: {info.file_size} bytes")
            print(f"Compressed size: {info.compress_size} bytes")
            print("-" * 20)
    print("*" * 20)
    print(" Done! ")
    print("*" * 20)
