"""
dataset.py
Student name: Jay Thakkar
Student number: s4786177
Description: Handles loading and sampling of image pairs from the ISIC 2020 dataset.
Stage: path resolution + CSV read only (no torch/transform yet).
"""


from __future__ import annotations  # use future annotations: lets us use forward refs in type hints
from pathlib import Path               # Path helps with OS-independent file paths
import pandas as pd                    # pandas used for csv reading and dataframe operations


def _find_data_root(start: Path) -> Path:  # search up from 'start' to find the project's data folder
    """
    Walk upward to find a directory containing 'data/train-metadata.csv'.
    Handles running from project root, ./recognition/, or this model folder.
    """
    cur = start.resolve()  # reslove the start path to an absolute path so comparisons are stable
    for _ in range(6):     # try up to 6 levels up the directory tree (heuristic)
        candidate = cur / "data" / "train-metadata.csv"  # build the expected metadata path
        if candidate.exists():  # if the file exists at this location
            return cur / "data"   # return the data folder (not the file) as the root
        cur = cur.parent  # move up one directory and try again
    # If we finish the loop without finding the file, raise a clear error
    raise FileNotFoundError(
        "could not find '../data/train-metadata.csv'. "
        "expected at <repo_root>/data/train-metadata.csv"
    )


HERE = Path(__file__).parent  # directory containing this file; start search here
DATA_ROOT = _find_data_root(HERE)  # locate the data root using the helper fuction
CSV_PATH = DATA_ROOT / "train-metadata.csv"  # full path to the csv we will load


def load_metadata() -> pd.DataFrame:  # read and normalize the metadata csv into minimal df
    """
    Load the train metadata csv and return a minimal cleaned df
    with columns: ['image_name', 'target'].
    """
    df = pd.read_csv(CSV_PATH)  # read the csv into a pandas DataFrame
    # Heuristic normalize: try to find reasonable column names users see in ISIC
    cols = {c.lower(): c for c in df.columns}  # map lowercase name -> original name for robust lookup

    def pick(*names):  # small helper to pick the first matching column name from several aliases
        for n in names:
            if n in cols:
                return cols[n]  # return the original column name that matched
        return None

    # try a set of common names for the image id column
    c_img = pick("image_name", "filename", "file_name", "image_id", "isic_id")
    # try a set of common names for the target/label column
    c_tgt = pick("target", "melanoma", "label", "diagnosis", "benign_malignant")

    # If we could not find a suitable image column, raise an informative error
    if c_img is None:
        raise ValueError(f"Could not find an image column in metadata. Columns={list(df.columns)}")
    # If we could not find a suitable target column, raise an informative error
    if c_tgt is None:
        raise ValueError(f"Could not find a target/label column in metadata. Columns={list(df.columns)}")

    out = df[[c_img, c_tgt]].copy()  # select only the two columns we need and copy to avoid view issues
    out.columns = ["image_name", "target"]  # rename to a stable, minimal API for downstream code
    return out  # return the cleaned dataframe