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
from typing import Tuple               # typing helpers for function signatures
from PIL import Image                  # PIL for opening images

import torch                           # torch for Dataset base class and tensors
from torch.utils.data import Dataset   # dataset base class we implement below
from torchvision import transforms     # image transforms / augmentations
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

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

    cols = {c.lower(): c for c in df.columns}
    c_isic = cols.get("isic_id") or cols.get("image_id") or cols.get("image_name")
    c_pid  = cols.get("patient_id") or cols.get("lesion_id")
    c_tgt  = cols.get("target") or cols.get("melanoma") or cols.get("label") or cols.get("benign_malignant")
    if c_isic is None or c_pid is None or c_tgt is None:
        raise ValueError(f"Need isic_id/patient_id/target in metadata. Columns={list(df.columns)}")

    out = df[[c_isic, c_pid, c_tgt]].copy()
    out.columns = ["isic_id", "patient_id", "target"]

    # many ISIC archives name files as "<isic_id>.jpg"
    out["image_name"] = out["isic_id"].astype(str)

    # normalize target to {0,1}
    if out["target"].dtype == object:
        out["target"] = out["target"].str.lower().map({"benign": 0, "malignant": 1}).astype(int)

    # final column order
    return out[["image_name", "target", "patient_id", "isic_id"]]

# possible locations for the images folder (for fallback issues)
IMG_CANDIDATES = [DATA_ROOT / "train-image" / "image", DATA_ROOT / "train-image"]  

def _resolve_image_root() -> Path:
    # Check each candidate directory in order and return the first valid one.
    # Helps support both flattened and nested image archive layouts.
    for d in IMG_CANDIDATES:
        if d.exists():
            return d
    # None of the candidates exist -> raise with a clear message.
    raise FileNotFoundError("Could not find 'data/train-image' or 'data/train-image/image'.")

IMG_ROOT = _resolve_image_root()  # resolved base path where training images live

def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """chore: Will complete later
    """
    # standard ImageNet normalization values (common default)
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Minimal train transform: convert to tensor and normalize
    # (Upgraded with common dermoscopy-friendly augmentations)
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02),
        transforms.ToTensor(),   # convert PIL image to torch.FloatTensor [0,1]
        norm                     # normalize channels to mean/std
    ])

    # eval transform should match train preprocessing (no randomness)
    eval_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm
    ])

    return train_t, eval_t

class ISICDataset(Dataset):
    """chore: Will complete later
    """
    def __init__(self, df: pd.DataFrame, root: Path, tfm=None):
        # store a copy of the dataframe indexed 0..N-1 for reliable iloc
        self.df = df.reset_index(drop=True)
        # root can be a string or Path; make sure it's a Path for path ops
        self.root = Path(root)
        # tfm is a torchvision transform callable (or None); applied to PIL Image
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        # normalize suffix
        name = str(r.image_name)
        if not name.lower().endswith(".jpg"):
            name += ".jpg"
        path = self.root / name
        # open the image and ensure RGB channels
        img = Image.open(path).convert("RGB")

        # apply transform if provided (usually train transforms expect PIL -> Tensor)
        if self.tfm:
            img = self.tfm(img)

        # ensure label is an integer (some CSVs use strings/ints)
        label = int(r.target)

        # return the image tensor (or PIL if no tfm), integer label, and path as string
        return img, label, str(path)

def set_seed(s: int = 42):
    # set random seeds for python/numpy/torch to make experiments reproducible
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    # enable cuDNN benchmark for potentially faster runtime (may affect reproducibility)
    torch.backends.cudnn.benchmark = True

def get_isic2020_data(seed: int = 42):
    # load metadata and split into train / val / test (stratified by label)
    df = load_metadata()
    X = df.image_name; y = df.target.astype(int)
    # first split off a 10% test set
    Xt, Xte, yt, yte = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)
    # from remaining, split ~11.1% to get ~10% of original as validation
    # so final split is 80% train, 10% val, 10% test
    Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.111, stratify=yt, random_state=seed)  # ~10% val
    return (
        pd.DataFrame({"image_name": Xtr, "target": ytr}),
        pd.DataFrame({"image_name": Xv, "target": yv}),
        pd.DataFrame({"image_name": Xte, "target": yte}),
    )

def _make_balanced_sampler(y_series: pd.Series) -> WeightedRandomSampler:
    # weights inversely proportional to class frequency -> balances sampling
    class_counts = y_series.value_counts().to_dict()
    weights_per_class = {c: 1.0 / float(cnt) for c, cnt in class_counts.items()}
    weights = y_series.map(weights_per_class).astype(float).to_numpy()
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def get_isic2020_data_loaders(bs: int = 32, workers: int = 2, seed: int = 42, balance: bool = True):
    # build DataLoaders for train, val, and test sets using the minimal transforms
    set_seed(seed)
    tr, v, te = get_isic2020_data(seed)
    ttf, etf = get_transforms()

    trd = ISICDataset(tr, IMG_ROOT, ttf)
    vd  = ISICDataset(v, IMG_ROOT, etf)
    td  = ISICDataset(te, IMG_ROOT, etf)

    sampler = _make_balanced_sampler(tr["target"]) if balance else None

    train_loader = DataLoader(
        trd,
        batch_size=bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(vd, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(td, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader
