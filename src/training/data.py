"""JigsawDataset and DataLoader for 32x32 jigsaw puzzle training."""
import os
import json
import random
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_SIZE = (32, 32)
NUM_PATCHES = 4
INPUT_SUBDIR = "acv_train_32x32"
SPLITS_SUBDIR = "acv_train_32x32_splits"


def _list_image_files(input_dir: str) -> List[str]:
    supported = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    if not os.path.exists(input_dir):
        return []
    return sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(supported)
    )


def _create_splits(
    input_dir: str,
    splits_dir: str,
    shuffle: bool = True,
    seed: int = 2025,
    train_ratio: float = 0.8,
) -> None:
    image_files = _list_image_files(input_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}.")
    if shuffle:
        random.seed(seed)
        random.shuffle(image_files)
    n = len(image_files)
    n_train = int(n * train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    split_data = {"train": train_files, "val": val_files}
    os.makedirs(splits_dir, exist_ok=True)
    json_path = os.path.join(splits_dir, "split.json")
    with open(json_path, "w") as f:
        json.dump(split_data, f, indent=4)
    print(f"Saved {json_path} ({len(train_files)} train, {len(val_files)} val)")


class JigsawDataset:
    """Dataset of images with random patch shuffling. dataset_dir is parent (e.g. 'datasets')."""

    image_size = IMAGE_SIZE
    num_patches = NUM_PATCHES

    def __init__(self, dataset_dir: str, train: bool = True):
        self.dataset_dir = dataset_dir.rstrip(os.sep)
        self.train = train
        input_dir = os.path.join(self.dataset_dir, INPUT_SUBDIR)
        splits_dir = os.path.join(self.dataset_dir, SPLITS_SUBDIR)
        json_path = os.path.join(splits_dir, "split.json")
        if not os.path.exists(json_path):
            _create_splits(input_dir, splits_dir, shuffle=True, seed=2025, train_ratio=0.8)
        with open(json_path, "r") as f:
            split_data = json.load(f)
        self.file_list = split_data["train"] if train else split_data["val"]
        self.input_dir = input_dir
        self.patch_size = IMAGE_SIZE[0] // NUM_PATCHES
        self._images = None

    def _do_preload(self) -> np.ndarray:
        n = len(self.file_list)
        h, w = self.image_size[0], self.image_size[1]
        arr = np.zeros((n, h, w, 1), dtype=np.float32)
        for i, f in enumerate(self.file_list):
            img_path = os.path.join(self.input_dir, f)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load {img_path}.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, self.image_size)
            if image.ndim == 2:
                image = image[:, :, None]
            image = image.astype(np.float32) / 255.0
            arr[i] = image
        return arr

    def _ensure_preloaded(self) -> None:
        if self._images is None:
            self._images = self._do_preload()

    def _shuffle_patches(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W, C = image.shape
        ph = pw = self.patch_size
        num_h = num_w = NUM_PATCHES
        num_patches = num_h * num_w
        patches = []
        for i in range(num_patches):
            row, col = i // num_w, i % num_w
            patch = image[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw, :]
            patches.append(patch)
        patches = np.array(patches)
        indices = np.arange(num_patches)
        np.random.shuffle(indices)
        shuffled = np.zeros_like(image)
        for idx, patch_idx in enumerate(indices):
            row, col = idx // num_w, idx % num_w
            shuffled[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw, :] = patches[patch_idx]
        return shuffled, indices

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_preloaded()
        image = self._images[idx]
        shuffled, indices = self._shuffle_patches(image)
        chw = np.transpose(shuffled, (2, 0, 1))
        return chw, indices


class DataLoader:
    """Batches samples from a JigsawDataset. Yields (X_batch, Y_batch) for enumerate(loader, 0)."""

    def __init__(self, dataset: JigsawDataset, batch_size: int = 2048, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            X_list, Y_list = [], []
            for j in batch_idx:
                chw, indices_arr = self.dataset[j]
                X_list.append(chw)
                Y_list.append(indices_arr)
            X_batch = np.stack(X_list, axis=0)
            Y_batch = np.stack(Y_list, axis=0)
            yield X_batch, Y_batch
