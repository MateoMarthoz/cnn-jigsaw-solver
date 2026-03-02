import os
import json
import random
import cv2
import numpy as np

from src.config import cfg

INPUT_DIR = os.path.join("datasets", "acv_train_32x32")

def _list_image_files(input_dir):
    supported = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")
    if not os.path.exists(input_dir):
        return []
    return sorted(
        filename for filename in os.listdir(input_dir)
        if filename.lower().endswith(supported)
    )

def _create_splits(input_dir, splits_dir, shuffle=True, seed=42, train_ratio=0.8):
    image_files = _list_image_files(input_dir)
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}.")
    if shuffle:
        random.seed(seed)
        random.shuffle(image_files)
    n_images = len(image_files)
    n_train = int(n_images * train_ratio)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]
    split_data = {"train": train_files, "val": val_files}
    os.makedirs(splits_dir, exist_ok=True)
    json_path = os.path.join(splits_dir, "split.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=4)
    print(f"Saved {json_path} ({len(train_files)} train, {len(val_files)} val)")


class JigsawDataset:
    d_image = cfg.d_image
    n_patches = cfg.n_patches
    d_patch = cfg.d_patch

    def __init__(self, output_dir, train=True):
        self.output_dir = output_dir.rstrip(os.sep)
        self.is_train = train
        splits_dir = self.output_dir
        json_path = os.path.join(splits_dir, "split.json")
        if not os.path.exists(json_path):
            _create_splits(INPUT_DIR, splits_dir, shuffle=True, seed=42, train_ratio=0.8)
        with open(json_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        self.file_list = split_data["train"] if train else split_data["val"]
        self.input_dir = INPUT_DIR
        # Preloaded images array
        self._images = None

    def _do_preload(self):
        n_images = len(self.file_list)
        height, width = self.d_image[0], self.d_image[1]
        preloaded_images = np.zeros((n_images, height, width, 1), dtype=np.float32)
        for i, filename in enumerate(self.file_list):
            img_path = os.path.join(self.input_dir, filename)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load {img_path}.")
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize to 32x32
            image = cv2.resize(image, self.d_image)
            # Add channel dimension: [height, width, 1]
            if image.ndim == 2:
                image = image[:, :, None]
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            preloaded_images[i] = image
        return preloaded_images

    def _ensure_preloaded(self):
        if self._images is None:
            self._images = self._do_preload()

    def _shuffle_patches(self, image):
        patch_height, patch_width = self.d_patch[0], self.d_patch[1]
        # Number of patches in each row and column
        n_rows = n_cols = self.n_patches
        total_patches = n_rows * n_cols
        # patches: list of [patch_height, patch_width, 1]
        patches = []
        for i in range(total_patches):
            # Floor division finds the row by counting completed rows; modulo finds the column as the remainder
            row, col = i // n_cols, i % n_cols
            # Calculate the pixel boundaries for the current patch
            x_start, x_end = col * patch_width, (col + 1) * patch_width
            y_start, y_end = row * patch_height, (row + 1) * patch_height
            # Extract the patch using the calculated coordinates
            patch = image[y_start:y_end, x_start:x_end, :]
            patches.append(patch)
        # Stack list of patches into: [total_patches, patch_height, patch_width, 1]
        patches = np.array(patches)
        # Create array from 0 to 15 and shuffle
        shuffle_map = np.arange(total_patches)
        np.random.shuffle(shuffle_map)
        # Initialize an empty array of the to store the reassembled image
        shuffled_image = np.zeros_like(image)
        for idx, patch_idx in enumerate(shuffle_map):
            # Map the current iteration index to 2D grid coordinates
            row, col = idx // n_cols, idx % n_cols
            # Define the destination pixel boundaries in the reconstructed image
            x_start, x_end = col * patch_width, (col + 1) * patch_width
            y_start, y_end = row * patch_height, (row + 1) * patch_height
            # Place the specific shuffled patch into its new grid position
            shuffled_image[y_start:y_end, x_start:x_end, :] = patches[patch_idx]
        return shuffled_image, shuffle_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        self._ensure_preloaded()
        image = self._images[idx]
        # shuffled_image: [height, width, 1]
        shuffled_image, shuffle_map = self._shuffle_patches(image)
        # Permute dimensions to: [1, height, width]
        chw = np.transpose(shuffled_image, (2, 0, 1))
        return chw, shuffle_map


class JigsawDataLoader:
    def __init__(self, dataset, batch_size=2048, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        all_indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(all_indices)
        # Iterate through all_indices in chunks of size batch_size
        for start in range(0, len(all_indices), self.batch_size):
            batch_indices = all_indices[start : start + self.batch_size]
            X_list, Y_list = [], []
            for j in batch_indices:
                chw, shuffle_map = self.dataset[j]
                X_list.append(chw)
                Y_list.append(shuffle_map)
            # X_batch: [batch_size, 1, height, width]
            X_batch = np.stack(X_list, axis=0)
            # Y_batch: [batch_size, total_patches]
            Y_batch = np.stack(Y_list, axis=0)
            yield X_batch, Y_batch
