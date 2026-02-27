"""
Predict original patch positions for shuffled 32x32 images.
Usage: python test.py --test-dir <test_dir_path> --model final.pkl
Output: one line per image: image_name.png,p_0,p_1,...,p_15 (row-major; p_i = original index of patch at position i)
"""
import argparse
import os
import pickle

import cv2
import cupy as cp
import numpy as np

from src.cnn.model import JigsawCNN
from src.training.utils import assign_patches, load_state_dict


def main():
    parser = argparse.ArgumentParser(description="Predict patch order for shuffled jigsaw images.")
    parser.add_argument("--test-dir", required=True, help="Directory containing 32x32 shuffled PNG images.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (e.g. final.pkl).")
    args = parser.parse_args()

    if not os.path.isdir(args.test_dir):
        raise SystemExit(f"Test directory not found: {args.test_dir}")
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model file not found: {args.model}")

    with open(args.model, "rb") as f:
        state_dict = pickle.load(f)

    net = JigsawCNN(in_channels=1)
    load_state_dict(net, state_dict)

    image_files = sorted(
        f for f in os.listdir(args.test_dir)
        if f.lower().endswith(".png")
    )

    for filename in image_files:
        path = os.path.join(args.test_dir, filename)
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]
        else:
            image = image[np.newaxis, :, :, :].transpose(0, 3, 1, 2)

        X = cp.asarray(image)
        out = net.forward(X)
        probs = cp.asnumpy(out[0])
        assignment = assign_patches(probs)
        line = filename + "," + ",".join(str(int(assignment[i])) for i in range(16))
        print(line)


if __name__ == "__main__":
    main()
