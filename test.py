import argparse
import os
import pickle
import cv2
import cupy as cp
import numpy as np

from src.config import cfg
from src.cnn.model import JigsawCNN
from src.training.utils import assign_patches, load_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model", required=True)
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
        filename for filename in os.listdir(args.test_dir)
        if filename.lower().endswith(".png")
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
        line = filename + "," + ",".join(str(int(assignment[i])) for i in range(cfg.n_tiles))
        print(line)

if __name__ == "__main__":
    main()
