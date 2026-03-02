import numpy as np
import cupy as cp
import scipy.optimize

from src.config import cfg

def _get_param_layers(net):
    out = []
    for block in [net.block1, net.block2, net.block3, net.block4]:
        for layer in block.layers:
            if hasattr(layer, "W") and hasattr(layer, "b"):
                out.append(layer)
    if hasattr(net, "fc"):
        out.append(net.fc)
    return out

def get_state_dict(net):
    state = {}
    for i, layer in enumerate(_get_param_layers(net)):
        state[f"layer_{i}_W"] = cp.asnumpy(layer.W)
        state[f"layer_{i}_b"] = cp.asnumpy(layer.b)
    return state

def load_state_dict(net, state_dict):
    param_layers = _get_param_layers(net)
    for i, layer in enumerate(param_layers):
        key_w, key_b = f"layer_{i}_W", f"layer_{i}_b"
        if key_w in state_dict:
            layer.W = cp.asarray(state_dict[key_w])
        if key_b in state_dict:
            layer.b = cp.asarray(state_dict[key_b])

def assign_patches(probs):
    n_tiles = cfg.n_tiles
    if hasattr(probs, "get"):
        probs = np.asarray(probs)
    if probs.shape != (n_tiles, n_tiles):
        probs = np.asarray(probs).reshape(n_tiles, n_tiles)
    cost = -probs
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    assignment = np.zeros(n_tiles, dtype=int)
    assignment[row_ind] = col_ind
    return assignment

def compute_reconstruction_accuracy(im, pred, gt, num_patches=None, eps=0.05):
    if num_patches is None:
        num_patches = cfg.num_patches
    C, H, W = im.shape
    if C != 1:
        raise ValueError("Only grayscale images supported.")
    if pred.shape != gt.shape:
        raise ValueError("pred and gt must have the same shape.")
    if len(pred.shape) != 1:
        raise ValueError("pred must be 1-dimensional.")
    ph = pw = H // num_patches
    im_2d = np.squeeze(im)
    n_correct = n_total = 0
    i = -1
    for y in range(0, H, ph):
        for x in range(0, W, pw):
            i += 1
            if np.all(im_2d[y : y + ph, x : x + pw] < eps):
                continue
            n_total += 1
            if pred[i] == gt[i]:
                n_correct += 1
    return n_correct / n_total if n_total > 0 else 0.0
