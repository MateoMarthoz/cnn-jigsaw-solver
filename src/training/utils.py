"""assign_patches, compute_reconstruction_accuracy, get_state_dict, load_state_dict."""
import numpy as np
import cupy as cp
import scipy.optimize


def _get_param_layers(net):
    """Collect all layers with W/b from block1..block4 and fc."""
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
    if hasattr(probs, "get"):
        probs = np.asarray(probs)
    if probs.shape != (16, 16):
        probs = np.asarray(probs).reshape(16, 16)
    cost = -probs
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    assignment = np.zeros(16, dtype=int)
    assignment[row_ind] = col_ind
    return assignment


def compute_reconstruction_accuracy(im, pred, gt, num_patches: int = 4, eps: float = 0.05):
    C, H, W = im.shape
    assert C == 1, "Only grayscale images supported."
    assert pred.shape == gt.shape
    assert len(pred.shape) == 1
    ph = pw = H // num_patches
    im_2d = np.squeeze(im)
    num_correct = num_total = 0
    i = -1
    for y in range(0, H, ph):
        for x in range(0, W, pw):
            i += 1
            if np.all(im_2d[y : y + ph, x : x + pw] < eps):
                continue
            num_total += 1
            if pred[i] == gt[i]:
                num_correct += 1
    return num_correct / num_total if num_total > 0 else 0.0
