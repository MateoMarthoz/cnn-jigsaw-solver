"""Training loop and entry point. train() lives here; run from cnn-jigsaw-solver."""
import pickle

import cupy as cp
import numpy as np

from src.training.data import JigsawDataset, DataLoader
from src.training.utils import (
    assign_patches,
    compute_reconstruction_accuracy,
    get_state_dict,
    load_state_dict,
)
from src.cnn.layers import CrossEntropyLossSinkhorn
from src.cnn.model import JigsawCNN


def train(
    net,
    trainloader,
    learning_rate,
    num_epochs,
    train_sample: float = 0.2,
    val_loader=None,
):
    dataset = trainloader.dataset
    num_patches = dataset.num_patches
    total_train = (
        max(1, int(len(dataset) * train_sample))
        if train_sample < 1.0
        else len(dataset)
    )

    for epoch in range(num_epochs):
        iter_train = 0
        samples_seen = 0
        for i, data in enumerate(trainloader, 0):
            if train_sample < 1.0 and samples_seen >= total_train:
                break
            X_batch, Y_batch = data
            X_gpu = cp.asarray(X_batch)
            Y_flat = np.asarray(Y_batch).reshape(Y_batch.shape[0], -1)

            out_batch = net.forward(X_gpu)
            loss_layer = CrossEntropyLossSinkhorn()
            loss_val = loss_layer.forward(out_batch, Y_flat)
            loss_scalar = float(loss_val)
            iter_train += 1
            print(
                f"[ INFO ] Mode: Training,    Epoch: {epoch},    Iteration: {iter_train},    Loss: {loss_scalar}"
            )

            dX = loss_layer.backward()
            net.backward(dX)
            net.step(learning_rate)
            samples_seen += X_batch.shape[0]

        if epoch % 5 == 0 and val_loader is not None:
            val_acc_sum = 0.0
            val_count = 0
            for _, data in enumerate(val_loader, 0):
                X_batch, Y_batch = data
                X_gpu = cp.asarray(X_batch)
                out_batch = net.forward(X_gpu)
                out_np = cp.asnumpy(out_batch)
                for j in range(X_batch.shape[0]):
                    probs = out_np[j]
                    pred = assign_patches(probs)
                    gt = Y_batch[j].flatten()
                    im = X_batch[j]
                    val_acc_sum += compute_reconstruction_accuracy(
                        im, pred, gt, num_patches=val_loader.dataset.num_patches
                    )
                    val_count += 1
            val_accuracy = val_acc_sum / val_count if val_count > 0 else 0.0
            print(f"[ INFO ] Epoch {epoch} — Validation accuracy: {val_accuracy}")


if __name__ == "__main__":
    batch_size = 2048
    num_epochs = 200
    learning_rate = 1e-3 * (2048 / 32)
    train_sample = 1.0

    trainset = JigsawDataset(dataset_dir="datasets", train=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = JigsawDataset(dataset_dir="datasets", train=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    net = JigsawCNN(in_channels=1)
    train(
        net,
        trainloader,
        learning_rate,
        num_epochs,
        train_sample=train_sample,
        val_loader=valloader,
    )

    val_acc_sum = 0.0
    val_count = 0
    for _, data in enumerate(valloader, 0):
        X_batch, Y_batch = data
        X_gpu = cp.asarray(X_batch)
        out_batch = net.forward(X_gpu)
        out_np = cp.asnumpy(out_batch)
        for i in range(X_batch.shape[0]):
            probs = out_np[i]
            pred = assign_patches(probs)
            gt = Y_batch[i].flatten()
            im = X_batch[i]
            val_acc_sum += compute_reconstruction_accuracy(
                im, pred, gt, num_patches=valloader.dataset.num_patches
            )
            val_count += 1
    val_accuracy = val_acc_sum / val_count if val_count > 0 else 0.0
    print(f"[ INFO ] Validation accuracy: {val_accuracy}")

    state_dict = get_state_dict(net)
    with open("improve_final.pkl", "wb") as f:
        pickle.dump(state_dict, f)
    print("Saved improve_final.pkl")
