import pickle
import cupy as cp
import numpy as np

from src.training import data
from src.training.utils import (
    assign_patches,
    compute_reconstruction_accuracy,
    get_state_dict
)
from src.cnn import optim
from src.cnn.layers import CrossEntropyLossSinkhorn
from src.cnn.model import JigsawCNN


def compute_validation_accuracy(net, val_loader):
    acc_sum, count = 0.0, 0
    for _, batch in enumerate(val_loader):
        X_batch, Y_batch = batch
        out_np = cp.asnumpy(net.forward(cp.asarray(X_batch)))
        for i in range(X_batch.shape[0]):
            pred = assign_patches(out_np[i])
            acc_sum += compute_reconstruction_accuracy(
                X_batch[i], pred, Y_batch[i].flatten(),
                num_patches=val_loader.dataset.n_patches,
            )
            count += 1
    return acc_sum / count if count > 0 else 0.0


def train(net, trainloader, optimizer, num_epochs, train_sample=1, val_loader=None):
    dataset = trainloader.dataset
    # Total number of training examples
    total_train = (max(1, int(len(dataset) * train_sample)) if train_sample < 1.0 else len(dataset))
    for epoch in range(num_epochs):
        batch_index = 0
        samples_seen = 0
        # Iterate through the training loader in batches
        for i, batch in enumerate(trainloader):
            if train_sample < 1.0 and samples_seen >= total_train:
                break
            X_batch, Y_batch = batch
            # Convert to GPU arrays
            X_gpu = cp.asarray(X_batch) # [batch_size, 1, height, width]
            Y_flat = np.asarray(Y_batch).reshape(Y_batch.shape[0], -1) # [batch_size, total_patches]
            out_batch = net.forward(X_gpu) # [batch_size, total_patches, 16]
            loss_layer = CrossEntropyLossSinkhorn()
            loss_val = loss_layer.forward(out_batch, Y_flat)
            loss_scalar = float(loss_val)
            batch_index += 1
            print(f"Epoch: {epoch},     Iteration: {batch_index},    Loss: {loss_scalar}")
            dX = loss_layer.backward()
            net.backward(dX)
            optimizer.step()
            samples_seen += X_batch.shape[0]
        # Print validation accuracy every 5 epochs
        if epoch % 5 == 0 and val_loader is not None:
            print(f"Epoch: {epoch}      Validation accuracy: {compute_validation_accuracy(net, val_loader)}")

if __name__ == "__main__":
    batch_size = 2048
    num_epochs = 200
    lr = 1e-2
    trainset = data.JigsawDataset(output_dir="datasets", train=True)
    trainloader = data.JigsawDataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = data.JigsawDataset(output_dir="datasets", train=False)
    valloader = data.JigsawDataLoader(valset, batch_size=batch_size, shuffle=False)
    net = JigsawCNN(in_channels=1)
    optimizer = optim.SGD(net, lr)
    train(net, trainloader, optimizer, num_epochs, train_sample=1.0, val_loader=valloader)
    print(f"Validation accuracy: {compute_validation_accuracy(net, valloader)}")
    with open("final.pkl", "wb") as f:
        pickle.dump(get_state_dict(net), f)
    print("Saved final.pkl")
