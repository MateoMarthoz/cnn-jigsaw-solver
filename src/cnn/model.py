"""JigsawCNN: wide 4-block architecture with block1..block4 (Sequential) + fc + Sinkhorn."""
from src.cnn.layers import (
    Conv2D,
    Mish,
    MaxPool2D,
    Linear,
    SinkhornLayer,
    Sequential,
)


class JigsawCNN:
    def __init__(self, in_channels=1):
        self.block1 = Sequential([
            Conv2D(in_channels, 64, kernel_size=3, stride=1, padding=1),
            Mish(),
            Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            Mish(),
            MaxPool2D(pool_size=2, stride=2),
        ])
        self.block2 = Sequential([
            Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
            Mish(),
            Conv2D(128, 128, kernel_size=3, stride=1, padding=1),
            Mish(),
            MaxPool2D(pool_size=2, stride=2),
        ])
        self.block3 = Sequential([
            Conv2D(128, 256, kernel_size=3, stride=1, padding=1),
            Mish(),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
            Mish(),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
            Mish(),
            MaxPool2D(pool_size=2, stride=2),
        ])
        self.block4 = Sequential([
            Conv2D(256, 512, kernel_size=3, stride=1, padding=1),
            Mish(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
            Mish(),
            Conv2D(512, 512, kernel_size=3, stride=1, padding=1),
            Mish(),
        ])
        self.fc = Linear(512 * 4 * 4, 256)
        self.sinkhorn = SinkhornLayer(n_iters=20, temperature=1.0)

    def forward(self, X):
        out = self.block1.forward(X)
        out = self.block2.forward(out)
        out = self.block3.forward(out)
        out = self.block4.forward(out)
        N = out.shape[0]
        out_flat = out.reshape(N, -1)
        fc_out = self.fc.forward(out_flat)
        logits = fc_out.reshape(N, 16, 16)
        return self.sinkhorn.forward(logits)

    def backward(self, dout):
        N = dout.shape[0]
        d_logits = self.sinkhorn.backward(dout)
        d_fc_out = d_logits.reshape(N, 256)
        d_out_flat = self.fc.backward(d_fc_out)
        d_out = d_out_flat.reshape(N, 512, 4, 4)
        d_out = self.block4.backward(d_out)
        d_out = self.block3.backward(d_out)
        d_out = self.block2.backward(d_out)
        d_out = self.block1.backward(d_out)
        return d_out

    def get_parameters(self):
        params = []
        for block in [self.block1, self.block2, self.block3, self.block4]:
            params.extend(block.get_parameters())
        params.append((self.fc.W, self.fc.dW))
        params.append((self.fc.b, self.fc.db))
        return params

    def step(self, learning_rate):
        for param, grad in self.get_parameters():
            if grad is not None:
                param -= learning_rate * grad
