import numpy as np
import cupy as cp

from src.config import cfg


def im2col(x, kernel_size, stride, padding=0):
    # x: [batch, in_channels, height, width]
    if padding > 0:
        # (pad_start, pad_end) tuple for each dimension
        x = cp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    batch, in_channels, padded_height, padded_width = x.shape
    out_height = (padded_height - kernel_size) // stride + 1
    out_width = (padded_width - kernel_size) // stride + 1
    kernel_row_offsets = cp.tile(cp.repeat(cp.arange(kernel_size, dtype=cp.int32), kernel_size), in_channels)
    output_row_starts = stride * cp.repeat(cp.arange(out_height, dtype=cp.int32), out_width)
    kernel_col_offsets = cp.tile(cp.arange(kernel_size, dtype=cp.int32), kernel_size * in_channels)
    output_col_starts = stride * cp.tile(cp.arange(out_width, dtype=cp.int32), out_height)
    # row_indices: [in_channels * k * k, out_height * out_width]
    row_indices = (kernel_row_offsets.reshape(-1, 1) + output_row_starts.reshape(1, -1)).astype(cp.int32)
    # col_indices: [in_channels * k * k, out_height * out_width]
    col_indices = (kernel_col_offsets.reshape(-1, 1) + output_col_starts.reshape(1, -1)).astype(cp.int32)
    # channel_indices: [in_channels * k * k, 1]
    channel_indices = cp.repeat(cp.arange(in_channels, dtype=cp.int32), kernel_size * kernel_size).reshape(-1, 1)
    columns = x[:, channel_indices, row_indices, col_indices]
    # columns: [in_channels * k * k, batch * out_height * out_width]
    columns = columns.transpose(1, 2, 0).reshape(in_channels * kernel_size * kernel_size, -1)
    return columns


def col2im(col, x_shape, kernel_size, stride, padding=0):
    # col: [in_channels * k * k, batch * out_height * out_width]
    batch, in_channels, height, width = x_shape
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    # col_by_kernel_pos: [in_channels, k, k, out_height, out_width, batch]
    col_by_kernel_pos = col.reshape(in_channels, kernel_size, kernel_size, out_height, out_width, batch)
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    # grad_padded: [batch, in_channels, padded_height, padded_width]
    grad_padded = cp.zeros((batch, in_channels, padded_height, padded_width), dtype=col.dtype)
    for kernel_row in range(kernel_size):
        for kernel_col in range(kernel_size):
            # grad_slice: [batch, in_channels, out_height, out_width]
            grad_slice = col_by_kernel_pos[:, kernel_row, kernel_col, :, :, :].transpose(3, 0, 1, 2)
            grad_padded[:, :, kernel_row : kernel_row + out_height * stride : stride,
                        kernel_col : kernel_col + out_width * stride : stride] += grad_slice
    if padding > 0:
        grad_padded = grad_padded[:, :, padding:-padding, padding:-padding]
    # grad_padded: [batch, in_channels, height, width]
    return grad_padded


class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * kernel_size * kernel_size
        std_dev = float(np.sqrt(2.0 / fan_in))
        W_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std_dev
        self.W = cp.asarray(W_np)
        self.b = cp.zeros((out_channels, 1), dtype=cp.float32)
        self.x_col = None
        self.x_shape = None

    def forward(self, X):
        self.x_shape = X.shape
        batch, channel, height, width = X.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        # self.x_col: [in_channels * k * k, batch * out_height * out_width]
        self.x_col = im2col(X, self.kernel_size, self.stride, self.padding)
        # W_col: [out_channels, in_channels * k * k]
        W_col = self.W.reshape(self.out_channels, -1)
        out = cp.einsum('oi,ix->ox', W_col.astype(cp.float16), self.x_col.astype(cp.float16)).astype(cp.float32) + self.b
        # out: [batch, out_channels, out_height, out_width]
        out = out.reshape(self.out_channels, out_height, out_width, batch).transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        # dout: [batch, out_channels, out_height, out_width]
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        # dout_reshaped: [out_channels, batch * out_height * out_width]
        self.dW = cp.einsum('ol,il->oi', dout_reshaped.astype(cp.float16), self.x_col.astype(cp.float16)).astype(cp.float32).reshape(self.W.shape)
        self.db = cp.einsum('ol->o', dout_reshaped)[:, cp.newaxis]
        W_col = self.W.reshape(self.out_channels, -1)
        # dX_col: [in_channels * k * k, batch * out_height * out_width]
        dX_col = cp.einsum('io,ol->il', W_col.astype(cp.float16), dout_reshaped.astype(cp.float16)).astype(cp.float32)
        dX = col2im(dX_col, self.x_shape, self.kernel_size, self.stride, self.padding)
        return dX

class Mish(Layer):
    def __init__(self):
        self.x = None

    def forward(self, X):
        self.x = X
        x_safe = cp.clip(X, -20.0, 20.0)
        softplus = cp.log1p(cp.exp(x_safe))
        out = X * cp.tanh(softplus)
        out = cp.where(X > 20.0, X, out)
        out = cp.where(X < -20.0, cp.zeros_like(X), out)
        return out

    def backward(self, dout):
        X = self.x
        x_safe = cp.clip(X, -20.0, 20.0)
        exp_x = cp.exp(x_safe)
        exp_2x = cp.exp(2 * x_safe)
        exp_3x = cp.exp(3 * x_safe)
        omega = exp_3x + 4 * exp_2x + (6 + 4 * x_safe) * exp_x + 4 * (1 + x_safe)
        delta = (exp_x + 1) ** 2 + 1
        derivative = (exp_x * omega) / (delta ** 2)
        derivative = cp.where(X > 20.0, cp.ones_like(X), derivative)
        derivative = cp.where(X < -20.0, cp.zeros_like(X), derivative)
        return dout * derivative

class SinkhornLayer(Layer):
    def __init__(self, n_iters=20, temperature=1.0):
        self.n_iters = n_iters
        self.temperature = temperature
        self.log_alpha = None
        self.out = None

    def forward(self, logits):
        # logits: [batch, n_tiles, n_tiles]
        self.log_alpha = logits / self.temperature
        for _ in range(self.n_iters):
            max_row = cp.max(self.log_alpha, axis=2, keepdims=True)
            row_exp = cp.exp(self.log_alpha - max_row)
            # row_lse: [batch, n_tiles, 1]
            row_lse = cp.log(cp.einsum('nij->ni', row_exp)[:, :, cp.newaxis] + 1e-12) + max_row
            self.log_alpha = self.log_alpha - row_lse
            max_col = cp.max(self.log_alpha, axis=1, keepdims=True)
            col_exp = cp.exp(self.log_alpha - max_col)
            # col_lse: [batch, 1, n_tiles]
            col_lse = cp.log(cp.einsum('nij->nj', col_exp)[:, cp.newaxis, :] + 1e-12) + max_col
            self.log_alpha = self.log_alpha - col_lse
        # self.out: [batch, n_tiles, n_tiles]
        self.out = cp.exp(self.log_alpha)
        return self.out

    def backward(self, dout):
        # dout: [batch, n_tiles, n_tiles]
        # row_sum: [batch, n_tiles, 1]
        row_sum = cp.einsum('nij,nij->ni', self.out, dout)[:, :, cp.newaxis]
        return self.out * (dout - row_sum) / self.temperature

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x_shape = None
        self.x_col = None
        self.max_idx = None

    def forward(self, X):
        # X: [batch, channel, height, width]
        self.x_shape = X.shape
        batch, channel, height, width = X.shape
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        # x_reshaped: [batch, channel, out_height, pool_size, out_width, pool_size]
        x_reshaped = X.reshape(batch, channel, out_height, self.pool_size, out_width, self.pool_size)
        # out: [batch, channel, out_height, out_width, pool_size * pool_size]
        out = x_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(batch, channel, out_height, out_width, self.pool_size * self.pool_size)
        self.x_col = out
        # self.max_idx: [batch, channel, out_height, out_width]
        self.max_idx = cp.argmax(out, axis=4)
        return cp.max(out, axis=4)

    def backward(self, dout):
        # dout: [batch, channel, out_height, out_width]
        n_batch, n_channels, out_height, out_width = dout.shape
        K = self.pool_size
        dX_col = cp.zeros_like(self.x_col)
        dout_flat = dout.flatten()
        max_idx_flat = self.max_idx.flatten()
        n_elements = dout_flat.size
        idx_offset = cp.arange(n_elements, dtype=cp.int32) * (K * K)
        abs_idx = max_idx_flat.astype(cp.int32) + idx_offset
        dX_col_flat = dX_col.flatten()
        dX_col_flat[abs_idx] = dout_flat
        dX_col = dX_col_flat.reshape(self.x_col.shape)
        # dX_col_reshaped: [channel, K, K, batch, out_height, out_width]
        dX_col_reshaped = dX_col.reshape(n_batch, n_channels, out_height, out_width, K, K).transpose(1, 4, 5, 0, 2, 3)
        dX = cp.zeros(self.x_shape, dtype=dout.dtype)
        for ky in range(K):
            for kx in range(K):
                # block: [batch, channel, out_height, out_width]
                block = dX_col_reshaped[:, ky, kx, :, :, :].transpose(1, 0, 2, 3)
                dX[:, :, ky : ky + out_height * self.stride : self.stride, kx : kx + out_width * self.stride : self.stride] += block
        return dX

class Linear(Layer):
    def __init__(self, in_features, out_features):
        std = float(np.sqrt(2.0 / in_features))
        # W_np: [in_features, out_features]
        W_np = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.W = cp.asarray(W_np)
        self.b = cp.zeros((1, out_features), dtype=cp.float32)
        self.x = None

    def forward(self, X):
        # X: [batch, in_features]
        self.x = X
        return cp.einsum('ni,io->no', X, self.W) + self.b

    def backward(self, dout):
        # dout: [batch, out_features]
        self.dW = cp.einsum('in,no->io', self.x, dout)
        self.db = cp.einsum('no->o', dout)[cp.newaxis, :]
        return cp.einsum('no,oi->ni', dout, self.W)

class Sequential(Layer):
    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "W"):
                params.append((layer.W, layer.dW))
            if hasattr(layer, "b"):
                params.append((layer.b, layer.db))
        return params

class CrossEntropyLossSinkhorn(Layer):
    def forward(self, X, Y):
        # X: [n_batch, n_tiles, n_tiles]
        n_batch = X.shape[0]
        n_tiles = cfg.n_tiles
        self.x = X
        # Y_flat: [n_batch, n_tiles]
        Y_flat = cp.asarray(Y).astype(cp.int32)
        if Y_flat.ndim != 2 or Y_flat.shape[1] != n_tiles:
            Y_flat = Y_flat.reshape(Y_flat.shape[0], -1)
        # self.target_one_hot: [n_batch, n_tiles, n_tiles]
        self.target_one_hot = cp.zeros_like(X, dtype=cp.float32)
        n_idx = cp.arange(n_batch, dtype=cp.int32)[:, None]
        i_idx = cp.arange(n_tiles, dtype=cp.int32)[None, :]
        self.target_one_hot[n_idx, i_idx, Y_flat] = 1.0
        log_probs = cp.log(cp.clip(X, 1e-12, 1.0))
        return -cp.einsum('nij,nij->', self.target_one_hot, log_probs) / (n_batch * n_tiles)

    def backward(self, dout=None):
        n_batch = self.x.shape[0]
        n_tiles = cfg.n_tiles
        # dX: [n_batch, n_tiles, n_tiles]
        dX = -self.target_one_hot / (self.x + 1e-12)
        dX /= (n_batch * n_tiles)
        return dX
