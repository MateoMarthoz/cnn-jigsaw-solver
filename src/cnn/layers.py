"""Layer classes: Conv2D, Mish, MaxPool2D, Linear, SinkhornLayer, Sequential, CrossEntropyLossSinkhorn."""
import numpy as np
import cupy as cp


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
        self.X_pad = None
        self.X_col = None
        self.X_shape = None

    def get_im2col_indices(self, x_shape):
        N, C, H, W = x_shape
        out_height = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        i0 = cp.repeat(cp.arange(self.kernel_size, dtype=cp.int32), self.kernel_size)
        i0 = cp.tile(i0, C)
        i1 = self.stride * cp.repeat(cp.arange(out_height, dtype=cp.int32), out_width)
        j0 = cp.tile(cp.arange(self.kernel_size, dtype=cp.int32), self.kernel_size * C)
        j1 = self.stride * cp.tile(cp.arange(out_width, dtype=cp.int32), out_height)
        i = (i0.reshape(-1, 1) + i1.reshape(1, -1)).astype(cp.int32)
        j = (j0.reshape(-1, 1) + j1.reshape(1, -1)).astype(cp.int32)
        k = cp.repeat(cp.arange(C, dtype=cp.int32), self.kernel_size * self.kernel_size).reshape(-1, 1)
        return k, i, j

    def forward(self, X):
        self.X_shape = X.shape
        N, C, H, W = X.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        self.X_pad = cp.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode="constant")
        k, i, j = self.get_im2col_indices(X.shape)
        X_col = self.X_pad[:, k, i, j]
        self.X_col = X_col.transpose(1, 2, 0).reshape(C * self.kernel_size * self.kernel_size, -1)
        W_col = self.W.reshape(self.out_channels, -1)
        out = cp.matmul(W_col.astype(cp.float16), self.X_col.astype(cp.float16)).astype(cp.float32) + self.b
        out = out.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)
        return out

    def backward(self, dout):
        N, C, H, W = self.X_shape
        K = self.kernel_size
        out_h = (H + 2 * self.padding - K) // self.stride + 1
        out_w = (W + 2 * self.padding - K) // self.stride + 1
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.dW = cp.matmul(dout_reshaped.astype(cp.float16), self.X_col.T.astype(cp.float16)).astype(cp.float32).reshape(self.W.shape)
        self.db = cp.sum(dout_reshaped, axis=1, keepdims=True)
        W_col = self.W.reshape(self.out_channels, -1)
        dX_col = cp.matmul(W_col.T.astype(cp.float16), dout_reshaped.astype(cp.float16)).astype(cp.float32)
        dX_pad = cp.zeros_like(self.X_pad)
        dX_col_reshaped = dX_col.reshape(C, K, K, out_h * out_w * N).reshape(C, K, K, out_h, out_w, N)
        for ky in range(K):
            for kx in range(K):
                block = dX_col_reshaped[:, ky, kx, :, :, :].transpose(3, 0, 1, 2)
                dX_pad[:, :, ky : ky + out_h * self.stride : self.stride, kx : kx + out_w * self.stride : self.stride] += block
        dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding > 0 else dX_pad
        return dX


class Mish(Layer):
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        X_safe = cp.clip(X, -20.0, 20.0)
        softplus = cp.log1p(cp.exp(X_safe))
        out = X * cp.tanh(softplus)
        out = cp.where(X > 20.0, X, out)
        out = cp.where(X < -20.0, cp.zeros_like(X), out)
        return out

    def backward(self, dout):
        X = self.X
        X_safe = cp.clip(X, -20.0, 20.0)
        exp_x = cp.exp(X_safe)
        exp_2x = cp.exp(2 * X_safe)
        exp_3x = cp.exp(3 * X_safe)
        omega = exp_3x + 4 * exp_2x + (6 + 4 * X_safe) * exp_x + 4 * (1 + X_safe)
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
        self.log_alpha = logits / self.temperature
        for _ in range(self.n_iters):
            max_row = cp.max(self.log_alpha, axis=2, keepdims=True)
            row_lse = cp.log(cp.sum(cp.exp(self.log_alpha - max_row), axis=2, keepdims=True) + 1e-12) + max_row
            self.log_alpha = self.log_alpha - row_lse
            max_col = cp.max(self.log_alpha, axis=1, keepdims=True)
            col_lse = cp.log(cp.sum(cp.exp(self.log_alpha - max_col), axis=1, keepdims=True) + 1e-12) + max_col
            self.log_alpha = self.log_alpha - col_lse
        self.out = cp.exp(self.log_alpha)
        return self.out

    def backward(self, dout):
        row_sum = cp.sum(self.out * dout, axis=2, keepdims=True)
        return self.out * (dout - row_sum) / self.temperature


class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.X_shape = None
        self.X_col = None
        self.max_idx = None

    def forward(self, X):
        self.X_shape = X.shape
        N, C, H, W = X.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        X_reshaped = X.reshape(N, C, out_h, self.pool_size, out_w, self.pool_size)
        out = X_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, out_h, out_w, self.pool_size * self.pool_size)
        self.X_col = out
        self.max_idx = cp.argmax(out, axis=4)
        return cp.max(out, axis=4)

    def backward(self, dout):
        N, C, out_h, out_w = dout.shape
        K = self.pool_size
        dX_col = cp.zeros_like(self.X_col)
        dout_flat = dout.flatten()
        max_idx_flat = self.max_idx.flatten()
        n_elements = dout_flat.size
        idx_offset = cp.arange(n_elements, dtype=cp.int32) * (K * K)
        abs_idx = max_idx_flat.astype(cp.int32) + idx_offset
        dX_col_flat = dX_col.flatten()
        dX_col_flat[abs_idx] = dout_flat
        dX_col = dX_col_flat.reshape(self.X_col.shape)
        dX_col_reshaped = dX_col.reshape(N, C, out_h, out_w, K, K).transpose(1, 4, 5, 0, 2, 3)
        dX = cp.zeros(self.X_shape, dtype=dout.dtype)
        for ky in range(K):
            for kx in range(K):
                block = dX_col_reshaped[:, ky, kx, :, :, :].transpose(1, 0, 2, 3)
                dX[:, :, ky : ky + out_h * self.stride : self.stride, kx : kx + out_w * self.stride : self.stride] += block
        return dX


class Linear(Layer):
    def __init__(self, in_features, out_features):
        std = float(np.sqrt(2.0 / in_features))
        W_np = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.W = cp.asarray(W_np)
        self.b = cp.zeros((1, out_features), dtype=cp.float32)
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W) + self.b

    def backward(self, dout):
        self.dW = self.X.T.dot(dout)
        self.db = cp.sum(dout, axis=0, keepdims=True)
        return dout.dot(self.W.T)


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
        N = X.shape[0]
        self.X = X
        Y_flat = cp.asarray(Y).astype(cp.int32)
        if Y_flat.ndim != 2 or Y_flat.shape[1] != 16:
            Y_flat = Y_flat.reshape(Y_flat.shape[0], -1)
        self.target_one_hot = cp.zeros_like(X, dtype=cp.float32)
        n_idx = cp.arange(N, dtype=cp.int32)[:, None]
        i_idx = cp.arange(16, dtype=cp.int32)[None, :]
        self.target_one_hot[n_idx, i_idx, Y_flat] = 1.0
        log_probs = cp.log(cp.clip(X, 1e-12, 1.0))
        return -cp.sum(self.target_one_hot * log_probs) / (N * 16)

    def backward(self):
        N = self.X.shape[0]
        dX = -self.target_one_hot / (self.X + 1e-12)
        dX /= (N * 16)
        return dX
