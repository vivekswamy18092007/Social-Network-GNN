import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def bce_loss(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return np.mean(loss)


def bce_grad(p, y, eps=1e-7):
    n = p.shape[0]
    return (p - y).reshape(-1, 1) / n


class Linear:
    def __init__(self, in_dim, out_dim, rng):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.gW = self.x.T @ dout
        self.gb = dout.sum(axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        return [(self.W, self.gW), (self.b, self.gb)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def params_and_grads(self):
        return []


def normalize_adj(adj):
    deg = adj.sum(axis=-1)
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D = np.einsum('bi,bij->bij', d_inv_sqrt, np.eye(adj.shape[-1])[None, :, :])
    return np.einsum('bij,bjk,bkl->bil', D, adj, D)


class GCNLayer:
    def __init__(self, in_dim, out_dim, rng, activation=True):
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        self.activation = activation

    def forward(self, H, A_norm):
        self.H = H
        self.A_norm = A_norm
        self.AH = np.einsum('bij,bjf->bif', A_norm, H)
        Z = self.AH @ self.W + self.b
        self.pre_act = Z
        if self.activation:
            self.mask = Z > 0
            return Z * self.mask
        return Z

    def backward(self, dout):
        if self.activation:
            dZ = dout * self.mask
        else:
            dZ = dout
        self.gW = np.einsum('bmf,bmg->fg', self.AH, dZ)
        self.gb = dZ.sum(axis=(0, 1))
        dAH = dZ @ self.W.T
        dH = np.einsum('bij,bif->bjf', self.A_norm, dAH)
        return dH

    def params_and_grads(self):
        return [(self.W, self.gW), (self.b, self.gb)]


class InstanceNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.ggamma = np.zeros(dim)
        self.gbeta = np.zeros(dim)
        self.eps = eps

    def forward(self, H, valid_mask):
        self.mask = valid_mask[:, :, None]
        n = self.mask.sum(axis=1, keepdims=True)
        n = np.maximum(n, 1.0)
        mean = (H * self.mask).sum(axis=1, keepdims=True) / n
        var = ((H - mean) ** 2 * self.mask).sum(axis=1, keepdims=True) / n
        self.xhat = (H - mean) / np.sqrt(var + self.eps)
        self.n = n
        self.var = var
        out = self.gamma * self.xhat + self.beta
        return out * self.mask

    def backward(self, dout):
        dout = dout * self.mask
        self.ggamma = (dout * self.xhat).sum(axis=(0, 1))
        self.gbeta = dout.sum(axis=(0, 1))
        dxhat = dout * self.gamma
        std_inv = 1.0 / np.sqrt(self.var + self.eps)
        dvar = (dxhat * (self.xhat) * -0.5 * std_inv).sum(axis=1, keepdims=True)
        dmean = (-dxhat * std_inv).sum(axis=1, keepdims=True)
        dH = dxhat * std_inv \
            + dvar * 2 * (self.xhat * np.sqrt(self.var + self.eps)) / self.n * self.mask \
            + dmean / self.n * self.mask
        return dH * self.mask

    def params_and_grads(self):
        return [(self.gamma, self.ggamma), (self.beta, self.gbeta)]


class Adam:
    def __init__(self, layers, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        self.layers = layers
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m, self.v = {}, {}
        self.t = 0
        for li, layer in enumerate(layers):
            for pi, (p, _) in enumerate(layer.params_and_grads()):
                key = (li, pi)
                self.m[key] = np.zeros_like(p)
                self.v[key] = np.zeros_like(p)

    def step(self):
        self.t += 1
        for li, layer in enumerate(self.layers):
            pgs = layer.params_and_grads()
            for pi, (p, g) in enumerate(pgs):
                g = g + self.wd * p
                key = (li, pi)
                self.m[key] = self.b1 * self.m[key] + (1 - self.b1) * g
                self.v[key] = self.b2 * self.v[key] + (1 - self.b2) * (g ** 2)
                mhat = self.m[key] / (1 - self.b1 ** self.t)
                vhat = self.v[key] / (1 - self.b2 ** self.t)
                p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
