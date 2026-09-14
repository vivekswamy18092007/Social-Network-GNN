import numpy as np
from .nn import Linear, ReLU, GCNLayer, InstanceNorm, normalize_adj, sigmoid


class GCNEncoder:
    def __init__(self, in_dim, hidden_dim, rng, use_instance_norm=True):
        self.gcn1 = GCNLayer(in_dim, hidden_dim, rng, activation=True)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, rng, activation=False)
        self.use_in = use_instance_norm
        self.inorm = InstanceNorm(hidden_dim) if use_instance_norm else None

    def forward(self, X, adj, valid_mask):
        A_norm = normalize_adj(adj)
        H = self.gcn1.forward(X, A_norm)
        H = self.gcn2.forward(H, A_norm)
        if self.use_in:
            H = self.inorm.forward(H, valid_mask)
        self._last_valid_mask = valid_mask
        return H

    def backward(self, dH):
        if self.use_in:
            dH = self.inorm.backward(dH)
        dH = self.gcn2.backward(dH)
        dH = self.gcn1.backward(dH)
        return dH

    def layers(self):
        ls = [self.gcn1, self.gcn2]
        if self.use_in:
            ls.append(self.inorm)
        return ls


class MLPHead:
    def __init__(self, in_dim, hidden_dim, rng):
        self.l1 = Linear(in_dim, hidden_dim, rng)
        self.act = ReLU()
        self.l2 = Linear(hidden_dim, 1, rng)

    def forward(self, x):
        h = self.act.forward(self.l1.forward(x))
        return self.l2.forward(h)

    def backward(self, dlogit):
        dh = self.l2.backward(dlogit)
        dh = self.act.backward(dh)
        dx = self.l1.backward(dh)
        return dx

    def layers(self):
        return [self.l1, self.l2]


class PlainGNN:
    def __init__(self, node_feat_dim, hidden_dim, rng, use_instance_norm=True):
        self.encoder = GCNEncoder(node_feat_dim, hidden_dim, rng, use_instance_norm)
        self.head = MLPHead(hidden_dim, hidden_dim, rng)

    def forward(self, X, adj, valid_mask, handcrafted=None):
        H = self.encoder.forward(X, adj, valid_mask)
        h_v = H[:, 0, :]
        logit = self.head.forward(h_v)
        self._H_shape = H.shape
        self._h_v_dim = h_v.shape[1]
        return sigmoid(logit).ravel(), logit.ravel()

    def backward(self, dlogit):
        dlogit = dlogit.reshape(-1, 1)
        dh_v = self.head.backward(dlogit)
        dH = np.zeros(self._H_shape)
        dH[:, 0, :] = dh_v
        self.encoder.backward(dH)

    def layers(self):
        return self.encoder.layers() + self.head.layers()


class FullModel:
    def __init__(self, node_feat_dim, handcrafted_dim, hidden_dim, rng, use_instance_norm=True):
        self.encoder = GCNEncoder(node_feat_dim, hidden_dim, rng, use_instance_norm)
        self.head = MLPHead(hidden_dim + handcrafted_dim, hidden_dim, rng)
        self._hc_dim = handcrafted_dim

    def forward(self, X, adj, valid_mask, handcrafted):
        H = self.encoder.forward(X, adj, valid_mask)
        h_v = H[:, 0, :]
        fused = np.concatenate([h_v, handcrafted], axis=1)
        logit = self.head.forward(fused)
        self._H_shape = H.shape
        self._h_v_dim = h_v.shape[1]
        return sigmoid(logit).ravel(), logit.ravel()

    def backward(self, dlogit):
        dlogit = dlogit.reshape(-1, 1)
        dfused = self.head.backward(dlogit)
        dh_v = dfused[:, :self._h_v_dim]
        dH = np.zeros(self._H_shape)
        dH[:, 0, :] = dh_v
        self.encoder.backward(dH)

    def layers(self):
        return self.encoder.layers() + self.head.layers()


def deepwalk_embeddings(G, dim=32, window=5, neg_samples=5):
    import networkx as nx
    nodes = list(G.nodes())
    n = len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    D_inv = np.diag(1.0 / deg)
    P = D_inv @ A
    vol = deg.sum()

    S = np.zeros_like(P)
    Pk = np.eye(n)
    for _ in range(window):
        Pk = Pk @ P
        S += Pk
    S /= window

    M = (vol / neg_samples) * S @ D_inv
    M = np.log(np.maximum(M, 1.0))

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    emb = U[:, :dim] * np.sqrt(s[:dim])
    return {node: emb[i] for i, node in enumerate(nodes)}


class SimpleMLPClassifier:
    def __init__(self, in_dim, hidden_dim, rng):
        self.head = MLPHead(in_dim, hidden_dim, rng)

    def forward(self, X):
        logit = self.head.forward(X)
        return sigmoid(logit).ravel(), logit.ravel()

    def backward(self, dlogit):
        self.head.backward(dlogit.reshape(-1, 1))

    def layers(self):
        return self.head.layers()
