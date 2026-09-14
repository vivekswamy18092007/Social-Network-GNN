import numpy as np
import networkx as nx


def make_graph(n_nodes=1200, m_edges=4, seed=0):
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)
    return G


def simulate_cascade(G, n_seeds=15, p=0.06, max_steps=25, rng=None):
    rng = rng or np.random.default_rng()
    nodes = list(G.nodes())
    seeds = rng.choice(nodes, size=n_seeds, replace=False)

    activation_time = {v: np.inf for v in nodes}
    for s in seeds:
        activation_time[s] = 0

    frontier = set(seeds.tolist())
    t = 0
    while frontier and t < max_steps:
        next_frontier = set()
        for u in frontier:
            for v in G.neighbors(u):
                if activation_time[v] == np.inf:
                    if rng.random() < p:
                        activation_time[v] = t + 1
                        next_frontier.add(v)
        t += 1
        frontier = next_frontier
    return activation_time


def simulate_multi_cascade(G, n_cascades=6, n_seeds=15, p=0.06, max_steps=25, seed=0):
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    best = {v: np.inf for v in nodes}
    for c in range(n_cascades):
        at = simulate_cascade(G, n_seeds=n_seeds, p=p, max_steps=max_steps, rng=rng)
        for v in nodes:
            if at[v] < best[v]:
                best[v] = at[v]
    return best


def sample_ego_network(G, v, k_hops=2, max_size=20, rng=None):
    rng = rng or np.random.default_rng()
    visited = {v}
    order = [v]
    frontier = [v]
    depth = 0
    while frontier and depth < k_hops and len(order) < max_size:
        next_frontier = []
        for u in frontier:
            neighbors = list(G.neighbors(u))
            rng.shuffle(neighbors)
            for w in neighbors:
                if w not in visited:
                    visited.add(w)
                    order.append(w)
                    next_frontier.append(w)
                    if len(order) >= max_size:
                        break
            if len(order) >= max_size:
                break
        frontier = next_frontier
        depth += 1
    return order[:max_size]


def build_instance(G, v, activation_time, k_hops=2, max_size=20, rng=None):
    nodes = sample_ego_network(G, v, k_hops=k_hops, max_size=max_size, rng=rng)
    n_real = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}

    adj = np.zeros((max_size, max_size), dtype=np.float64)
    for i, n in enumerate(nodes):
        for w in G.neighbors(n):
            if w in idx:
                adj[i, idx[w]] = 1.0
    np.fill_diagonal(adj, 1.0)

    t_v = activation_time[v]
    neighbor_state = np.zeros(max_size, dtype=np.float64)
    for i, n in enumerate(nodes):
        if n == v:
            continue
        t_n = activation_time[n]
        horizon = t_v if np.isfinite(t_v) else 1e9
        neighbor_state[i] = 1.0 if t_n < horizon else 0.0

    valid_mask = np.zeros(max_size, dtype=np.float64)
    valid_mask[:n_real] = 1.0

    y = 1.0 if np.isfinite(t_v) else 0.0
    return {
        "nodes": nodes,
        "adj": adj,
        "neighbor_state": neighbor_state,
        "valid_mask": valid_mask,
        "y": y,
    }


def handcrafted_features(G, v, activation_time, neighbor_state_full=None):
    deg = G.degree(v)
    clustering = nx.clustering(G, v)
    neighbors = list(G.neighbors(v))
    t_v = activation_time[v]
    horizon = t_v if np.isfinite(t_v) else 1e9
    active_neighbors = [n for n in neighbors if activation_time[n] < horizon]
    n_active = len(active_neighbors)
    frac_active = n_active / max(deg, 1)
    influence_ratio = n_active / max(deg, 1)
    return {
        "degree": deg,
        "clustering": clustering,
        "n_active_neighbors": n_active,
        "influence_ratio": influence_ratio,
        "frac_active": frac_active,
    }


def make_synthetic_attributes(G, seed=1):
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    age_bucket = rng.integers(0, 5, size=n)
    activity_level = rng.exponential(1.0, size=n)
    historical_freq = rng.beta(2, 5, size=n)
    return {
        v: np.array([age_bucket[v], activity_level[v], historical_freq[v]])
        for v in G.nodes()
    }
