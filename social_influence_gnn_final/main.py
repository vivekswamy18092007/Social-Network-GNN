import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import (make_graph, simulate_multi_cascade, build_instance,
                       handcrafted_features, make_synthetic_attributes)
from src.models import PlainGNN, FullModel, SimpleMLPClassifier, deepwalk_embeddings
from src.train import train_gnn_model, train_mlp_model, evaluate

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

t0 = time.time()
print("=" * 70)
print("1. Building synthetic social graph + simulating action cascades")
print("=" * 70)
G = make_graph(n_nodes=1200, m_edges=4, seed=RNG_SEED)
activation_time = simulate_multi_cascade(G, n_cascades=6, n_seeds=15, p=0.06,
                                          max_steps=25, seed=RNG_SEED)
n_active = sum(1 for t in activation_time.values() if np.isfinite(t))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Positive rate (fraction of users who acted): {n_active/G.number_of_nodes():.3f}")

attrs = make_synthetic_attributes(G, seed=1)
max_deg = max(dict(G.degree()).values())


def build_dataset(k_hops=2, max_size=20, node_ids=None):
    node_ids = node_ids if node_ids is not None else list(G.nodes())
    instances = []
    for v in node_ids:
        inst = build_instance(G, v, activation_time, k_hops=k_hops, max_size=max_size, rng=rng)
        hc = handcrafted_features(G, v, activation_time)
        a = attrs[v]
        hc_vec = np.array([hc["degree"] / max_deg, hc["clustering"], hc["n_active_neighbors"],
                            hc["influence_ratio"], hc["frac_active"], a[0] / 4.0, a[1], a[2]])
        deg_arr = np.array([G.degree(n) / max_deg for n in inst["nodes"]] +
                            [0.0] * (max_size - len(inst["nodes"])))
        is_ego = np.zeros(max_size); is_ego[0] = 1.0
        X = np.stack([inst["neighbor_state"], deg_arr, is_ego], axis=1)
        instances.append({
            "v": v, "X": X, "adj": inst["adj"], "valid_mask": inst["valid_mask"],
            "handcrafted": hc_vec, "y": inst["y"],
        })
    return instances


def stack(instances, hc_scaler=None, fit_scaler=False):
    X = np.stack([i["X"] for i in instances])
    adj = np.stack([i["adj"] for i in instances])
    vm = np.stack([i["valid_mask"] for i in instances])
    hc = np.stack([i["handcrafted"] for i in instances])
    y = np.array([i["y"] for i in instances])
    if hc_scaler is not None:
        if fit_scaler:
            hc = hc_scaler.fit_transform(hc)
        else:
            hc = hc_scaler.transform(hc)
    return {"X": X, "adj": adj, "valid_mask": vm, "handcrafted": hc, "y": y}


print("\n" + "=" * 70)
print("2. Ego-network sampling (k=2 hops, max 20 nodes) for every user")
print("=" * 70)
all_nodes = list(G.nodes())
rng.shuffle(all_nodes)
n = len(all_nodes)
n_train, n_val = int(0.7 * n), int(0.15 * n)
train_ids = all_nodes[:n_train]
val_ids = all_nodes[n_train:n_train + n_val]
test_ids = all_nodes[n_train + n_val:]
print(f"train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

histories = {}

instances_all = {"train": build_dataset(node_ids=train_ids),
                  "val": build_dataset(node_ids=val_ids),
                  "test": build_dataset(node_ids=test_ids)}

scaler = StandardScaler()
data = {
    "train": stack(instances_all["train"], scaler, fit_scaler=True),
    "val": stack(instances_all["val"], scaler),
    "test": stack(instances_all["test"], scaler),
}

results = {}

print("\n" + "=" * 70)
print("3. Baseline 1: Logistic Regression on handcrafted features only")
print("=" * 70)
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(data["train"]["handcrafted"], data["train"]["y"])
p_test = lr_clf.predict_proba(data["test"]["handcrafted"])[:, 1]
results["logistic_regression"] = evaluate(data["test"]["y"], p_test)
print(" test:", results["logistic_regression"])

print("\n" + "=" * 70)
print("4. Baseline 2: DeepWalk-style (NetMF) embeddings + MLP")
print("=" * 70)
emb = deepwalk_embeddings(G, dim=32, window=5)
def emb_matrix(ids):
    return np.stack([emb[v] for v in ids])
Xtr_dw, Xval_dw, Xtest_dw = emb_matrix(train_ids), emb_matrix(val_ids), emb_matrix(test_ids)
ytr = data["train"]["y"]; yval = data["val"]["y"]; ytest = data["test"]["y"]
dw_model = SimpleMLPClassifier(in_dim=32, hidden_dim=16, rng=rng)
histories["deepwalk_mlp"] = train_mlp_model(
    dw_model, Xtr_dw, ytr, Xval_dw, yval, epochs=80, lr=0.02, tag="DeepWalk+MLP")
p_test, _ = dw_model.forward(Xtest_dw)
results["deepwalk_mlp"] = evaluate(ytest, p_test)
print(" test:", results["deepwalk_mlp"])

print("\n" + "=" * 70)
print("5. Baseline 3: Plain GCN (graph embedding only, no handcrafted fusion)")
print("=" * 70)
plain_model = PlainGNN(node_feat_dim=3, hidden_dim=16, rng=rng, use_instance_norm=True)
histories["plain_gcn"] = train_gnn_model(
    plain_model, data["train"], data["val"], epochs=60, lr=0.02, tag="PlainGCN")
p_test, _ = plain_model.forward(data["test"]["X"], data["test"]["adj"],
                                 data["test"]["valid_mask"], data["test"]["handcrafted"])
results["plain_gcn"] = evaluate(data["test"]["y"], p_test)
print(" test:", results["plain_gcn"])

print("\n" + "=" * 70)
print("6. Full model: GCN embedding + handcrafted feature fusion (proposed)")
print("=" * 70)
full_model = FullModel(node_feat_dim=3, handcrafted_dim=8, hidden_dim=16, rng=rng,
                        use_instance_norm=True)
histories["full_model"] = train_gnn_model(
    full_model, data["train"], data["val"], epochs=60, lr=0.02, tag="FullModel")
p_test, _ = full_model.forward(data["test"]["X"], data["test"]["adj"],
                                data["test"]["valid_mask"], data["test"]["handcrafted"])
results["full_model"] = evaluate(data["test"]["y"], p_test)
print(" test:", results["full_model"])

print("\n" + "=" * 70)
print("7. Ablation: Full model WITHOUT instance normalization")
print("=" * 70)
full_noin = FullModel(node_feat_dim=3, handcrafted_dim=8, hidden_dim=16, rng=rng,
                       use_instance_norm=False)
histories["full_model_no_instance_norm"] = train_gnn_model(
    full_noin, data["train"], data["val"], epochs=60, lr=0.02, tag="Full-noIN")
p_test, _ = full_noin.forward(data["test"]["X"], data["test"]["adj"],
                               data["test"]["valid_mask"], data["test"]["handcrafted"])
results["full_model_no_instance_norm"] = evaluate(data["test"]["y"], p_test)
print(" test:", results["full_model_no_instance_norm"])

print("\n" + "=" * 70)
print("8. Ablation: sensitivity to ego-network radius k (k=1,2,3)")
print("=" * 70)
k_results = {}
for k in [1, 2, 3]:
    inst_tr = build_dataset(k_hops=k, node_ids=train_ids)
    inst_va = build_dataset(k_hops=k, node_ids=val_ids)
    inst_te = build_dataset(k_hops=k, node_ids=test_ids)
    sc = StandardScaler()
    d_tr = stack(inst_tr, sc, fit_scaler=True)
    d_va = stack(inst_va, sc)
    d_te = stack(inst_te, sc)
    m = FullModel(node_feat_dim=3, handcrafted_dim=8, hidden_dim=16, rng=rng, use_instance_norm=True)
    train_gnn_model(m, d_tr, d_va, epochs=50, lr=0.02, verbose=False, tag=f"k={k}")
    p_te, _ = m.forward(d_te["X"], d_te["adj"], d_te["valid_mask"], d_te["handcrafted"])
    k_results[f"k={k}"] = evaluate(d_te["y"], p_te)
    print(f"  k={k}: {k_results[f'k={k}']}")
results["k_hop_sensitivity"] = k_results

print("\n" + "=" * 70)
print(f"Done in {time.time()-t0:.1f}s. Saving results + weights.")
print("=" * 70)

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

np.savez("outputs/full_model_weights.npz",
         gcn1_W=full_model.encoder.gcn1.W, gcn1_b=full_model.encoder.gcn1.b,
         gcn2_W=full_model.encoder.gcn2.W, gcn2_b=full_model.encoder.gcn2.b,
         in_gamma=full_model.encoder.inorm.gamma, in_beta=full_model.encoder.inorm.beta,
         head_l1_W=full_model.head.l1.W, head_l1_b=full_model.head.l1.b,
         head_l2_W=full_model.head.l2.W, head_l2_b=full_model.head.l2.b)

rows = [("Logistic Regression", results["logistic_regression"]),
        ("DeepWalk + MLP", results["deepwalk_mlp"]),
        ("Plain GCN", results["plain_gcn"]),
        ("Full model (proposed)", results["full_model"]),
        ("Full model, no instance norm", results["full_model_no_instance_norm"])]
lines = ["| Model | AUC-ROC | F1 | Precision | Recall |",
         "|---|---|---|---|---|"]
for name, m in rows:
    lines.append(f"| {name} | {m['auc']:.3f} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |")
lines.append("")
lines.append("| k-hop radius | AUC-ROC | F1 | Precision | Recall |")
lines.append("|---|---|---|---|---|")
for k, m in k_results.items():
    lines.append(f"| {k} | {m['auc']:.3f} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |")
with open("outputs/results_table.md", "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))

with open("outputs/histories.json", "w") as f:
    json.dump(histories, f, indent=2, default=float)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "deepwalk_mlp": "#E69F00",
    "plain_gcn": "#0072B2",
    "full_model": "#009E73",
    "full_model_no_instance_norm": "#D55E00",
}
LABELS = {
    "deepwalk_mlp": "DeepWalk + MLP",
    "plain_gcn": "Plain GCN",
    "full_model": "Full model (proposed)",
    "full_model_no_instance_norm": "Full model, no instance norm",
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
for key, hist in histories.items():
    epochs = [h["epoch"] for h in hist]
    train_loss = [h["train_loss"] for h in hist]
    ax.plot(epochs, train_loss, color=COLORS[key], linewidth=2, label=LABELS[key])
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (BCE)")
ax.set_title("Training loss vs. epoch, all models")
ax.grid(True, alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
hist = histories["full_model"]
epochs = [h["epoch"] for h in hist]
train_loss = [h["train_loss"] for h in hist]
val_loss = [h["val_loss"] for h in hist]
ax.plot(epochs, train_loss, color="#0072B2", linewidth=2, label="Train loss")
ax.plot(epochs, val_loss, color="#D55E00", linewidth=2, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("BCE loss")
ax.set_title("Full model: train vs. validation loss")
ax.grid(True, alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig("outputs/loss_vs_epoch.png", dpi=160)
print("\nSaved outputs/loss_vs_epoch.png")
