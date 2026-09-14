import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from .nn import bce_loss, bce_grad, Adam


def evaluate(y_true, p_pred, threshold=0.5):
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    y_hat = (p_pred >= threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, p_pred)
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
    }


def train_gnn_model(model, train_data, val_data, epochs=60, lr=0.01, verbose=True, tag=""):
    opt = Adam(model.layers(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        p, logit = model.forward(train_data["X"], train_data["adj"],
                                  train_data["valid_mask"], train_data["handcrafted"])
        loss = bce_loss(p, train_data["y"])
        dlogit = bce_grad(p, train_data["y"])
        model.backward(dlogit)
        opt.step()

        pv, _ = model.forward(val_data["X"], val_data["adj"],
                               val_data["valid_mask"], val_data["handcrafted"])
        val_loss = bce_loss(pv, val_data["y"])
        row = {"epoch": ep, "train_loss": loss, "val_loss": val_loss}

        if ep % 5 == 0 or ep == epochs:
            val_metrics = evaluate(val_data["y"], pv)
            row.update(val_metrics)
            if verbose:
                print(f"  [{tag}] epoch {ep:3d}  train_loss={loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_auc={val_metrics['auc']:.3f}  val_f1={val_metrics['f1']:.3f}")
        history.append(row)
    return history


def train_mlp_model(model, Xtr, ytr, Xval, yval, epochs=80, lr=0.01, verbose=True, tag=""):
    opt = Adam(model.layers(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        p, logit = model.forward(Xtr)
        loss = bce_loss(p, ytr)
        dlogit = bce_grad(p, ytr)
        model.backward(dlogit)
        opt.step()

        pv, _ = model.forward(Xval)
        val_loss = bce_loss(pv, yval)
        row = {"epoch": ep, "train_loss": loss, "val_loss": val_loss}

        if ep % 10 == 0 or ep == epochs:
            val_metrics = evaluate(yval, pv)
            row.update(val_metrics)
            if verbose:
                print(f"  [{tag}] epoch {ep:3d}  train_loss={loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_auc={val_metrics['auc']:.3f}  val_f1={val_metrics['f1']:.3f}")
        history.append(row)
    return history
