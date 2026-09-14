| Model | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.666 | 0.426 | 1.000 | 0.270 |
| DeepWalk + MLP | 0.662 | 0.000 | 0.000 | 0.000 |
| Plain GCN | 0.807 | 0.604 | 1.000 | 0.432 |
| Full model (proposed) | 0.835 | 0.538 | 0.933 | 0.378 |
| Full model, no instance norm | 0.819 | 0.607 | 0.895 | 0.459 |

| k-hop radius | AUC-ROC | F1 | Precision | Recall |
|---|---|---|---|---|
| k=1 | 0.707 | 0.462 | 0.800 | 0.324 |
| k=2 | 0.801 | 0.618 | 0.944 | 0.459 |
| k=3 | 0.865 | 0.643 | 0.947 | 0.486 |