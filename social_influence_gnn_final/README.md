# Social Influence Prediction — GNN pipeline

Implements the pipeline from the problem statement: ego-network sampling, a
GCN encoder, instance normalization, handcrafted-feature fusion, an MLP
prediction head, BCE training, plus the four baselines and the ablations
from the evaluation section.

## Setup

```
pip install -r requirements.txt
python3 main.py
```

Takes about 75-80 seconds on a normal laptop, no GPU needed. Outputs land in
`outputs/`: `results.json`, `results_table.md`, `histories.json`,
`loss_vs_epoch.png`, `full_model_weights.npz`.

## Notes on choices made

**No PyTorch.** Couldn't get `torch` installed in time, so the GCN layer,
instance-norm layer, MLP, Adam optimizer and BCE loss are all written by
hand in `src/nn.py` with manual forward/backward passes instead of
autograd. Slower to write, but it works and doesn't need a GPU. If you have
torch available, `GCNLayer` maps directly onto `torch_geometric.nn.GCNConv`
and the rest onto standard `torch.nn` modules — the architecture doesn't
change, just the backend.

**Synthetic data instead of OAG/Digg/Weibo.** Couldn't get the real DeepInf
datasets in time either, so this generates a scale-free graph (Barabási–
Albert) and runs an independent-cascade simulation on top of it to get
activation times, which double as the action labels. The problem statement
lists this as one of the two valid dataset options, so it's not really a
shortcut. Swapping in the real dataset later only touches `src/data.py`.

**DeepWalk baseline is NetMF, not literal skip-gram.** Same embedding
objective as DeepWalk (Perozzi et al.), computed via matrix factorization
(Qiu et al., WSDM 2018) instead of running skip-gram SGD. Much faster in
plain NumPy, no need for gensim.

**Two of the handcrafted features are duplicates.** "Neighbor influence
ratio" and "fraction of active neighbors" work out to the same number
(active neighbors / degree) on this graph. Left both in since the spec
asks for both, but worth flagging if asked about it.

## Layout

```
src/data.py     graph + cascade simulation, ego-network sampling, handcrafted features
src/nn.py       Linear, GCNLayer, InstanceNorm, Adam, BCE loss
src/models.py   GCNEncoder, PlainGNN, FullModel, DeepWalk embeddings
src/train.py    training loops + evaluation metrics
main.py         runs everything end to end
```
