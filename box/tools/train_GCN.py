import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
import torch.nn.functional as F
import torch.optim as optim



class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, use_linear=False, activation=F.relu):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                #h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h


def train(model, G, args):
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(G, G.ndata["feat"])
        loss_train = F.mse_loss(output[G.ndata["train_mask"]], G.ndata["label"][G.ndata["train_mask"]])
        loss_train.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print("epoch ", end="")
            print(epoch, end=" : ")
            print("loss", end=" = ")
            print(loss_train.item())
    model.eval()
    with torch.no_grad():
        ret = model(G, G.ndata["feat"])
    return ret
