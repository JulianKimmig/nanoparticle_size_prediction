# uc
import torch
import torch.nn as nn

from dgllife.model import GCN, WeightedSumAndMax, MLPPredictor

from model.layer.graph_convolution.gcn import GCN2
from model.layer.graph_convolution.gcn_layer import WeightedSumAndMax2
import numpy as np


class GCNMultiInputPredictor(nn.Module):
    def __init__(
        self,
        in_feats,
        additional_inputs,
        hidden_feats=None,
        activation=None,
        residual=None,
        batchnorm=None,
        dropout=None,
        classifier_hidden_feats=128,
        hidden_graph_output=128,
        post_input_module=None,
        classifier_dropout=0.0,
        n_tasks=1,
    ):

        super().__init__()
        if len(hidden_feats) > 0:
            if hidden_feats[0] is None:
                hidden_feats[0] = in_feats

            for i in range(1, len(hidden_feats)):
                if hidden_feats[i] is None:
                    hidden_feats[i] = hidden_feats[i - 1]

        self.gnn = GCN(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            activation=activation,
            residual=residual,
            batchnorm=batchnorm,
            dropout=dropout,
        )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.graph_output = MLPPredictor(
            2 * gnn_out_feats,
            classifier_hidden_feats,
            hidden_graph_output,
            classifier_dropout,
        )

        last_out = hidden_graph_output + additional_inputs

        self.post_input_module = post_input_module
        if self.post_input_module:
            op = self.post_input_module.forward(
                torch.from_numpy(np.random.random(last_out)).float()
            )
            self.post_input_module.output_size = op.size()[0]
            last_out = self.post_input_module.output_size

        if last_out != n_tasks:
            self.condense = nn.Linear(in_features=last_out, out_features=n_tasks)
        else:
            self.condense = None

    def forward(self, bg, feats, additional_inputs):

        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        out = self.graph_output(graph_feats)
        out = torch.cat([out, additional_inputs], dim=1)

        #  for l in self.post_input_hidden_layer:
        #       print(l)
        #       out=l(out)
        if self.post_input_module:
            out = self.post_input_module(out)

        if self.condense is not None:
            out = self.condense(out)

        return out


class GCNMultiInputPredictor2(nn.Module):
    def __init__(
        self,
        in_feats,
        additional_inputs,
        hidden_feats=None,
        activation=None,
        residual=None,
        batchnorm=None,
        dropout=None,
        classifier_hidden_feats=128,
        hidden_graph_output=128,
        post_input_module=None,
        classifier_dropout=0.0,
        n_tasks=1,
    ):

        super().__init__()
        if len(hidden_feats) > 0:
            if hidden_feats[0] is None:
                hidden_feats[0] = in_feats

            for i in range(1, len(hidden_feats)):
                if hidden_feats[i] is None:
                    hidden_feats[i] = hidden_feats[i - 1]

        self.gnn = GCN2(
            in_feats=in_feats,
            hidden_feats=hidden_feats,
            activation=activation,
            residual=residual,
            batchnorm=batchnorm,
            dropout=dropout,
        )
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.pooling = WeightedSumAndMax2(gnn_out_feats)

        self.graph_collect = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(2 * gnn_out_feats, classifier_hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(classifier_hidden_feats),
            nn.Linear(classifier_hidden_feats, hidden_graph_output),
        )

        last_out = hidden_graph_output + additional_inputs

        self.post_input_module = post_input_module
        if self.post_input_module:
            last_out = self.post_input_module.output_size

        if last_out != n_tasks:
            self.condense = nn.Linear(in_features=last_out, out_features=n_tasks)
        else:
            self.condense = None

    def forward(self, bg, feats, additional_inputs):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.pooling(bg, node_feats)
        out = self.graph_collect(graph_feats)
        out = torch.cat([out, additional_inputs], dim=1)

        #  for l in self.post_input_hidden_layer:
        #       print(l)
        #       out=l(out)
        if self.post_input_module:
            out = self.post_input_module(out)

        if self.condense is not None:
            out = self.condense(out)

        return out
