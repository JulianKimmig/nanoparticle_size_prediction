import torch
from torch import nn

from dev.model.layer.graph_convolution.gcn import GCN
from dev.model.layer.graph_convolution.gcn_layer import WeightedSumAndMax


class GCNMultiInputPredictor(nn.Module):
    def __init__(self, in_feats, additional_inputs, hidden_feats=None, activation=None, residual=None, batchnorm=None,
                 dropout=None, classifier_hidden_feats=128, hidden_graph_output=128, post_input_module=None,
                 classifier_dropout=0., n_tasks=1):
        super().__init__()

        if len(hidden_feats)>0:
            if hidden_feats[0] is None:
                hidden_feats[0] = in_feats

            for i in range(1,len(hidden_feats)):
                if hidden_feats[i] is None:
                    hidden_feats[i] = hidden_feats[i-1]

        self.gnn = GCN(in_feats=in_feats,
                        hidden_feats=hidden_feats,
                        activation=activation,
                        residual=residual,
                        batchnorm=batchnorm,
                        dropout=dropout)
        gnn_out_feats = self.gnn.hidden_feats[-1]
        self.pooling = WeightedSumAndMax(gnn_out_feats)

    def forward(self, bg, feats, additional_inputs):
        node_feats = self.gnn(bg, feats)
        graph_feats = self.pooling(bg, node_feats)
        print("AAA",graph_feats.shape)
        out = self.graph_collect(graph_feats)
        print("BBBB",out)
        out = torch.cat([out, additional_inputs], dim=1)

        #  for l in self.post_input_hidden_layer:
        #       print(l)
        #       out=l(out)
        if self.post_input_module:
            out = self.post_input_module(out)

        if self.condense is not None:
            out = self.condense(out)

        return out