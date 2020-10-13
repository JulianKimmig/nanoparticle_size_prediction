#uc
"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch as th
from torch import nn
from torch.nn import init

from model.layer.graph_convolution.graph import MolGraph, FixedGraph


class GraphConvolution(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph: FixedGraph, feat):


        normed_adj = graph.get_normed_adj(device=feat.device)
        result = th.matmul(normed_adj, feat)
        result = th.matmul(result, self.weight)

        norm = graph.get_back_norm(device=feat.device)

        result = result * norm

        result = result + self.bias

        if self._activation is not None:
            result = self._activation(result)

        return result

        graph = graph.local_var()

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            if weight is not None:
                feat = th.matmul(feat, weight)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
        else:
            # aggregate first then mult W
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            if weight is not None:
                rst = th.matmul(rst, weight)

        if self._norm != 'none':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None,
                 residual=True, batchnorm=True, dropout=0.):
        super().__init__()

        self.activation = activation
        self.graph_conv = GraphConvolution(in_feats=in_feats, out_feats=out_feats,
                                           norm='none', activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats



class WeightAndSum(nn.Module):
    def __init__(self, n_in_feats):
        super(WeightAndSum, self).__init__()
        self.weighting_of_nodes = nn.Sequential(
            nn.Linear(n_in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g, feats):
        weights = self.weighting_of_nodes(feats) #dims = nodes,1
        weight_feats = weights*feats  #dims = nodes,feats
        sumed_nodes = torch.sum(weight_feats,dim=0).unsqueeze_(0)
        return sumed_nodes



class WeightedSumAndMax2(nn.Module):
    def __init__(self, in_feats):
        super().__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        sumed_nodes = self.weight_and_sum(bg, feats)
        max_nodes,_ = torch.max(feats,dim=0, keepdim=True)
        sum_max = torch.cat([sumed_nodes, max_nodes],dim=1)
        return sum_max