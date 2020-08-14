"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch as th
from torch import nn
from torch.nn import init

from dev.model.layer.graph_convolution.graph import TorchGraph
from model.layer.graph_convolution.graph import MolGraph


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
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph: TorchGraph, feat):
        normed_adj = graph.sparse_norm_adj_tilde


        result = th.matmul(normed_adj, feat)
        result = th.matmul(result, self.weight)
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


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None,
                 residual=True, batchnorm=True, dropout=0.):
        super().__init__()

        self.activation = activation
        self.graph_conv = GraphConvolution(in_feats=in_feats, out_feats=out_feats,
                                           norm='none', activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.batchnorm = batchnorm
        self.bn_layer = nn.BatchNorm1d(out_feats) if batchnorm else None


    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        new_feats = self.dropout(new_feats)
        if self.batchnorm:
            new_feats = self.bn_layer(new_feats)
        return new_feats



class WeightAndSum(nn.Module):
    def __init__(self, n_in_feats):
        super(WeightAndSum, self).__init__()
        self.weighting_of_nodes = nn.Sequential(
            nn.Linear(n_in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, g:TorchGraph, feats):
        #feats = nodes,last_gcn_feats
        weights = self.weighting_of_nodes(feats).squeeze() #dims = nodes,
        weight_feats = feats.transpose(1,0) #dims = last_gcn_feats,nodes
        weight_feats = weights*weight_feats  #dims = last_gcn_feats,nodes
        weight_feats = weight_feats.transpose(1,0) #dims = nodes,last_gcn_feats
        #sumed_nodes = torch.sum(weight_feats,dim=0).unsqueeze_(0)

        segment_ids = g.node_indices.repeat(feats.shape[1]).view((-1,feats.shape[1])) #dims = nodes,last_gcn_feats
        num_segments = g.subgraphs  #dims = 1  == number of graphs

        zeros=torch.zeros(num_segments,feats.shape[1]).to(g.device) #dims = number of graphs,last_gcn_feats
        summed_nodes = zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes

class GraphMax(nn.Module):
    def __init__(self):
        super(GraphMax, self).__init__()

    def forward(self, g:TorchGraph, feats):
        print(feats.shape)
        segment_ids = g.node_indices.repeat(feats.shape[1]).view((-1,feats.shape[1])) #dims = nodes,last_gcn_feats
        num_segments = g.subgraphs  #dims = 1  == number of graphs

        zeros=torch.zeros(num_segments,feats.shape[1]).to(g.device)
        print(num_segments)

        zeros=torch.zeros(num_segments,feats.shape[1]).to(g.device) #dims = number of graphs,last_gcn_feats
        summed_nodes = zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes

class WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.weight_and_sum = WeightAndSum(in_feats)
        self.graph_max = GraphMax()

    def forward(self, bg, feats):
        summed_nodes = self.weight_and_sum(bg, feats)#dims = number of graphs,last_gcn_feats
        print(summed_nodes.shape)

        #max_nodes = self.graph_max(bg,feats) #dims = number of graphs,last_gcn_feats

        print(bg.node_split_indices)
        
        max_nodes = torch.max(feats,)

        print(max_nodes.shape)



        sum_max = torch.cat([summed_nodes, max_nodes],dim=1)
        return sum_max