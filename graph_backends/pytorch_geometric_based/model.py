#uc
import torch
from torch import nn
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.pool import avg_pool
from torch_scatter import scatter_add, scatter_max, scatter_mean
import numpy as np


class WeightedSum(nn.Module):
    def __init__(self, n_in_feats):
        super(WeightedSum, self).__init__()
        self.weighting_of_nodes = nn.Sequential(
            nn.Linear(n_in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, feats, batch):
        # feats = nodes,last_gcn_feats
        weights = self.weighting_of_nodes(feats).squeeze()  # dims = nodes,
        weight_feats = feats.transpose(1, 0)  # dims = last_gcn_feats,nodes
        weight_feats = weights * weight_feats  # dims = last_gcn_feats,nodes
        weight_feats = weight_feats.transpose(1, 0)  # dims = nodes,last_gcn_feats

        summed_nodes = scatter_add(weight_feats, batch,
                                   dim=0)  # zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes


class PoolMax(nn.Module):
    def forward(self, feats, batch):
        maxed_nodes, _ = scatter_max(feats, batch, dim=0)
        return maxed_nodes

class PoolMean(nn.Module):
    def forward(self, feats, batch):
        meaned_nodes  = scatter_mean(feats, batch, dim=0)
        return meaned_nodes

class PoolSum(nn.Module):
    def forward(self, feats, batch):
        summed_nodes = scatter_add(feats, batch,
                                   dim=0)  # zeros.scatter_add(0, segment_ids, weight_feats) #dims = number of graphs,last_gcn_feats
        return summed_nodes


class WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.weighed_sum = WeightedSum(in_feats)

    def forward(self, feats, batch):
        summed_nodes = self.weighed_sum(feats, batch)  # dims = number of graphs,last_gcn_feats

        # max_nodes = self.graph_max(bg,feats) #dims = number of graphs,last_gcn_feats
#        segment_ids = batch.repeat(feats.shape[1]).view((-1, feats.shape[1]))  # dims = nodes,last_gcn_feats
#        num_segments = batch[-1] + 1  # dims = 1  == number of graphs

        maxed_nodes, _ = scatter_max(feats, batch, dim=0)

        sum_max = torch.cat([summed_nodes, maxed_nodes], dim=1)
        return sum_max


class MergedPooling(nn.Module):
    def __init__(self, pooling_layer):
        super().__init__()
        self.pooling_layer =  nn.ModuleList(pooling_layer)

    def forward(self, feats, batch):
        return torch.cat([pl(feats, batch) for pl in self.pooling_layer], dim=1)


class GCNMultiInputPredictor(nn.Module):
    def __init__(self, in_feats, additional_inputs, hidden_graph_output, hidden_feats, post_input_module, n_tasks,
                 activation=None, pooling="wsum_max"):
        super(GCNMultiInputPredictor, self).__init__()

        if len(hidden_feats) > 0:
            if hidden_feats[0] is None:
                hidden_feats[0] = in_feats

            for i in range(1, len(hidden_feats)):
                if hidden_feats[i] is None:
                    hidden_feats[i] = hidden_feats[i - 1]
        in_channels = in_feats

        gnn_l = []
        for out_feats in hidden_feats:
            gnn_l.append(
                GCNConv(in_channels=in_channels,
                        out_channels=out_feats)
            )
            in_channels = out_feats

        self.gnn = nn.ModuleList(gnn_l)

        pools = []
        last_out = 0
        for p in pooling:
            if p == "wsum_max":
                pools.append(WeightedSumAndMax(in_channels))
                last_out += 2 * in_channels
            elif p == "sum":
                pools.append(PoolSum())
                last_out += in_channels
            elif p == "max":
                pools.append(PoolMax())
                last_out += in_channels
            elif p == "mean":
                pools.append(PoolMean())
                last_out += in_channels
            elif p == "weight_sum":
                pools.append(WeightedSum(in_channels))
                last_out += in_channels
            else:
                raise NotImplementedError("pooling '{}' not found".format(p))
        self.pooling = MergedPooling(pools)

        graph_out_layer=[nn.Linear(last_out, hidden_graph_output),]
        sig_graph_out=False
        if sig_graph_out:
            graph_out_layer.append(nn.Sigmoid())

        self.graph_out = nn.Sequential(
            *graph_out_layer
        )

        last_out = hidden_graph_output + additional_inputs

        self.post_input_module = post_input_module
        if self.post_input_module:
            op = self.post_input_module.forward(torch.from_numpy(np.random.random(last_out)).float())
            self.post_input_module.output_size = op.size()[0]
            last_out = self.post_input_module.output_size

        self.final_layer_needed = False
        if last_out != n_tasks:
            self.final_layer_needed = True
            self.final_layer = nn.Linear(last_out, n_tasks)

    def forward(self, data):
        feats = data.x
        edges = data.edge_index
        add_ip = data.additional_input
        #print(feats)
        for gnn in self.gnn:
            feats = gnn(feats, edges)

        feats = self.pooling(feats, data.batch)
        feats = self.graph_out(feats)

        feats = torch.cat([feats, add_ip], dim=1)
        if self.post_input_module is not None:
            feats = self.post_input_module(feats)

        if self.final_layer_needed:
            feats = self.final_layer(feats)
        return feats

    @staticmethod
    def batch_data_converter(data):
        return data, data.y

    @staticmethod
    def predict_function(model,data,device):
        data.to(device)
        pred = model(data)
        #print(pred)
        return pred