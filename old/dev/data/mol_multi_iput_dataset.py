import rdkit
import torch

from dev.data.dataset import Dataset
from dev.model.layer.graph_convolution.graph import mol_to_graph, merge_graphs, TorchGraph
import numpy as np


class MolGraphMultinputDataset(Dataset):
    def __init__(self, df, smiles_column, tasks, featurizer, add_input_cols=None):
        if add_input_cols is None:
            add_input_cols = []

        unique_graphs = {smiles: mol_to_graph(rdkit.Chem.MolFromSmiles(smiles), featurizer)[0]
                         for smiles in df[smiles_column].unique()
                         }

        smiles = df[smiles_column].values
        add_ip = df[add_input_cols].values

        input = np.zeros((len(smiles), 3), dtype=object)

        for i, s in enumerate(smiles):
            input[i, 0] = s
            input[i, 1] = unique_graphs[s]
            input[i, 2] = add_ip[i]
        output = df[tasks].values

        super().__init__(input_data=input, output_data=output)

    def collate(self, input_data, output_data):

        input_data, output_data = super(MolGraphMultinputDataset, self).collate(input_data, output_data)
        smiles = input_data[:, 0]
        graphs = input_data[:, 1]
        additional_input = np.stack(input_data[:, 2], axis=0)

        graph = merge_graphs(graphs)
        return (smiles,graph,additional_input), output_data

    @staticmethod
    def model_inject(model, data, device):
        smiles, bg, addi = data

        bg = TorchGraph(bg)
        bg.to(device)
        #h = torch.from_numpy(bg.feature_matrix)
        #h = h.to(device)
        addi = torch.from_numpy(addi).to(device)

        return model(bg, bg.feature_matrix, addi)

