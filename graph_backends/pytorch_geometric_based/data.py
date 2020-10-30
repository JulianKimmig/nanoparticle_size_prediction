# uc
import random

import torch
import torch_geometric
from torch_geometric.data.dataloader import DataLoader
import numpy as np
import pandas as pd


def smile_mol_featurizer(atom_featurizer, canonical_rank=True, with_H=True):
    import rdkit
    from rdkit.Chem import rdmolfiles, rdmolops
    from rdkit.Chem.rdmolfiles import MolFromSmiles

    def to_graph(smiles):
        smiles = smiles[0]
        mol = MolFromSmiles(smiles)
        if with_H:
            mol = rdkit.Chem.AddHs(mol)
        elif with_H is None:
            pass
        else:
            mol = rdkit.Chem.RemoveHs(mol)

        if canonical_rank:
            atom_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, atom_order)
        atoms = mol.GetAtoms()
        node_features = np.zeros((len(atoms), len(atom_featurizer)))

        for i, atom in enumerate(atoms):
            atom_features = atom_featurizer(atom)
            node_features[i] = atom_features

        row, col = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]

        edge_index = np.array([row, col])

        return {"mol": mol, "node_features": node_features, "edge_index": edge_index}

    return to_graph


def dataframe_to_data_predict_set(
    df, to_graph, graph_columns, batch_size=32, add_kwargs={}, verbose=True
):
    data_list = []
    l = len(df)
    gd = {}
    for i, row in df.iterrows():
        if verbose:
            print(
                "load data {}/{} ({:.2f}%)    ".format(i + 1, l, 100 * (i + 1) / l),
                end="\r",
            )
        gdi = row[graph_columns].values
        rep_gdi = repr(gdi)
        if rep_gdi in gd:
            graph_data = gd[rep_gdi]
        else:
            graph_data = to_graph(gdi)
            gd[rep_gdi] = graph_data
        _add_kwargs = {}
        for key, col_list in add_kwargs.items():
            _add_kwargs[key] = torch.from_numpy(
                np.expand_dims(row[col_list].values.astype(float), 0)
            ).float()

        data = torch_geometric.data.data.Data(
            x=torch.from_numpy(graph_data["node_features"]).float(),
            edge_index=torch.from_numpy(
                graph_data["edge_index"],
            ).long(),
            y=row.name,
            **_add_kwargs
        )
        data_list.append(data)
    if verbose:
        print("")
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


def dataframe_to_data_train_sets(
    df,
    to_graph,
    graph_columns,
    task_cols,
    batch_size=32,
    add_kwargs={},
    seed=None,
    split=[1],
    shuffle=True,
    check_null=True,
    verbose=True,
):
    data_list = []
    l = len(df)
    gd = {}
    if check_null:
        null_graph = pd.isnull(df[graph_columns])
        if np.any(null_graph):
            raise ValueError("one of the graph columns contains none values")
        for key, col_list in add_kwargs.items():
            null_add = pd.isnull(df[col_list])
            if np.any(null_add):
                raise ValueError("one of the add columns contains none values")

    for i, row in df.iterrows():
        if verbose:
            print("load data {}/{}    ".format(i + 1, l), end="\r")
        gdi = row[graph_columns].values
        rep_gdi = repr(gdi)
        if rep_gdi in gd:
            graph_data = gd[rep_gdi]
        else:
            graph_data = to_graph(gdi)
            gd[rep_gdi] = graph_data
        y = torch.from_numpy(
            np.expand_dims(row[task_cols].values.astype(float), 0)
        ).float()

        _add_kwargs = {}
        for key, col_list in add_kwargs.items():
            add_vals = np.expand_dims(row[col_list].values.astype(float), 0)
            _add_kwargs[key] = torch.from_numpy(add_vals).float()

        data = torch_geometric.data.data.Data(
            x=torch.from_numpy(graph_data["node_features"]).float(),
            edge_index=torch.from_numpy(
                graph_data["edge_index"],
            ).long(),
            y=y,
            **_add_kwargs
        )
        data_list.append(data)
    if verbose:
        print("")
    if shuffle:
        if seed is not None:
            rand = random.Random(seed)
        else:
            rand = random.Random()
        rand.shuffle(data_list)

    split = np.array(split)
    split = split / sum(split)

    start = 0
    sets = []
    original_length = len(data_list)
    for i in split[:-1]:
        end = start + int(original_length * i)
        sets.append(
            DataLoader(data_list[start:end], batch_size=batch_size, shuffle=shuffle)
        )
        start = end
    end = original_length
    sets.append(
        DataLoader(data_list[start:end], batch_size=batch_size, shuffle=shuffle)
    )

    return sets


# data_laoder = DataLoader(data_list, batch_size=32, shuffle=True)

# data_laoder.
