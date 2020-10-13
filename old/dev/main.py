import pandas as pd
import torch

from dev.data.dataloader import DataLoader
from dev.data.mol_multi_iput_dataset import MolGraphMultinputDataset
from dev.model.gcn import GCNMultiInputPredictor
from dev.featurizer.chem_featurizer import default_atom_featurizer
from model.torch_model import PytorchModel

import dev.model.loss as model_loss

Dataset = MolGraphMultinputDataset


def load_data(df, smiles_column, tasks, add_input_cols, featurizer):
    full_ds = Dataset(df=df, smiles_column=smiles_column,
                      tasks=tasks,
                      featurizer=featurizer,
                      add_input_cols=add_input_cols
                      )
    train_set, valid_set, test_set = full_ds.random_split([8, 1, 1])

    train_dl = DataLoader(train_set, batch_size=10)
    valid_dl = DataLoader(valid_set, batch_size=10)
    test_dl = DataLoader(test_set, batch_size=10)

    # t0 = train_dl[0]
    # g=t0[0][1]._graph_backend
    # import networkx as nx
    # import matplotlib.pyplot as plt
    # nx.draw(g, with_labels=True, font_weight='bold')
    # plt.show()
    # plt.close()
    # for gs in [g.subgraph(c) for c in nx.algorithms.components.strongly_connected_components(g)]:
    #    nx.draw(gs, with_labels=True, font_weight='bold')
    #    plt.show()
    #    plt.close()
    return train_dl, valid_dl, test_dl


def gen_model(featurizer, additional_inputs, hidden_graph_output, hidden_feats, n_tasks):
    post_input_module = None

    model = PytorchModel(GCNMultiInputPredictor(in_feats=len(featurizer),
                                                additional_inputs=additional_inputs,
                                                hidden_graph_output=hidden_graph_output,
                                                hidden_feats=hidden_feats,
                                                post_input_module=post_input_module,
                                                # post_input_hidden_layer=args['post_input_hidden_layer'],
                                                n_tasks=n_tasks,
                                                ),
                         predict_function=Dataset.model_inject,
                         #                        name="test",
                         #                        dir="~/test",
                         )
    model.gcn_featurizer = featurizer
    # model.config = config

    optimizer = torch.optim.Adam(model.module.parameters(), lr=0.001)

    loss_fn = getattr(model_loss, "MSE")()

    model.compile(optimizer, loss_fn, metrics=["relmae", "rmse"])

    return model


if __name__ == '__main__':
    import rdkit
    import rdkit.Chem.Descriptors
    import numpy as np

    np.random.seed(1)
    df = pd.read_csv("../notebooks/delaney-processed.csv")
    smiles_column = "smiles"
    tasks = ["mw"]
    add_input_cols = []
    featurizer = default_atom_featurizer

    mols = {smiles: rdkit.Chem.MolFromSmiles(smiles) for smiles in df[smiles_column]}

    df["mw"] = df[smiles_column].apply(lambda s: rdkit.Chem.Descriptors.MolWt(mols[s]))

    # df["id"] = np.arange(len(df))
    # df["id2"] = np.arange(len(df)) ** 2

    df = df[[smiles_column] + tasks + add_input_cols].iloc[:50]

    train_dl, valid_dl, test_dl = load_data(df, smiles_column, tasks, add_input_cols, featurizer=featurizer)




    model = gen_model(
        featurizer=featurizer,
        additional_inputs=len(add_input_cols),
        hidden_graph_output=100,
        hidden_feats=[3,2],
        n_tasks=len(tasks)
    )

    model.train(
        data_loader=train_dl,
        validation_loader=valid_dl,
        test_loader=test_dl,
        epochs=100,
        callbacks=[]
    )
