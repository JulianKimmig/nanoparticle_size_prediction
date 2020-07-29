import argparse
import os
import sys
import time

import torch
from dgllife.utils import smiles_to_bigraph, RandomSplitter
from json_dict import JsonDict
from torch import nn
from torch.utils.data import DataLoader

from featurizer import load_featurizer
from model import callbacks
from model.data_loader import MoleculeAdditionalInputCSVDataset
from model.gcn import GCNMultiInputPredictor
import numpy as np
import pandas as pd

# from model.layer import Flatten
from model.layer import Flatten
import model.loss as model_loss
from model.torch_model import PytorchModel


def generate_fcn_model(config, input_dims):
    def load_layer(layer_config):
        module = getattr(nn, layer_config["module"])
        return module(*layer_config.get("args", []), **layer_config.get("kwargs", {}))

    layer = [load_layer(l) for l in config.get("layer")]

    if len(input_dims) > 1:
        layer.append(Flatten())

    fcn_model = nn.Sequential(*layer)

    op = fcn_model.forward(torch.from_numpy(np.random.random(input_dims)).float())
    fcn_model.output_size = op.size()[0]

    return fcn_model


def load_model(config,load=True):
    featurizer = load_featurizer(config.get("featurizer"))

    fcn_model_config = config.getsubdict("fcn_model")
    post_input_module = generate_fcn_model(fcn_model_config, input_dims=(
        int(config.get("gcn_output_dims")) + len(config.get("additional_input_names")),
    ))

    model = PytorchModel(GCNMultiInputPredictor(in_feats=featurizer.feat_size(),
                                                additional_inputs=len(config.get("additional_input_names")),
                                                hidden_graph_output=int(config.get("gcn_output_dims")),
                                                hidden_feats=config.get("gcn_layer_sizes"),
                                                post_input_module=post_input_module,
                                                # post_input_hidden_layer=args['post_input_hidden_layer'],
                                                n_tasks=len(config.get("task_names")),
                                                ),
                         predict_function=MoleculeAdditionalInputCSVDataset.model_regress,
                         name=config.get("name"),
                         dir=config.get("path"),
                         )
    model.gcn_featurizer = featurizer
    model.config = config

    optimizer = torch.optim.Adam(model.module.parameters(), lr=config.get("learning_rate"))

    loss_fn = getattr(model_loss, config.get("loss_function", "name"))(
        **config.get("loss_function", "kwargs"))

    model.compile(optimizer, loss_fn, metrics=config.get('metrics'))

    if load:
        try:
            print("try loading model from {}".format(model.default_filename()))
            model.load()
            print("load successful")
        except FileNotFoundError as e:
            print("new model, please train me!")

    return model


def train(model, training_data, train_config):
    model.save()
    model_config = model.config
    df = training_data

    tasks = model_config.get("task_names")
    missing_tasks = list(set(tasks) - set(df.columns))
    if len(missing_tasks) > 0:
        raise ValueError("Missing task columns: {}".format(",".join(missing_tasks)))

    add_input_cols = model_config.get("additional_input_names")
    missing_inputs = list(set(add_input_cols) - set(df.columns))
    if len(missing_inputs) > 0:
        raise ValueError("Missing input columns: {}".format(",".join(missing_inputs)))

    smiles_col = model_config.get("smiles_column")
    assert smiles_col in df.columns, "smiles column '{}' not in dataframe".format(smiles_col)

    cache_file_path = os.path.join(model.dir, "training", "data")
    os.makedirs(cache_file_path, exist_ok=True)
    dataset = MoleculeAdditionalInputCSVDataset(df=df, smiles_to_graph=smiles_to_bigraph,
                                                smiles_column=smiles_col,
                                                additional_input_names=add_input_cols,
                                                task_names=tasks,

                                                node_featurizer=model.gcn_featurizer,
                                                edge_featurizer=None,  # BaseBondFeaturizer({
                                                #     'he': lambda bond: [0 for _ in range(10)]
                                                # }),
                                                cache_file_path=os.path.join(cache_file_path, "data"),
                                                load=True
                                                )
    train_set, val_set, test_set = RandomSplitter.train_val_test_split(dataset,
                                                                       random_state=config.get("random_seed"))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_config.get("batch_size"),
                              shuffle=True,
                              collate_fn=train_set.dataset.collate)
    val_loader = DataLoader(dataset=val_set,
                            batch_size=train_config.get("batch_size"),
                            shuffle=False,
                            collate_fn=val_set.dataset.collate)

    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=train_config.get("batch_size"),
                                 collate_fn=test_set.dataset.collate)

    cbs = train_config.get("callbacks").copy()
    for cb in cbs:
        for key, val in cb.get("kwargs", {}).items():
            if val == 'validation_loader':
                cb["kwargs"][key] = val_loader
            if val == 'training_loader':
                cb["kwargs"][key] = train_loader

        for i, arg in enumerate(cb.get("args", [])):
            if arg == 'validation_loader':
                cb["args"][i] = val_loader
            if arg == 'training_loader':
                cb["args"][i] = train_loader
    train_config.save()
    cbs = [callbacks.deserialize(cb) for cb in cbs]
    model.train(data_loader=train_loader, validation_loader=val_loader, test_loader=test_loader, epochs=train_config.get("epochs"),
                callbacks=cbs)


#
def predict(model, dataframe,return_array=False,verbose=True):
    model_config = model.config
    df = dataframe.copy()

    add_input_cols = model_config.get("additional_input_names")
    missing_inputs = list(set(add_input_cols) - set(df.columns))
    if len(missing_inputs) > 0:
        raise ValueError("Missing input columns: {}".format(",".join(missing_inputs)))

    smiles_col = model_config.get("smiles_column")
    assert smiles_col in df.columns, "smiles column '{}' not in dataframe".format(smiles_col)

    df_len = len(df)

    tasks = ["predicted_" + t for t in model_config.get("task_names")]

    if verbose:
        print("generate graphs for smiles")
    graphs = {smiles: smiles_to_bigraph(smiles, node_featurizer=model.gcn_featurizer) for smiles in
              df[smiles_col].unique()}
    feats = {smiles: graph.ndata['nfeats'] for smiles, graph in graphs.items()}

    i = 1
    for t in tasks:
        df[t] = None

    if verbose:
        print("predict data")

    for index, row in df.iterrows():
        if verbose:
            print("\r{}/{} ({}%)".format(i, df_len, 100 * i / (df_len))," "*16, end="")
        smiles = row[smiles_col]
        add_input = torch.from_numpy(row[add_input_cols].values.astype(np.float32)).unsqueeze(0)
        graph = graphs[smiles]
        nfeats = feats[smiles]
        graph.ndata['nfeats'] = nfeats

        pred = model.predict((smiles, graph, add_input))[0]

        df.loc[index, tasks] = pred
        i += 1
    if verbose:
        print()
    if return_array:
        return df[tasks].values
    return df


def update_config(config, args):
    model_config = config.getsubdict("model")
    train_config = config.getsubdict("training")

    if args.model_name:
        model_config.put("name", value=args.model_name)
    model_config.get("name", default="np_model_{}".format(int(time.time())))

    if args.model_path:
        model_config.put("path", value=args.model_path)
    model_config.get("path",
                     default=os.path.join(os.path.expanduser("~"), ".smartchem", model_config.get("name")))

    model_config.get("smiles_column", default="smiles")

    fcn_model_config = model_config.getsubdict("fcn_model")
    fcn_model_config.get("layer", default=[])

    model_config.get("loss_function", "name", default="RMAE")
    model_config.get("loss_function", "kwargs", default={})

    model_config.get("featurizer", default=["atom_type_one_hot",
                                            "atom_degree_one_hot",
                                            "atom_implicit_valence_one_hot",
                                            "atom_formal_charge",
                                            "atom_num_radical_electrons",
                                            "atom_hybridization_one_hot",
                                            "atom_is_aromatic",
                                            "atom_total_num_H_one_hot"])

    model_config.get("gcn_output_dims", default=100)
    model_config.get("additional_input_names", default=[])
    model_config.get("gcn_layer_sizes", default=[16, 16, 4, 4, 2, 2])
    model_config.get("task_names", default=['z_average'])
    model_config.get("learning_rate", default=0.0001)
    model_config.get('metrics', default=["relmae"])

    train_config.get("batch_size", default=32)
    train_config.get("epochs", default=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nanoparticle size prediction model')

    parser.add_argument('--train', type=str,
                        help='train or predict')

    parser.add_argument('--predict', type=str,
                        help='train or predict')

    parser.add_argument('--config', type=str, default="config.json",
                        help='path to the config file, default "./config.json"')

    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--model_path', type=str, help='model name')

    parser.add_argument('--model', help='print model' , action="store_true")
    parser.add_argument('--force_new', help='dont load model model' , action="store_true")

    args = parser.parse_args()

    config_file = args.config
    config = JsonDict(config_file)

    update_config(config, args)

    model = load_model(config.getsubdict("model"),load = not args.force_new)

    if args.train:
        train(model, pd.read_csv(args.train), config.getsubdict("training"))
    if args.predict:
        predict(model, pd.read_csv(args.predict))

    if args.model:
        print(model.module)
