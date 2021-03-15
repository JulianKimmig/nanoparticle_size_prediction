import argparse
import os
from datetime import date, datetime
from typing import List

import torch
from torch import nn
from torch.nn.modules import Flatten

from json_dict import JsonDict
import time

import pandas as pd
import numpy as np


from featurizer import load_featurizer
from model.torch_model import PytorchModel
import model.loss as model_loss
from model import callbacks

DEFAULT_DEVICE="cpu"

def generate_fcn_model(config, input_dims):
    def load_layer(layer_config):
        module = getattr(torch.nn, layer_config["module"])
        return module(*layer_config.get("args", []), **layer_config.get("kwargs", {}))

    layer: List[nn.Module] = [load_layer(l) for l in config.get("layer")]

    if len(input_dims) > 1:
        layer.append(Flatten())

    fcn_model = nn.Sequential(*layer)

    return fcn_model


def load_model(config, load=True, device=DEFAULT_DEVICE):
    backend = config.get("backend")
    if backend == "pytorch_geometric":
        import graph_backends.pytorch_geometric_based as backend
    else:
        raise NotImplementedError("cannot load backend '{}'".format(backend))

    featurizer = load_featurizer(config.get("featurizer"))

    fcn_model_config = config.getsubdict("fcn_model")
    post_input_module = generate_fcn_model(
        fcn_model_config,
        input_dims=(
            int(config.get("gcn_output_dims"))
            + len(config.get("additional_input_names")),
        ),
    )

    model = PytorchModel(
        backend.model.GCNMultiInputPredictor(
            in_feats=len(featurizer),
            additional_inputs=len(config.get("additional_input_names")),
            hidden_graph_output=int(config.get("gcn_output_dims")),
            hidden_feats=config.get("gcn_layer_sizes"),
            post_input_module=post_input_module,
            n_tasks=len(config.get("task_names")),
            pooling=config.get("pooling"),
        ),
        predict_function=backend.model.GCNMultiInputPredictor.predict_function,
        batch_data_converter=backend.model.GCNMultiInputPredictor.batch_data_converter,
        name=config.get("name"),
        dir=os.path.abspath(config.get("path")),
    )
    model.gcn_featurizer = featurizer
    model.config = config

    optimizer = torch.optim.Adam(
        model.module.parameters(), lr=config.get("learning_rate")
    )

    loss_fn = getattr(model_loss, config.get("loss_function", "name"))(
        **config.get("loss_function", "kwargs",as_json_dict=False)
    )

    model.compile(optimizer, loss_fn, metrics=config.get("metrics"))

    if load:
        try:
            print("try loading model from {}".format(model.default_filename()))
            model.load(device=device)
            print("load successful")
        except FileNotFoundError as e:
            print("new model, please train me!")
        except Exception as e:
            raise Exception("Model corrupt!, please use the --force_new attribute")

    return model


def update_config(config, args):
    model_config = config.getsubdict("model")
    train_config = config.getsubdict("training")

    model_config.get(
        "name", default=os.path.basename(os.path.dirname(os.path.abspath(config.file)))
    )

    print("Model name:", model_config.get("name"))
    if args.model_path:
        model_config.put("path", value=args.model_path)
    model_config.get("path", default=os.path.dirname(os.path.abspath(config.file)))

    print("Model name:", model_config.get("path"))

    model_config.get("smiles_column", default="smiles")

    fcn_model_config = model_config.getsubdict("fcn_model")
    fcn_model_config.get("layer", default=[])

    model_config.get("loss_function", "name", default="RMAE")
    model_config.get("loss_function", "kwargs", default={})

    model_config.get(
        "featurizer",
        default=[
            "atom_symbol_one_hot",
            "atom_degree_one_hot",
            "atom_implicit_valence_one_hot",
            "atom_formal_charge",
            "atom_num_radical_electrons",
            "atom_hybridization_one_hot",
            "atom_is_aromatic",
            "atom_total_num_H_one_hot",
        ],
    )

    model_config.get("gcn_output_dims", default=100)
    model_config.get("backend", default="pytorch_geometric")
    model_config.get("additional_input_names", default=[])
    model_config.get("gcn_layer_sizes", default=[None, None, None, None, None, None])
    model_config.get("task_names", default=["z_average"])
    model_config.get("learning_rate", default=0.0001)
    model_config.get("metrics", default=["MAPE"])
    model_config.get("pooling", default=["weight_sum", "max"]),
    train_config.get("batch_size", default=32)
    train_config.get("epochs", default=10)
    train_config.get("callbacks", default=[])


def train(model, training_data, train_config, verbose=True):
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
    assert smiles_col in df.columns, "smiles column '{}' not in dataframe".format(
        smiles_col
    )

    cache_file_path = os.path.join(model.dir, "training", "data")
    os.makedirs(cache_file_path, exist_ok=True)

    backend = model_config.get("backend")
    if backend == "pytorch_geometric":
        import graph_backends.pytorch_geometric_based as backend

        smiles_to_graph_data = backend.data.smile_mol_featurizer(
            atom_featurizer=model.gcn_featurizer
        )
    else:
        raise NotImplementedError("cannot load backend '{}'".format(backend))

    train_set, val_set, test_set = backend.data.dataframe_to_data_train_sets(
        df=df,
        to_graph=smiles_to_graph_data,
        graph_columns=[smiles_col],
        task_cols=tasks,
        add_kwargs={"additional_input": add_input_cols},
        seed=config.get("random_seed", default=np.random.randint(2 ** 32)),
        split=[8, 1, 1],
        verbose=verbose,
    )
    # dataset.random_split(frac_train=0.8, frac_val=0.1, frac_test=0.1,#seed=config.get("random_seed"))
    # train_set, val_set, test_set = RandomSplitter.train_val_test_split(dataset,random_state=config.get("random_seed"))

    print(train_config)
    cbs = train_config.get("callbacks").copy()
    for cb in cbs:
        for key, val in cb.get("kwargs", {}).items():
            if val == "validation_loader":
                cb["kwargs"][key] = val_set
            if val == "training_loader":
                cb["kwargs"][key] = train_set

        for i, arg in enumerate(cb.get("args", [])):
            if arg == "validation_loader":
                cb["args"][i] = val_set
            if arg == "training_loader":
                cb["args"][i] = train_set
    train_config.save()
    cbs = [callbacks.deserialize(cb) for cb in cbs]
    model.train(
        data_loader=train_set,
        validation_loader=val_set,
        test_loader=test_set,
        epochs=train_config.get("epochs"),
        callbacks=cbs,
    )
    model.save()


def predict(model, dataframe, return_array=False, verbose=True):
    model_config = model.config
    df = dataframe.copy()
    df.reset_index(inplace=True)
    tasks = ["predicted_" + t for t in model_config.get("task_names")]

    add_input_cols = model_config.get("additional_input_names")
    missing_inputs = list(set(add_input_cols) - set(df.columns))
    if len(missing_inputs) > 0:
        raise ValueError("Missing input columns: {}".format(",".join(missing_inputs)))

    smiles_col = model_config.get("smiles_column")
    assert smiles_col in df.columns, "smiles column '{}' not in dataframe".format(
        smiles_col
    )

    backend = model_config.get("backend")
    if backend == "pytorch_geometric":
        import graph_backends.pytorch_geometric_based as backend

        smiles_to_graph_data = backend.data.smile_mol_featurizer(
            atom_featurizer=model.gcn_featurizer
        )
    else:
        raise NotImplementedError("cannot load backend '{}'".format(backend))
    predict_set = backend.data.dataframe_to_data_predict_set(
        df=df,
        to_graph=smiles_to_graph_data,
        graph_columns=[smiles_col],
        add_kwargs={"additional_input": list(add_input_cols)},
        verbose=verbose,
    )
    #
    for t in tasks:
        if t not in df.columns:
            df[t] = np.nan

    for batch_data in predict_set:
        X, y = model.batch_data_converter(batch_data)
        pred = model.predict(X)
        df.loc[y, tasks] = pred

    if return_array:
        return df[tasks].values
    return df


if __name__ == "__main__":
    print("RUN")
    parser = argparse.ArgumentParser(description="Nanoparticle size prediction model")

    parser.add_argument("--train", type=str, help="train or predict")

    parser.add_argument("--predict", type=str, help="train or predict")

    def_conf = os.path.join(
        os.path.expanduser("~"),
        ".smartchem",
        "np_model_{}".format(int(time.time())),
        "config.json",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help='path to the config file, default "{}"'.format(
            os.path.join(
                os.path.expanduser("~"),
                ".smartchem",
                "np_model_{int(time.time())}",
                "config.json",
            )
        ),
    )

    parser.add_argument("--model_path", type=str, help="model path")

    parser.add_argument("--model", help="print model", action="store_true")
    parser.add_argument(
        "--force_new", help="dont load model model", action="store_true"
    )

    args = parser.parse_args()

    config_file = args.config

    model_path = args.model_path

    if config_file is None:
        if model_path is not None:
            config_file = os.path.join(model_path, "config.json")
        else:
            config_file = def_conf
    config_file = os.path.abspath(config_file)
    config = JsonDict(config_file,default_as_json_dict=False)
    print("Using config:", config.file)

    update_config(config, args)

    model = load_model(config.getsubdict("model"), load=not args.force_new)

    if args.train:
        train_dict = config.getsubdict("training")
        training_history = train_dict.get("training_history", default=[])
        if args.train == "__last":
            assert (
                len(training_history) > 0
            ), "if train on __last the history must not be empty!"
            args.train = training_history[-1]["args"]["train"]
        training_history.append(
            {"args": vars(args), "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        )
        train_dict.put("training_history", value=training_history)
        train(model, pd.read_csv(args.train), config.getsubdict("training"))
    if args.predict:
        predict_df = predict(model, pd.read_csv(args.predict))
        predict_df.to_csv(args.predict.rsplit(".", maxsplit=1)[0] + "_prediction.csv")

    if args.model:
        print(model.module)
