import os
import time

import torch

import numpy as np
from pint import UnitRegistry

from model.metrics.meter import Meter

ureg = UnitRegistry()


class PytorchModel:
    def __init__(
        self,
        module,
        predict_function=None,
        dir=None,
        name=None,
        batch_data_converter=None,
    ):
        # privat
        self._training = False

        # public
        if predict_function is None:
            predict_function = module
        self.batch_data_converter = batch_data_converter
        self.predict_function = predict_function
        self.module = module
        self.metrics = []
        self.log = {}

        # blank properties
        self._dir = None
        self._name = None
        # set properties
        if dir:
            self.dir = dir
        if name:
            self.name = name

        self.save_last_prediction_tensor = False
        self.save_last_input_data = False
        self.last_input_data = None
        self.last_prediction_tensor = None

    def get_dir(self):
        if self._dir is None:
            raise ValueError(
                "Model has not dir please set via model.dir=<dir> "
                "or fall back to default by providing"
                f"(models/<model.name>) a model name"
            )
        return self._dir

    def set_dir(self, dir):
        self._dir = dir

    dir = property(get_dir, set_dir)

    def get_name(self):
        return self._name

    def set_name(self, name):
        self._name = name
        if self._dir is None:
            self.dir = os.path.join("models", self._name)

    name = property(get_name, set_name)

    def compile(self, optimizer, loss_fn, metrics="rmse", device=None):
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics = metrics
        self.module.to(self.device)

    def default_filename(self):
        return os.path.join(self.dir, self.name + ".pth")

    def save(self, filename=None):
        if filename is None:
            filename = self.default_filename()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.module.state_dict(),
                "model_log": self.log,
            },
            filename,
        )

    def load(self, filename=None, strict=True):
        if filename is None:
            filename = self.default_filename()
        load = torch.load(filename)
        self.module.load_state_dict(load["model_state_dict"], strict=strict)
        self.log = load["model_log"]
        if hasattr(self, "device"):
            self.module.to(self.device)

    def stop_training(self):
        self._training = False

    def predict(self, data):
        if self.save_last_input_data:
            self.last_input_data = data
        self.module.eval()
        with torch.no_grad():
            prediction = self.predict_function(self.module, data, self.device)
        if self.save_last_prediction_tensor:
            self.last_prediction_tensor = prediction
        return prediction.detach().cpu().numpy()

    def train(
        self,
        data_loader,
        epochs=1,
        validation_loader=None,
        test_loader=None,
        callbacks=None,
        start_epoch=None,
        verbose=True,
        batch_data_converter=None,
    ):
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            callback.model = self

        if batch_data_converter is None:
            batch_data_converter = self.batch_data_converter

        pre_bdt = self.batch_data_converter
        self.batch_data_converter = batch_data_converter

        metrics = self.log.setdefault("metrics", {})
        training_metrics = metrics.setdefault("training", {})
        validation_metrics = metrics.setdefault("validation", {})
        testing_metrics = metrics.setdefault("testing", {})
        epochs_log = metrics.setdefault("epochs", {})

        self._training = True
        self.module.train()

        if start_epoch is None:
            start_epoch = max([0] + list(epochs_log.keys()))

        epoch = start_epoch
        epochs = start_epoch + epochs

        if verbose:
            training_start = time.time()

            def _print_batch_step(
                type,
                batch,
                batch_len,
                len_labels,
                batch_time_start,
                step_time_start,
                meter,
            ):
                current_time = time.time()
                step_time_running = current_time - step_time_start

                batches_done = batch + 1
                batches_to_go = batch_len - batches_done

                batch_time_instance = current_time - batch_time_start
                batch_time_mean = step_time_running / (batch + 1)

                batch_time_to_go_mean = batch_time_mean * (batches_to_go)

                epoch_time_estimation_mean = step_time_running + batch_time_to_go_mean
                time_per_label = batch_time_instance / len_labels

                time_per_label = f"{(time_per_label * ureg.s).to_compact():.3f~}"
                epoch_time_running = f"{(step_time_running * ureg.s).to_compact():.3f~}"
                epoch_time_estimation_mean = (
                    f"{(epoch_time_estimation_mean * ureg.s).to_compact():.3f~}"
                )

                score = {m: np.mean(meter.compute_metric(m)) for m in self.metrics}
                scre_text = "score(s): " + ", ".join(
                    ["{}:{:.2e}".format(k, v) for k, v in score.items()]
                )
                verb_batch_str = f"{type} {batches_done}/{batch_len}[{'=' * int(100 * batches_done / batch_len)}{' ' * int(100 * batches_to_go / batch_len)}], {epoch_time_running}/{epoch_time_estimation_mean},  {time_per_label}/sample, {scre_text}"
                print("\r", verb_batch_str, end=" " * 10)

        for epoch in range(start_epoch, epochs):
            if not self._training:
                break

            epoch_time_start = time.time()
            self.module.train()
            train_meter = Meter()
            if verbose:
                loge = f"0{int(np.log10(epochs)) + 1}d"
                print(f"epoch {format(epoch + 1, loge)}/{epochs}")
                batch_len = len(data_loader)
                step_time_start = time.time()

            for batch_id, batch_data in enumerate(data_loader):
                if verbose:
                    batch_time_start = time.time()
                if self.batch_data_converter:
                    X, y = self.batch_data_converter(batch_data)
                else:
                    X, y = batch_data
                y = y.to(self.device)
                prediction = self.predict_function(self.module, X, self.device)
                # print(prediction.shape)
                y_pred = prediction
                y_true = y
                loss = (self.loss_fn(y_pred, y_true)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_meter.update(prediction, y)

                if verbose:
                    _print_batch_step(
                        type="train",
                        batch=batch_id,
                        batch_len=batch_len,
                        len_labels=len(y),
                        batch_time_start=batch_time_start,
                        step_time_start=step_time_start,
                        meter=train_meter,
                    )

            if verbose:
                print()
            train_score = {
                m: np.mean(train_meter.compute_metric(m)) for m in self.metrics
            }
            training_metrics[epoch] = train_score

            if validation_loader:
                self.module.eval()
                eval_meter = Meter()
                if verbose:
                    batch_len = len(validation_loader)
                    step_time_start = time.time()
                with torch.no_grad():
                    for batch_id, batch_data in enumerate(validation_loader):
                        if self.batch_data_converter:
                            X, y = self.batch_data_converter(batch_data)
                        else:
                            X, y = batch_data
                        prediction = self.predict_function(self.module, X, self.device)
                        eval_meter.update(prediction, y)

                        if verbose:
                            _print_batch_step(
                                type="valid",
                                batch=batch_id,
                                batch_len=batch_len,
                                len_labels=len(y),
                                batch_time_start=batch_time_start,
                                step_time_start=step_time_start,
                                meter=eval_meter,
                            )

                if verbose:
                    print()
                valid_score = {
                    m: np.mean(eval_meter.compute_metric(m)) for m in self.metrics
                }
                validation_metrics[epoch] = valid_score

            # print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            #    epoch + 1, epochs, self.metric, total_score))
            epoch_time = time.time()
            epochs_log[epoch] = epoch_time

            if verbose:
                callback_times = []
            for callback in callbacks:
                if verbose:
                    callback_start = time.time()
                cb_run = callback.step(epoch, self.log, force=(epoch == (epochs - 1)))
                if verbose and cb_run:
                    callback_times.append((callback.name, time.time() - callback_start))

            if verbose:
                ct = time.time()
                epoch_time = ct - epoch_time_start
                lines = []
                lines.append(
                    f"training time: {((ct-training_start) * ureg.s).to_compact():.3f~}"
                )
                lines.append(f"epoch time: {(epoch_time * ureg.s).to_compact():.3f~}")
                if len(callback_times) > 0:
                    lines.append("Callbacks:")
                    for cbt in callback_times:
                        lines.append(
                            f"  {cbt[0]} {(cbt[1] * ureg.s).to_compact():.3f~}"
                        )

                lines_length = max([len(l) for l in lines])
                print("#" * (lines_length + 4))
                for l in lines:
                    print("# ", l, " " * (lines_length - len(l)), " #", sep="")
                print("#" * (lines_length + 4))

        self.module.eval()

        if test_loader is not None:
            eval_meter = Meter()
            with torch.no_grad():
                if verbose:
                    batch_len = len(test_loader)
                    step_time_start = time.time()

                for batch_id, batch_data in enumerate(test_loader):
                    if verbose:
                        batch_time_start = time.time()
                    if self.batch_data_converter:
                        X, y = self.batch_data_converter(batch_data)
                    else:
                        X, y = batch_data
                    prediction = self.predict_function(self.module, X, self.device)
                    eval_meter.update(prediction, y)

                    if verbose:
                        _print_batch_step(
                            type="test",
                            batch=batch_id,
                            batch_len=batch_len,
                            len_labels=len(y),
                            batch_time_start=batch_time_start,
                            step_time_start=step_time_start,
                            meter=eval_meter,
                        )

            test_score = {
                m: np.mean(eval_meter.compute_metric(m)) for m in self.metrics
            }
            if verbose:
                print()
            testing_metrics[epoch] = test_score

        self.batch_data_converter = pre_bdt
        self._training = False
