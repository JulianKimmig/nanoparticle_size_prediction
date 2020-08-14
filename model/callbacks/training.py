import io
import json
import logging
import os
import pickle
import shutil
import time
from copy import deepcopy
from datetime import datetime

import imageio as imageio
from PIL import Image

from os import path as osp

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from .basic import PytorchCallback

try:
    import seaborn as sns

    sns.set(color_codes=True)
except:
    pass

logger = logging.getLogger(__name__)


class ModelCheckpointCallback(PytorchCallback):
    def __init__(self, n_checkpoints=3, path=None, basename=None, reload=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_checkpoints = n_checkpoints
        if basename is None:
            dt = datetime.now()
            basename = f'checkpoint_{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}.pth'
        self.basename = basename
        self._dir = path
        self._saves = 0
        self.files = []

    @property
    def dir(self):
        if self._dir:
            return os.path.join(self._dir)
        return os.path.join(self.model.dir, "training", "checkpoints")

    @property
    def filename(self):
        if self._dir:
            return osp.join(self._dir, self.basename)
        return osp.join(self.model.dir, self.basename)

    def execute(self, epoch, log):
        self._saves += 1
        filename = self.filename + str(self._saves)
        self.model.save(filename)
        self.files.append(filename)
        while len(self.files) > self.n_checkpoints:
            os.remove(self.files[0])
            self.files = self.files[1:]


class EarlyStoppingCallback(PytorchCallback):
    def __init__(self, mode='lower', patience=50, path=None, basename=None, metric=None, load_best=True,
                 delete_after_load=True):
        super().__init__()
        self.delete_after_load = delete_after_load
        self.load_best = load_best
        self.metric = metric
        self.best_epoch = 0
        if basename is None:
            dt = datetime.now()
            basename = f'early_stop_{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}.pth'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = lambda score: score > self.best_score
        else:
            self._check = lambda score: score < self.best_score

        self.patience = patience
        self.counter = 0
        self.basename = basename
        self._dir = path
        self.best_score = None

    @property
    def filename(self):
        if self._dir:
            return osp.join(self._dir, self.basename)
        return osp.join(self.model.dir, self.basename)

    def execute(self, epoch, log):
        metric = self.metric
        if metric is None:
            metric = self.model.metrics[0]

        try:
            score = log["metrics"]["validation"][epoch][metric]
        except:
            try:
                score = log["metrics"]["training"][epoch][metric]
            except:
                return

        if self.best_score is None:
            self.best_score = score
            self.model.save(self.filename)
            self.best_epoch = epoch

        elif self._check(score):
            self.best_score = score
            self.best_epoch = epoch
            self.model.save(self.filename)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.model.stop_training()
            if self.load_best:
                print(f'Training stopped early, load best from epoch {self.best_epoch}')
                self.model.load(self.filename)
                if self.delete_after_load:
                    os.remove(self.filename)
            else:
                print('Training stopped early')


class ValidationPointRecorderCallback(PytorchCallback):
    def __init__(self, data_loader, path=None, max_points=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dir = path
        self.data_loader = data_loader

        if max_points is None:
            max_points = data_loader.batch_size * len(data_loader)
        batches = max(1, int(max_points / data_loader.batch_size))
        self.indices = np.round(np.linspace(0, len(data_loader) - 1, batches)).astype(int)
        self.record_data = None

    @property
    def dir(self):
        if self._dir:
            return os.path.join(self._dir)
        return os.path.join(self.model.dir, "training")

    def _load(self):
        self.record_data = []
        if os.path.exists(os.path.join(self.dir, f"{self.name}.lst")):
            try:
                self.record_data = json.load(open(os.path.join(self.dir, f"{self.name}.lst"), "r"))
            except Exception as e:
                logger.exception(e)

    def animate(self, interpolate=False):
        if self.record_data is None:
            self._load()

        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        scat = ax.scatter([], [], alpha=0.5, edgecolors='b')
        fps = 60
        val_per_sec = 2
        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        valid = [-1, -1, -1]

        indices = np.array([i for i in range(len(self.record_data)) if len(self.record_data[i][0]) > 0])
        # print(indices)

        t_l = int(len(indices) * fps / val_per_sec)
        indices = np.interp(np.linspace(indices.min(), indices.max(), t_l),
                            np.linspace(indices.min(), indices.max(), len(indices)), indices)
        #  print(indices)
        # ni = 0
        # while ni < len(self.record_data):
        #     indices.append(ni)
        #     ni += self.every_n_epoch_func(ni)

        lendata = indices.max()
        loge = f"0{int(np.log10(lendata)) + 4}.2f"

        print(f"{self.name}:")
        images = []

        def update(i):
            # print(i)
            if i % 1 == 0:
                i = int(i)
                if len(self.record_data[i][0]) > 0:
                    valid[0] = i
            if valid[0] < 0:
                return

            if valid[1] < i:
                while valid[1] < i or len(self.record_data[valid[1]][0]) == 0:
                    valid[1] += 1
                    if valid[1] >= len(self.record_data):
                        valid[1] = -1
                        break

            x = np.array(self.record_data[valid[0]][0])
            y = np.array(self.record_data[valid[0]][1])
            if valid[2] < 0:
                truerange = np.array([x.min(), x.max()])
                ax.set_xlim(np.array([-1, 1]) * 1.1 / 2 * np.diff(truerange) + np.mean(truerange))
                ax.set_ylim(ax.get_xlim())
                line.set_ydata(truerange)
                line.set_xdata(truerange)
                valid[2] = 0

            if valid[0] < i:
                if valid[1] > i:
                    nx = np.array(self.record_data[valid[1]][0])
                    ny = np.array(self.record_data[valid[1]][1])
                    fac = ((i - valid[0]) / (valid[1] - valid[0]))
                    x += (nx - x) * fac
                    y += (ny - y) * fac
            scat.set_offsets(np.c_[x, y])
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            images.append(deepcopy(Image.open(buf)))
            buf.close()
            print(f"\r  animate {format(i, loge)}/{lendata}[{('=' * int(100 * (i / lendata)) + ' ' * 100)[:100]}]",
                  end="")

            return (scat, line)

        for i in indices:
            update(i)
        time.sleep(0.05)
        print(f"\r  animate {format(lendata / 1.0, loge)}/{lendata}[{'=' * 100}]")
        print("saving...", end="", flush=True)
        imageio.mimsave(os.path.join(self.dir, f"{self.name}.gif"), images, fps=fps)
        print("done")
        plt.close()

    def execute(self, epoch, log):
        if self.record_data is None:
            self._load()
        self.record_data = self.record_data + [[[], []]] * max(0, ((epoch + 1) - len(self.record_data)))
        self.record_data = self.record_data[:epoch + 1]
        y_true = []
        y_pred = []
        idx_to_go = list(self.indices)
        for i, batch_data in enumerate(self.data_loader):
            if i in idx_to_go:
                idx_to_go.remove(i)
                if self.model.batch_data_converter:
                    X, y = self.model.batch_data_converter(batch_data)
                else:
                    X, y = batch_data
                prediction = self.model.predict(X)
                y_true.append(np.array(y))
                y_pred.append(np.array(prediction))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        self.record_data[epoch] = ([y_true.tolist(), y_pred.tolist()])

        os.makedirs(self.dir, exist_ok=True)
        json.dump(self.record_data, open(os.path.join(self.dir, f"{self.name}.lst"), "w+"))


class MetricsCallback(PytorchCallback):
    def __init__(self, path=None, basename=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if basename is None:
            basename = "training_metrics.mtr"
        self.basename = basename
        self._dir = path

    @property
    def dir(self):
        if self._dir:
            return os.path.join(self._dir)
        return os.path.join(self.model.dir, "training")

    @property
    def file(self):
        return osp.join(self.dir, self.basename)

    def execute(self, epoch, log):
        data = {}
        try:
            if os.path.exists(self.file):
                data = pickle.load(open(self.file, "rb"))
        except:
            shutil.copy(self.file, self.file + "." + str(int(time.time())))

        for m, mdata in log.get("metrics").items():
            if m not in data:
                data[m] = mdata
            else:
                data[m].update(mdata)
        pickle.dump(data, open(self.file, "w+b"))


class ModelSaveCallback(PytorchCallback):
    def execute(self, epoch, log):
        self.model.save()
