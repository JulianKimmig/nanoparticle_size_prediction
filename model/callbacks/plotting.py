import glob
import os

import imageio
import torch


import matplotlib.pyplot as plt
import numpy as np

from .basic import PytorchCallback

import seaborn as sns
sns.set(color_codes=True)



class PlotValidationCallback(PytorchCallback):
    def __init__(self, data_loader, path=None, max_points=None,
                 title="correlation true/predicted for {model} in epoch {epoch}", animate=True,normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animate = animate
        self.title = title
        self._dir = path
        self.data_loader = data_loader
        self._images = []
        self.normalize = normalize

        if max_points is None:
            max_points = data_loader.batch_size * len(data_loader)
        batches = max(1, int(max_points / data_loader.batch_size))
        self.indices = np.round(np.linspace(0, len(data_loader) - 1, batches)).astype(int)

    @property
    def dir(self):
        if self._dir:
            return os.path.join(self._dir)
        return os.path.join(self.model.dir, "training", "plot_validation")

    def execute(self, epoch, log):

        y_true = []
        y_pred = []
        idx_to_go = list(self.indices)
        for i, batch_data in enumerate(self.data_loader):
            if i in idx_to_go:
                if self.model.batch_data_converter:
                    X, y = self.model.batch_data_converter(batch_data)
                else:
                    X, y = batch_data
                prediction = self.model.predict(X)
                y_true.append(np.array(y))
                y_pred.append(np.array(prediction))
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        y_true_min = np.min(y_true, axis=0)[0]*(1-10**-5)
        y_true_max = np.max(y_true, axis=0)[0]
        if self.normalize:
            y_true = (y_true - y_true_min) / (y_true_max - y_true_min)

        y_pred_min = np.min(y_pred, axis=0)[0]*(1-10**-8)
        y_pred_max = np.max(y_pred, axis=0)[0]
        if self.normalize:
            y_pred = (y_pred - y_pred_min) / (y_pred_max - y_pred_min)
        bins = 50



        if self.normalize:
            prange = (0,1)
        else:
            prange = (y_true_min-0.05*(y_true_max-y_true_min),y_true_max+0.05*(y_true_max-y_true_min))

        y_true_plot = y_true
        y_pred_plot = y_pred
      #  print(y_pred_plot.shape)
        def customJoint(x, y):
            plt.plot(prange, prange,'--', alpha=0.3)
            label = None
            if len(y_pred_plot.shape)>1:
                for i in range(y_true_plot.shape[1]):
                    if self.normalize:
                        label = f"(out{i}-{y_true_min[i]:.2e})/({y_true_max[i]:.2e}-{y_true_min[i]:.2e})"
                    pltable=(y_pred_plot[:, i]>=prange[0])&(y_pred_plot[:, i]<=prange[1])
                    plt.scatter(y_true_plot[:, i][pltable], y_pred_plot[:, i][pltable],
                                alpha=0.6,
                                label=label)

        def customMarginal(x, vertical=True, *args, **kwargs):
            if vertical:
                ar = y_pred_plot
                barf = plt.barh
            else:
                ar = y_true
                barf = plt.bar
            #print(ar)
            if len(ar.shape)>1:
                for i in range(ar.shape[1]):
                    h = np.histogram(ar[:, i], bins,range=prange)
           #     print(ar[:, i])
           #     print(prange)
           #     print(h)
           #     print(h[1][:-1] + (prange[1]-prange[0]) / (2 * bins))
           #     print( h[0] / h[0].max())
            #    print((prange[1]-prange[0]) / bins)
                    hist1 = barf(h[1][:-1] + (prange[1]-prange[0]) / (2 * bins), h[0] / h[0].max(), (prange[1]-prange[0]) / bins)

        g = sns.JointGrid(x=[0], y=[0])
        g.plot(customJoint, customMarginal)
        if self.normalize:
            plt.legend(loc='lower right')

        #  truerange=np.array([y_true.min(), y_true.max()])
        #  plt.plot(truerange, truerange, alpha=0.5)
        #   plt.scatter(y_true, y_pred,edgecolors='b', alpha=0.5)
        #  plt.xlim(np.array([-1,1])*1.1/2*np.diff(truerange) + np.mean(truerange))
        #  plt.ylim(plt.xlim())
        s = [""]
        i = 0
        max_length = 30
        title = self.title.replace("{epoch}", str(epoch)).replace("{model}", str(self.model.name))

        for word in title.split():
            if len(s[i]) == 0:
                s[i] += word
            elif len(s[i]) + len(word) > max_length:
                s.append(word)
                i += 1
            else:
                s[i] += " " + word
        g.fig.suptitle("\n".join(s))
        g.fig.subplots_adjust(top=0.95-(0.035*len(s)))

        plt.xlabel("true")
        plt.ylabel("predicted")

        os.makedirs(self.dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, f"{epoch}.png"))
        plt.close()
        if self.animate:
            if len(self._images) == 0:
                available_epoch_images = []
                for f in os.listdir(self.dir):
                    try:
                        if f.endswith(".png"):
                            i = int(f.replace(".png", ""))
                            if i < epoch:
                                available_epoch_images.append(i)
                    except:
                        pass
                self._images.extend(
                    [imageio.imread(os.path.join(self.dir, f"{i}.png")) for i in sorted(available_epoch_images)])
            self._images.append(imageio.imread(os.path.join(self.dir, f"{epoch}.png")))
            imageio.mimsave(os.path.join(self.dir, f"{self.name}.gif"), self._images, fps=5)
