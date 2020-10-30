import numpy as np


class PytorchCallback:
    def __init__(self, every_n_epoch=1, name=None):

        if isinstance(every_n_epoch, int):
            every_n_epoch_func = lambda epoch: every_n_epoch
        elif every_n_epoch == "linear":
            every_n_epoch_func = lambda epoch: epoch
        elif every_n_epoch == "sqrt":
            every_n_epoch_func = np.sqrt
        else:
            if isinstance(every_n_epoch, str):
                raise ValueError("unknown stepcalc '{}'".format(every_n_epoch))
            every_n_epoch_func = every_n_epoch
        self.every_n_epoch_func = every_n_epoch_func
        self._model = None
        self.name = self.__class__.__name__ if name is None else name
        self.next_point = 0

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model

    model = property(get_model, set_model)

    def step(self, epoch, log, force=False):
        while epoch > int(self.next_point):
            self.next_point += max(1, self.every_n_epoch_func(self.next_point))

        if epoch == int(self.next_point) or force:
            self.execute(epoch, log)
            return True
        return False

    def execute(self, epoch, log):
        raise NotImplementedError(f"execute misssing in {self.__class__.__name__}")


class LambdaCallback(PytorchCallback):
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def execute(self, epoch, log):
        self.func(epoch, log)
