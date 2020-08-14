import numpy as np
import torch


class Dataset():
    def __init__(self, input_data, output_data=None):
        if output_data is None:
            output_data = np.array([None] * len(input_data))
        self.output_data = np.array(output_data)
        self.input_data = np.array(input_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, item):
        return self.input_data[item], self.output_data[item]

    def random_split(self, ratios, seed=None):
        ratios = np.array(ratios)
        ratios = (len(self) * ratios / ratios.sum()).round().astype(int)

        indices = np.arange(len(self))

        if seed:
            np.random.seed(seed)

        np.random.shuffle(indices)

        end = np.cumsum(ratios)
        end[-1] = len(self)

        start = end - ratios
        return (DataSubset(self, indices[start[i]:end[i]]) for i in range(len(start)))

    def collate(self,input_data,output_data):
        return input_data, torch.from_numpy(output_data)

class DataSubset:
    def __init__(self, parent, indices):
        self.indices = indices
        self.parent = parent

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.parent[self.indices[item]]

    def collate(self,*args,**kwargs):
        return self.parent.collate(*args,**kwargs)
