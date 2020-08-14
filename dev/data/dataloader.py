import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, collate_function=None):
        self._shuffle = shuffle
        self.batch_size = batch_size
        if collate_function is None:
            collate_function = dataset.collate

        self.collate_function = collate_function
        self.dataset = dataset

        self.indices = np.arange(len(dataset), dtype=int)

        if self.shuffle:
            self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.indices) / self.batch_size)

    def __getitem__(self, item):
        return self.collate_function(*self.dataset[self.indices[item:item + self.batch_size]])
