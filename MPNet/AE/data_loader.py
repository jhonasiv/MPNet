import random
from abc import ABC
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from plotly import graph_objs as go

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def load_perms(num):
    perms = np.loadtxt(f'{project_path}/obs/perm.csv', delimiter=',')
    perms = perms.reshape((-1, 7, 2))
    return perms[:num]


def create_samples(perm_unit, cached_perm):
    samples = []
    for obs in perm_unit:
        if tuple(obs) not in cached_perm.keys():
            sample = np.random.uniform(obs - 2.5, obs + 2.5, (200, 2))
            samples.append(sample)
            cached_perm[tuple(obs)] = sample
        else:
            samples.append(cached_perm[tuple(obs)])
    samples = np.array(samples).flatten()
    return samples


class EnvDataset(Dataset, ABC):
    def __init__(self, size):
        super().__init__()
        self.perms = load_perms(size)
        self.cached_perms = {}
    
    def __len__(self):
        return len(self.perms)
    
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        
        if isinstance(item, slice):
            samples = []
            for perm in self.perms[item]:
                sample = create_samples(perm, self.cached_perms)
                samples.append(sample)
            return np.array(samples)
        else:
            sample = create_samples(self.perms[item], self.cached_perms)
        return torch.from_numpy(sample), torch.from_numpy(sample)


if __name__ == '__main__':
    dataset = EnvDataset(30000)
    x = dataset[540][0]
    y = dataset[500][0]
    for a in [x, y]:
        fig = go.Figure()
        a = a.reshape((-1, 2))
        fig.add_trace(go.Scatter(x=a[:, 0], y=a[:, 1], mode='markers'))
        fig.update_xaxes(range=[-20, 20])
        fig.update_yaxes(range=[-20, 20])
        fig.show()
