from abc import ABC
from typing import Iterable

import numpy as np
import torch
from google.cloud import storage
from torch.utils.data import DataLoader, Dataset


def load_perms(project, bucket_name, path, num, start_point=0):
    client = storage.Client(project)
    bucket = client.get_bucket(bucket_name)
    
    with bucket.blob(path).open('r') as f:
        perms = np.loadtxt(f, delimiter=',')
    perms = perms.reshape((-1, 7, 2))
    assert num + start_point < len(perms), f"Dataset has shape {perms.shape}. Received request for " \
                                           f"{num + start_point} data points."
    return perms[start_point: start_point + num]


def create_samples(perm_unit, cached_perm={}):
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
    def __init__(self, size, start_point=0, project="", bucket_name="", path=""):
        super().__init__()
        self.perms = load_perms(project, bucket_name, path, size, start_point)
        self.cached_perms = {}
    
    def __len__(self):
        return len(self.perms)
    
    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        
        if isinstance(item, slice) or isinstance(item, Iterable):
            samples = []
            for perm in self.perms[item]:
                sample = create_samples(perm, self.cached_perms)
                samples.append(sample)
            return np.array(samples)
        else:
            sample = create_samples(self.perms[item], self.cached_perms)
        return torch.from_numpy(sample)


def loader(project, bucket_name, path, num_envs, batch_size, start_point=0, workers=0):
    batch_size = int(batch_size)
    dataset = EnvDataset(num_envs, start_point, project=project, bucket_name=bucket_name,
                         path=path)
    if batch_size > 1:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True)
    else:
        dataloader = DataLoader(dataset)
    return dataloader


if __name__ == '__main__':
    data = loader(300, 100, 0)
    for n, (inp, ref) in enumerate(data):
        print(inp)
