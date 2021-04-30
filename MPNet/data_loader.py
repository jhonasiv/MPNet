import os
from abc import ABC

import torch
from torch.utils.data import IterableDataset

import AE.data_loader as ae_dl
from AE.CAE import ContractiveAutoEncoder

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def load_envs(folder, qtt):
    envs = ae_dl.load_perms(qtt, 0)
    env_names = os.listdir(f"{project_path}/{folder}")
    env_names = [name for name in env_names if int(name.split('e')[-1]) < qtt]
    
    return envs, env_names


def get_path_files(folder, env_name):
    path_files = os.listdir(f"{project_path}")

def join_path_files(path_files, env_names):
    pass


def process_data_files(envs, path_files, cae):
    sample = ae_dl.create_samples(envs[0], {})
    sample = torch.tensor(sample).float()
    cae = ContractiveAutoEncoder.load_from_checkpoint(f"{project_path}/models/cae.ckpt")
    embedding = cae(sample)
    a = 0


class MPNetDataLoader(IterableDataset, ABC):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, item):
        pass


if __name__ == '__main__':
    env, env_names = load_envs('env', 10)
    process_data_files(env, None, None)
