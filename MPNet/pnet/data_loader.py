import argparse
import os
from abc import ABC

import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

import MPNet.enet.data_loader as ae_dl
from MPNet.enet.CAE import ContractiveAutoEncoder

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class PathToTar:
    @staticmethod
    def _load_envs(folder, qtt):
        envs = ae_dl.load_perms(qtt, 0)
        env_names = os.listdir(f"{project_path}/{folder}")
        env_names = [name for name in env_names if int(name.split('e')[-1]) < qtt]
        
        return envs, env_names
    
    @staticmethod
    def _get_path_files(folder, env_name, qtt):
        path_files = os.listdir(f"{project_path}/{folder}/{env_name}")
        path_files = [file for file in path_files if int(file.split("path")[-1].split('.')[0]) < qtt]
        return path_files
    
    @staticmethod
    def _join_path_files(envs, folder, env_names, paths_per_env, output_path):
        key_id = 0
        sink = wds.TarWriter(output_path, encoder=False, compress=True)
        for env, env_name in zip(envs, env_names):
            env = ae_dl.create_samples(env)
            path_files = PathToTar._get_path_files(folder, env_name, paths_per_env)
            for file in path_files:
                path = np.fromfile(f"{project_path}/{folder}/{env_name}/{file}", dtype=float).reshape((-1, 2))
                for n, point in enumerate(path[:-1]):
                    sample = {"__key__"       : f"data_{key_id}",
                              "env.ten"       : env.tobytes(),
                              "trajectory.ten": np.array([*point, *path[-1]]).tobytes(),
                              "target.ten"    : np.array([*path[n + 1]]).tobytes()}
                    
                    sink.write(sample)
                    key_id += 1
        sink.close()
    
    @staticmethod
    def to_tar(parent_folder, num_envs, paths_per_env, output_path):
        loaded_envs, env_names = PathToTar._load_envs(parent_folder, num_envs)
        PathToTar._join_path_files(loaded_envs, parent_folder, env_names, paths_per_env, output_path)


class FromTar:
    def __init__(self, ae=None):
        self._ae = ae
    
    def decode(self, sample):
        inp = torch.from_numpy(np.frombuffer(sample['trajectory.ten']).reshape((-1,)))
        target = torch.from_numpy(np.frombuffer(sample['target.ten']))
        env = torch.from_numpy(np.frombuffer(sample['env.ten'])).float()
        env = self._ae(env)
        
        resulting_input = torch.cat([env, inp])
        return resulting_input, target
    
    def load_dataset(self, file_path, batch_size=1, num_workers=0, shuffle=0, ae=None):
        self._ae = ae
        dataset = wds.Dataset(file_path)
        dataset = wds.Processor(dataset, wds.map, self.decode).batched(batch_size)
        if shuffle:
            dataset = dataset.shuffle(shuffle)
        dataset = DataLoader(dataset, batch_size=None, num_workers=num_workers)
        return dataset


class PNetDataset(Dataset, ABC):
    def __init__(self, cae, folder, qtt_envs, envs_start_idx=0):
        super().__init__()
        self.env_start_idx = envs_start_idx
        envs = ae_dl.load_perms(qtt_envs, envs_start_idx)
        sampled_envs = map(ae_dl.create_samples, envs)
        sampled_envs = map(torch.from_numpy, sampled_envs)
        self.cae_envs = list(map(cae, sampled_envs))
        self.path_files = []
        for _, envs, _ in os.walk(folder):
            envs = sorted(envs, key=lambda x: int(''.join(filter(str.isdigit, x))))
            envs = envs[envs_start_idx:qtt_envs]
            for env_name in envs:
                env_idx = int(''.join(filter(str.isdigit, env_name)))
                for _, _, files in os.walk(f"{folder}/{env_name}"):
                    for filename in files:
                        path = np.fromfile(f"{folder}/{env_name}/{filename}", dtype=float).reshape((-1, 2))
                        for n in range(len(path) - 1):
                            self.path_files.append((env_idx, path, n))
    
    def process(self, item, extra_info=None):
        embed_idx, path, path_idx = item
        embedding = self.cae_envs[embed_idx]
        inputs = torch.as_tensor([*embedding, *path[path_idx], *path[-1]]).float()
        target = torch.as_tensor([*path[path_idx + 1]]).float()
        # inputs = torch.as_tensor(inputs).float()
        # target = torch.as_tensor(target).float()
        if extra_info:
            return inputs, target, path, embed_idx
        return inputs, target
    
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return self.process(self.path_files[item[0]], extra_info=item[1])
        else:
            return self.process(self.path_files[item])
    
    def __len__(self):
        return len(self.path_files)


def loader(enet, paths_folder, qtt_envs, envs_start_idx, batch_size, get_dataset=False, *args, **kwargs):
    if isinstance(enet, str):
        enet = ContractiveAutoEncoder.load_from_checkpoint(enet)
        enet.freeze()
    
    ds = PNetDataset(enet, paths_folder, qtt_envs, envs_start_idx)
    if get_dataset:
        return ds
    else:
        ds = DataLoader(ds, batch_size=batch_size, *args, **kwargs)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--parent', type=str, default="")
    parser.add_argument('--num_envs', type=int, default=100)
    parser.add_argument('--paths_per_env', type=int, default=4000)
    parser.add_argument('--output_path', type=str, default='')
    
    args = parser.parse_args()
    
    c = ContractiveAutoEncoder.load_from_checkpoint('../../models/cae.ckpt')
    c.freeze()
    ds = PNetDataset(c, '../../env', 100)
    # PathToTar.to_tar(args.parent, args.num_envs, args.paths_per_env, args.output_path)
