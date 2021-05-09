import argparse
import os
from abc import ABC

import numpy as np
import torch
from google.cloud import storage
from torch.utils.data import DataLoader, Dataset

import MPNet.enet.data_loader as ae_dl
from MPNet.enet.CAE import ContractiveAutoEncoder

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


def list_gcs_directories(client, bucket, prefix):
    # from https://github.com/GoogleCloudPlatform/google-cloud-python/issues/920
    iterator = client.list_blobs(bucket, prefix=prefix, delimiter='/')
    prefixes = set()
    for page in iterator.pages:
        prefixes.update(page.prefixes)
    return prefixes


class PNetDataset(Dataset, ABC):
    def __init__(self, cae, folder, qtt_envs, envs_start_idx=0, project=None, bucket=None, env_path=None):
        super().__init__()
        client = storage.Client(project)
        bucket = client.get_bucket(bucket)
        self.env_start_idx = envs_start_idx
        envs = ae_dl.load_perms(project, bucket, env_path, qtt_envs, envs_start_idx)
        sampled_envs = map(ae_dl.create_samples, envs)
        sampled_envs = map(torch.from_numpy, sampled_envs)
        self.cae_envs = list(map(cae, sampled_envs))
        self.path_files = []
        
        envs = list_gcs_directories(client, bucket, folder if "/" in folder else f"{folder}/")
        
        envs = sorted(envs, key=lambda x: int(''.join(filter(str.isdigit, x))))
        envs = envs[envs_start_idx:qtt_envs]
        blob = bucket.get_blob(f"{folder}/envs.npz")
        with blob.open('rb') as f:
            env_paths_file = np.load(f)
            env_paths_array = np.array(np.split(env_paths_file['array'], env_paths_file['index'], axis=0))
            env_indexes = env_paths_file['envs']
        for env_name in envs:
            env_idx = int(''.join(filter(str.isdigit, env_name)))
            arrays_indexes = np.where(env_indexes == env_name[:-1])
            for arr in env_paths_array[arrays_indexes]:
                path = arr.reshape((-1, 2))
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


def loader(enet, paths_folder, qtt_envs, envs_start_idx, batch_size, project, bucket, path, get_dataset=False, *args,
           **kwargs):
    if isinstance(enet, str):
        client = storage.Client(project)
        bucket = client.get_bucket(bucket)
        with bucket.blob(enet).open('rb') as f:
            enet = ContractiveAutoEncoder.load_from_checkpoint(f)
            enet.freeze()
    
    ds = PNetDataset(enet, paths_folder, qtt_envs, envs_start_idx, project, bucket, path)
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
