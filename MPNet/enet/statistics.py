import argparse
import json
import os
import warnings
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing as mp
from google.cloud import storage
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim import Adagrad

from MPNet.enet.CAE import ContractiveAutoEncoder
from MPNet.enet.data_loader import loader

warnings.filterwarnings("ignore", category=UserWarning)

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class TrainingDataCallback(pl.Callback):
    def __init__(self, project: str, bucket_name: str, log_file: str, log_stats: Dict):
        super().__init__()
        self.project = project
        self.bucket_name = bucket_name
        self.log_file = log_file
        check_for_error = [key for key in log_stats if key not in ("epoch", "val_loss")]
        if check_for_error:
            warnings.warn(
                    f"log_stats variable only accepts epochs and val_loss as parameters, but it received "
                    f"{log_stats}")
        
        self.stats = {key: [] for key in log_stats if key in ("epoch", "val_loss")}
    
    def on_train_end(self, trainer, pl_module: LightningModule) -> None:
        self.stats['val_loss'] = np.array(self.stats["val_loss"])
        self.stats['epoch'] = np.array(self.stats["epoch"])
        min_idx = np.where(self.stats['val_loss'] == min(self.stats['val_loss']))
        self.stats['val_loss'] = self.stats['val_loss'][min_idx][0]
        self.stats['epoch'] = self.stats['epoch'][min_idx][0]
        client = storage.Client(self.project)
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self.log_file)
        blob.upload_from_string(json.dumps(self.stats))
    
    def on_validation_end(self, trainer, pl_module: LightningModule) -> None:
        for key, val in self.stats.items():
            self.stats[key].append(trainer.logged_metrics[key].item())


def train(args):
    configs = [{"l1_units": 512, "l2_units": 256, "l3_units": 128, "lambda": 1e-3, "actv": nn.PReLU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 560, " ""l2_units": 304, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 560, " ""l2_units": 328, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 512, " ""l2_units": 256, "l3_units": 160, "lambda": 1e-5, "actv": nn.SELU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 576, " ""l2_units": 328, "l3_units": 176, "lambda": 1e-5, "actv": nn.PReLU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               ]
    
    training = loader(args.gcloud_project, args.bucket, "obs/perm.csv", 55000, args.batch_size, 0,
                      workers=args.workers)
    validation = loader(args.gcloud_project, args.bucket, "obs/perm.csv", 7500, 1, 55000, workers=args.workers)
    
    if args.model_id is None:
        for n, config in enumerate(configs):
            iteration_loop(config, n, args.itt, training, validation, args.num_gpus, args.gcloud_project,
                           args.bucket, args.log_path)
    else:
        iteration_loop(configs[args.model_id], args.model_id, args.itt, training, validation, args.num_gpus,
                       args.gcloud_project, args.bucket, args.log_path)


def iteration_loop(config, n, num_itt, training, validation, num_gpus, gcloud_project, bucket, log_path):
    for itt in range(num_itt):
        pl.seed_everything(itt)
        
        es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
        logging = TrainingDataCallback(gcloud_project, bucket, f"{log_path}/cae_{n}_{itt}.json",
                                       log_stats=["val_loss", "epoch"])
        trainer = pl.Trainer(gpus=num_gpus, stochastic_weight_avg=True, callbacks=[es, logging],
                             weights_summary=None, deterministic=True, progress_bar_refresh_rate=1)
        cae = ContractiveAutoEncoder(training, validation, config=config, reduce=True, seed=itt)
        
        trainer.fit(cae)


def parallel_main(args):
    configs = [{"l1_units": 512, "l2_units": 256, "l3_units": 128, "lambda": 1e-3, "actv": nn.PReLU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 560, " ""l2_units": 304, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 560, " ""l2_units": 328, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 512, " ""l2_units": 256, "l3_units": 160, "lambda": 1e-5, "actv": nn.SELU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               {"l1_units": 576, " ""l2_units": 328, "l3_units": 176, "lambda": 1e-5, "actv": nn.PReLU,
                "lr":       args.learning_rate, "optimizer": Adagrad},
               ]
    
    torch.set_num_interop_threads(1)
    processes = []
    for n, config in enumerate(configs):
        for itt in range(args.itt):
            p = mp.Process(target=worker,
                           args=(
                               config, n, itt, args.num_gpus, args.log_path, args.gcloud_project, args.bucket))
            p.start()
            processes.append(p)
            while int(len(processes)) == args.workers:
                for proc in processes:
                    proc.join(.1)
                    if proc.exitcode == 0:
                        processes.remove(proc)
    for proc in processes:
        proc.join()


def worker(config, idx, itt, num_gpus, log_path, gcloud_project, bucket):
    print(f"Starting worker for config {idx} -> iteration {itt}")
    
    torch.set_num_threads(1)
    training = loader(args.gcloud_project, args.bucket, "obs/perm.csv", 55000, args.batch_size, 0)
    validation = loader(args.gcloud_project, args.bucket, "obs/perm.csv", 7500, 1, 55000)
    
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
    logging = TrainingDataCallback(gcloud_project, bucket, f"{log_path}/cae_{idx}_{itt}.json",
                                   log_stats=["val_loss", "epoch"])
    trainer = pl.Trainer(gpus=num_gpus, stochastic_weight_avg=True, callbacks=[es, logging],
                         progress_bar_refresh_rate=0, weights_summary=None, deterministic=True)
    cae = ContractiveAutoEncoder(training, validation, config=config, reduce=True, seed=itt)
    
    trainer.fit(cae)
    print(f"\nWorker done for config {idx} -> iteration {itt}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', default="", type=str)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--itt', type=int, default=20)
    parser.add_argument('--gcloud_project', default="", type=str, required=True)
    parser.add_argument('--bucket', default="", type=str, required=True)
    parser.add_argument('--model_id', default=None, type=int)
    parser.add_argument('--log_path', default="data", type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    
    args = parser.parse_args()
    if args.model_id is None:
        parallel_main(args)
    else:
        train(args)
