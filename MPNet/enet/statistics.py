import argparse
import json
import os
import warnings
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from torch import multiprocessing as mp
from torch import nn
from torch.optim import Adagrad

from MPNet.enet.CAE import ContractiveAutoEncoder
from data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class TrainingDataCallback(pl.Callback):
    def __init__(self, log_file: str, log_stats):
        super().__init__()
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
        
        with open(self.log_file, "w") as f:
            json.dump(self.stats, f)
    
    def on_validation_end(self, trainer, pl_module: LightningModule) -> None:
        for key, val in self.stats.items():
            self.stats[key].append(trainer.logged_metrics[key].item())


def train(args):
    configs = [{"l1_units" : 512, "l2_units": 256, "l3_units": 128, "lambda": 1e-3, "actv": nn.PReLU, "lr": 1e-2,
                "optimizer": Adagrad},
               {"l1_units" : 560, "l2_units": 304, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU, "lr": 1e-2,
                "optimizer": Adagrad},
               {"l1_units" : 560, "l2_units": 328, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU, "lr": 1e-2,
                "optimizer": Adagrad},
               {"l1_units" : 512, "l2_units": 256, "l3_units": 160, "lambda": 1e-5, "actv": nn.SELU, "lr": 1e-2,
                "optimizer": Adagrad},
               {"l1_units" : 576, "l2_units": 328, "l3_units": 176, "lambda": 1e-5, "actv": nn.PReLU, "lr": 1e-2,
                "optimizer": Adagrad},
               ]
    
    training = loader(55000, args.batch_size, 0)
    validation = loader(7500, 1, 55000)
    
    if args.model_id is None:
        for n, config in enumerate(configs):
            iteration_loop(config, n, args.itt, training, validation, args.num_gpus, args.log_path)
    else:
        iteration_loop(configs[args.model_id], args.model_id, args.itt, training, validation, args.num_gpus,
                       args.log_path)


def iteration_loop(config, n, num_itt, training, validation, num_gpus, log_path):
    for itt in range(num_itt):
        pl.seed_everything(itt)
        
        es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
        logging = TrainingDataCallback(f"{log_path}/cae_{n}_{itt}.json",
                                       log_stats=["val_loss", "epoch"])
        trainer = pl.Trainer(gpus=num_gpus, stochastic_weight_avg=True, callbacks=[es, logging],
                             weights_summary=None, deterministic=True, progress_bar_refresh_rate=1)
        cae = ContractiveAutoEncoder(training, validation, config=config, reduce=True, seed=itt)
        
        trainer.fit(cae)


def parallel_main(args):
    configs = [{"l1_units": 512, "l2_units": 256, "l3_units": 128, "lambda": 1e-3, "actv": nn.PReLU},
               {"l1_units": 560, " ""l2_units": 304, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU},
               {"l1_units": 560, " ""l2_units": 328, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU},
               {"l1_units": 512, " ""l2_units": 256, "l3_units": 160, "lambda": 1e-5, "actv": nn.SELU},
               {"l1_units": 576, " ""l2_units": 328, "l3_units": 176, "lambda": 1e-5, "actv": nn.PReLU},
               ]
    
    torch.set_num_interop_threads(1)
    processes = []
    for n, config in enumerate(configs):
        for itt in range(args.itt):
            p = mp.Process(target=worker,
                           args=(
                                   config, n, itt, args.num_gpus, args.log_path))
            p.start()
            processes.append(p)
            while len(processes) == args.workers:
                for proc in processes:
                    proc.join(.1)
                    if proc.exitcode == 0:
                        processes.remove(proc)
    for proc in processes:
        proc.join()


def worker(config, idx, itt, num_gpus, log_path):
    print(f"Starting worker for config {idx} -> iteration {itt}")
    
    torch.set_num_threads(1)
    training = loader(55000, args.batch_size, 0)
    validation = loader(7500, 1, 55000)
    
    es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, mode='min', verbose=True)
    logging = TrainingDataCallback(f"{log_path}/cae_{idx}_{itt}.json", log_stats=["val_loss", "epoch"])
    trainer = pl.Trainer(gpus=num_gpus, stochastic_weight_avg=True, callbacks=[es, logging],
                         progress_bar_refresh_rate=0, weights_summary=None, deterministic=True)
    cae = ContractiveAutoEncoder(training, validation, config=config, reduce=True, seed=itt)
    
    trainer.fit(cae)
    print(f"\nWorker done for config {idx} -> iteration {itt}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--itt', type=int, default=20)
    parser.add_argument('--model_id', default=None, type=int)
    parser.add_argument('--log_path', default="data", type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    
    args = parser.parse_args()
    if args.workers > 0 and args.model_id is None:
        parallel_main(args)
    else:
        train(args)
