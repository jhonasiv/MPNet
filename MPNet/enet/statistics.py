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

from MPNet.enet.CAE import ContractiveAutoEncoder
from data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class TrainingDataCallback(pl.Callback):
    def __init__(self, log_file: str, log_stats: Dict):
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


def main(args):
    configs = [{"l1_units": 512, "l2_units": 256, "l3_units": 128, "lambda": 1e-3, "actv": nn.PReLU},
               {"l1_units": 560, " ""l2_units": 304, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU},
               {"l1_units": 560, " ""l2_units": 328, "l3_units": 208, "lambda": 1e-5, "actv": nn.SELU},
               {"l1_units": 512, " ""l2_units": 256, "l3_units": 160, "lambda": 1e-5, "actv": nn.SELU},
               {"l1_units": 576, " ""l2_units": 328, "l3_units": 176, "lambda": 1e-5, "actv": nn.PReLU},
               ]
    
    training = loader(55000, args.batch_size, 0)
    validation = loader(7500, 1, 55000)
    
    torch.set_num_interop_threads(1)
    processes = []
    for n, config in enumerate(configs):
        for itt in range(args.itt):
            pl.seed_everything(itt)
            p = mp.Process(target=worker, args=(config, n, itt, training, validation, args.num_gpus))
            p.start()
            processes.append(p)
            while len(processes) == args.workers:
                for proc in processes:
                    proc.join(.1)
                    if proc.exitcode == 0:
                        processes.remove(proc)


def worker(config, idx, itt, training, validation, num_gpus):
    print(f"Starting worker for config {idx} -> iteration {itt}")
    torch.set_num_threads(1)
    es = EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=12, mode='min', verbose=True)
    logging = TrainingDataCallback(f"{project_path}/data/cae_{idx}_{itt}.json", log_stats=["val_loss", "epoch"])
    trainer = pl.Trainer(gpus=num_gpus, stochastic_weight_avg=True, callbacks=[es, logging],
                         progress_bar_refresh_rate=0, weights_summary=None, benchmark=True)
    cae = ContractiveAutoEncoder(training, validation, config, reduce=True)
    
    trainer.fit(cae)
    print(f"\nWorker done for config {idx} -> iteration {itt}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--itt', type=int, default=20)
    parser.add_argument('--batch_size', default=100, type=int)
    
    args = parser.parse_args()
    main(args)
