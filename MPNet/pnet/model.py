from typing import Any, Dict, List, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adagrad, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

import os
from MPNet.pnet.data_loader import loader

project_path = f"{os.path.abspath(__file__).split('mpnet')[0]}mpnet"


class PNet(pl.LightningModule):
    def __init__(self, input_size=32, output_size=2, training_config: Dict = {}, validation_config: Dict = {},
                 test_config: Dict = {}, config: Dict = {}, reduce=False):
        super(PNet, self).__init__()
        
        if config:
            self.save_hyperparameters('config', 'training_config')
        
        self.training_config = training_config
        self.validation_config = validation_config
        self.test_config = test_config
        if "/home" in self.training_config['enet']:
            self.training_config['enet'] = os.path.join(project_path, self.training_config['enet'].split('mpnet/')[-1])
            self.validation_config['enet'] = os.path.join(project_path,
                                                          self.training_config['enet'].split('mpnet/')[-1])
        self.training_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None
        
        drop_rate = config.get('dropout_rate', 0.5)
        activation = config.get('activation', nn.PReLU)
        self.optimizer = config.get('optimizer', Adagrad)
        
        dropout = nn.AlphaDropout if activation == nn.SELU else nn.Dropout
        
        self.fc = nn.Sequential(
                nn.Linear(input_size, 1280), activation(), dropout(drop_rate),
                nn.Linear(1280, 1024), activation(), dropout(drop_rate),
                nn.Linear(1024, 896), activation(), dropout(drop_rate),
                nn.Linear(896, 768), activation(), dropout(drop_rate),
                nn.Linear(768, 512), activation(), dropout(drop_rate),
                nn.Linear(512, 384), activation(), dropout(drop_rate),
                nn.Linear(384, 256), activation(), dropout(drop_rate),
                nn.Linear(256, 256), activation(), dropout(drop_rate),
                nn.Linear(256, 128), activation(), dropout(drop_rate),
                nn.Linear(128, 64), activation(), dropout(drop_rate),
                nn.Linear(64, 32), activation(),
                nn.Linear(32, output_size))
        
        self.learning_rate = config.get('lr', 1e-2)
        
        if activation == nn.SELU:
            self.init_weights()
        self.reduce = reduce
    
    def init_weights(self):
        def init_for_selu(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        
        self.apply(init_for_selu)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.fc(x)
        loss = mse_loss(x, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.fc(x)
        loss = mse_loss(result, y)
        self.log("val_loss", loss)
        return {"val_loss", loss}
    
    def forward(self, x):
        out = self.fc(x)
        return out
    
    def configure_optimizers(self):
        optim = self.optimizer(self.parameters(), lr=self.learning_rate)
        if self.reduce:
            # reduce_lr = ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=5, cooldown=2,
            #                               threshold=1e-2, verbose=True, min_lr=1e-6, threshold_mode='abs')
            # gen_scheduler = {"scheduler": reduce_lr, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
            annealing = CosineAnnealingLR(optim, 3000)
            
            return {"optimizer": optim, "lr_scheduler": {"scheduler": annealing, "monitor": 'val_loss'}}
        else:
            return [optim]
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.fc(x)
        loss = mse_loss(x, y)
        return {"test_loss": loss}
    
    def prepare_data(self) -> None:
        if self.training_config:
            self.training_dataloader = loader(**self.training_config)
        if self.validation_config:
            self.validation_dataloader = loader(**self.validation_config)
        if self.test_config:
            self.test_dataloader = loader(**self.test_config)
    
    def train_dataloader(self) -> Any:
        return self.training_dataloader
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.validation_dataloader
    
    def test_dataloader(self):
        return self.test_dataloader
