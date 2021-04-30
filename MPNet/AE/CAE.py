from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torch.autograd import Variable
import pytorch_lightning as pl
from torch.nn.functional import mse_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class ContractiveAutoEncoder(pl.LightningModule):
    def __init__(self, training_dataloader=None, val_dataloader=None, config: Dict = {}, reduce: bool = False):
        super(ContractiveAutoEncoder, self).__init__()
        self.training_dataloader = training_dataloader
        self.validation_dataloader = val_dataloader
        
        self.learning_rate = 1e-4
        self.reduce = reduce
        
        l1_units = config.get("l1_units", 512)
        l2_units = config.get("l2_units", 256)
        l3_units = config.get("l3_units", 128)
        actv = config.get("actv", nn.PReLU)
        self.lambd = config.get("lambda", 1e-3)
        
        self.encoder = nn.Sequential(nn.Linear(2800, l1_units), actv(),
                                     nn.Linear(l1_units, l2_units), actv(),
                                     nn.Linear(l2_units, l3_units), actv())
        self.encoder.add_module("embedding", nn.Linear(l3_units, 28))
        self.decoder = nn.Sequential(nn.Linear(28, l3_units), actv(),
                                     nn.Linear(l3_units, l2_units), actv(),
                                     nn.Linear(l2_units, l1_units), actv(),
                                     nn.Linear(l1_units, 2800))
        self.code = None
    
    def loss(self, reconstruction, x, weights, h):
        mse = mse_loss(reconstruction, x)
        dh = h * (1 - h)
        contractive_loss = torch.sum(dh ** 2 * torch.sum(Variable(weights) ** 2, dim=1), dim=1).mul_(self.lambd)
        return mse + contractive_loss
    
    def forward(self, inputs):
        self.code = self.encoder(inputs)
        return self.code
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch).mean()
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)
        return {"val_loss": loss.item()}
    
    def shared_step(self, batch):
        x = batch.float()
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        weights = self.encoder.state_dict()['embedding.weight']
        reconstruction = self.decoder(h)
        loss = self.loss(reconstruction, x, weights, h)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.reduce:
            reduce_lr = ReduceLROnPlateau(self.optim, mode='min', factor=0.2, patience=6, cooldown=2,
                                          threshold=1e-4, verbose=True, min_lr=1e-6, threshold_mode='abs')
            gen_scheduler = {"scheduler": reduce_lr, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
            
            return [optim], [gen_scheduler]
        else:
            return [optim]
    
    def train_dataloader(self) -> Any:
        return self.training_dataloader
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.validation_dataloader
