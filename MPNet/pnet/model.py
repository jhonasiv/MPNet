from typing import Any, Dict, List, Union

import pytorch_lightning as pl
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adagrad, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class PNet(pl.LightningModule):
    def __init__(self, input_size=32, output_size=2, training_dataloader=None, validation_dataloader=None,
                 test_dataloader=None, config: Dict = {}, reduce=False):
        super(PNet, self).__init__()
        
        if config:
            self.save_hyperparameters(config)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        
        dropout = config.get('dropout', 0.5)
        activation = config.get('activation', nn.PReLU)
        linear = config.get('linear', False)
        
        self.fc = nn.Sequential(
                nn.Linear(input_size, 1280), activation(), nn.Dropout(dropout),
                nn.Linear(1280, 1024), activation(), nn.Dropout(dropout),
                nn.Linear(1024, 896), activation(), nn.Dropout(dropout),
                nn.Linear(896, 768), activation(), nn.Dropout(dropout),
                nn.Linear(768, 512), activation(), nn.Dropout(dropout),
                nn.Linear(512, 384), activation(), nn.Dropout(dropout),
                nn.Linear(384, 256), activation(), nn.Dropout(dropout),
                nn.Linear(256, 256), activation(), nn.Dropout(dropout),
                nn.Linear(256, 128), activation(), nn.Dropout(dropout),
                nn.Linear(128, 64), activation(), nn.Dropout(dropout),
                nn.Linear(64, 32), activation(),
                nn.Linear(32, output_size))
        if linear:
            self.fc.add_module("output_activation", activation())
        
        self.learning_rate = 1e-3
        self.reduce = reduce
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.fc(x)
        loss = mse_loss(x, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.fc(x)
        loss = mse_loss(x, y)
        self.log("val_loss", loss)
        return {"val_loss", loss.item()}
    
    def forward(self, x):
        out = self.fc(x)
        return out
    
    def configure_optimizers(self):
        optim = Adagrad(self.parameters(), lr=self.learning_rate)
        if self.reduce:
            reduce_lr = ReduceLROnPlateau(optim, mode='min', factor=0.2, patience=6, cooldown=2,
                                          threshold=1e-4, verbose=True, min_lr=1e-6, threshold_mode='abs')
            gen_scheduler = {"scheduler": reduce_lr, 'reduce_on_plateau': True, 'monitor': 'val_loss'}
            
            return [optim], [gen_scheduler]
        else:
            return [optim]
    
    def test_step(self, batch, batch_idx):
        x, y = batch.float()
        x = self.fc(x)
        loss = mse_loss(x, y)
        return {"test_loss": loss}
    
    def train_dataloader(self) -> Any:
        return self.training_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.validation_dataloader

    def test_dataloader(self):
        return self.test_dataloader
