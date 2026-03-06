# COALA: COllapsed Autoencoder with Local Adaptation
# in principle this can use any spatially-matched encoder and decoder
# for bio-realism; use CNN
# but for scaling up, could use ConvNext; and possibly even (Swin)ViT
import argparse
import os
from types import FunctionType

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cc.datasets.mnist import mnist
from cc.datasets.msmnist import msmnist
from cc.ml.cc_modules import CCModule
from cc.ml.architecture import COALANet
from cc.ml.heads.task_head import TaskHead
from cc.ml.heads.classifier import TemporalCEloss

class COALA(pl.LightningModule):
    '''
    COALA: Collapsed Autoencoder with Local Adaptation
    '''
    def __init__(self, model:COALANet, head:TaskHead,
                 loss_fn:FunctionType = TemporalCEloss()):
        super().__init__()
        self.model:COALANet = model
        self.head:TaskHead = head
        self.loss_fn:FunctionType = loss_fn

    def forward(self, images:torch.Tensor, targets:list[torch.Tensor]
                )->torch.Tensor:
        ''''
        Expects tensor with temporal dimension of shape (B, T, C, H, W).
        Returns outputs for each timestep in the batch
        '''
        predictions = self.model(images)
        return self.loss_fn(predictions, targets)
    

# TODO: logging like in MAEmodel, make sure that 
# when training (COALANet.training_params == True); COALAnet.dynamic_updates == True
#   only the learning rates within LambdaModules
#   and time_alphas within CCModules are updated, everything else must remain frozen
# after each batch call LambdaModule.reset()
# After training, freeze everything in place and checkpoint everything 
# At test time, can run separate tests with and without COALANet.dynamic_updates

