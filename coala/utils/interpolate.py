import torch
import torch.nn as nn
import torch.nn.functional as F

class Interpolate(nn.Module):
    def __init__(self, size:tuple[int, int], mode:str = "bilinear", align_corners:bool = False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)