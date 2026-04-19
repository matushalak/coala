# author: Matúš Halák (@matushalak)
import torch
import torch.nn.functional as F

class EMA(torch.nn.Module):
    '''
    EMA (Exponential Moving Average) = Discretized Leaky Integrator
        Alpha controls history dependence and stability (how many steps it takes to decay to baseline); 
        Low alpha (eg. 1e-4): 
            slower integration, more history dependence, 
            slower decay, takes 10000 steps to decay to baseline
        High alpha (eg. 1e-2): 
            faster integration, more current input dependence, 
            faster decay, takes 100 steps to decay to baseline
    
    If basline is provided, decay towards baseline in absence of input; 
        otherwise, decay towards 0.
    '''
    def __init__(self, shape:tuple, alpha:float = 0.1, baseline:torch.Tensor | None = None,
                 learnable_alpha:bool = False, matrix_alpha:bool = False):
        super().__init__()
        if matrix_alpha:
            self.alpha = torch.nn.Parameter(torch.full(shape, alpha), requires_grad=learnable_alpha)
        else:
            self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=learnable_alpha)
        self.register_buffer("baseline", baseline if baseline is not None else torch.zeros(shape, requires_grad=False))
        self.register_buffer("ema", self.baseline.clone())
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.ema = (1 - F.sigmoid(self.alpha)) * self.ema + F.sigmoid(self.alpha) * (x+self.baseline)
        return self.ema
    
    def reset_state(self, batch_size:int | None = None):
        if batch_size is None:
            self.ema = self.baseline.clone()
            return
        self.ema = self.baseline.expand(batch_size, *self.baseline.shape[1:]).clone()
