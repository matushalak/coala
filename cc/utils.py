import torch

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
    def __init__(self, shape:tuple, alpha:float = 0.1, baseline:torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.baseline = baseline if baseline is not None else torch.zeros(shape, requires_grad=False)
        self.ema = self.baseline.clone()
    
    @torch.no_grad()
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.ema = (1 - self.alpha) * self.ema + self.alpha * (x+self.baseline)
        return self.ema
    
    def reset_state(self):
        self.ema = self.baseline.clone()

def nonnegative(x:torch.Tensor)->torch.Tensor:
    '''
    Performs x'= max(0, x) elementwise, ensuring all synaptic weights are non-negative.
    '''
    return torch.clamp(x, min=0.0)
