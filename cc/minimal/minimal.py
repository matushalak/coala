# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
from cc.utils import EMA, nonnegative, randn_reparam, ThresholdReLU
from typing import Literal

class CCNeuron(nn.Module):
    """
    Minimal contextual-contrasting model with:
      - one pyramidal neuron y (scalar),
      - two PV neurons p (vector of size 2),
      - feedforward input x (size 2),
      - contextual input c (size 2).

    Dynamics:
      p = phi(W_pv x)
      y = phi(w_ff^T x + w_fb^T c - w_lat^T p)

    Local learning rules:
      dw_ff  ~ -(y * x)                           (anti-Hebbian)
      dw_fb  ~ (alpha / (y + alpha)) * (c)   (dampened-anti-Hebbian)
            OR y * c (Hebbian)
      dw_lat ~  (y * p)                           (Hebbian)
      dW_pv  ~  p x^T                             (Hebbian)
    """
    # TODO: add weight initialization according to specified distribution
    # using randn_reparam
    def __init__(
        self,
        n_features: int = 2,
        n_pv: int = 2,
        n_context: int = 2,
        activation: nn.Module | None = None,
        lr_ff: float = 0.01,
        w_ff_init:dict = {'mu': [0.5, 0.5], 'sigma': 1e-2},
        lr_fb: float = 0.01,
        w_fb_init:dict = {'mu': [0.1, 0.1], 'sigma': 1e-2},
        lr_lat: float = 0.01,
        w_lat_init:dict = {'mu': [0.2, 0.2], 'sigma': 1e-2},
        lr_pv: float = 0.01,
        W_pv_init:dict = {'mu': ([0.1, 0.1], [0.1,0.1]), 'sigma': [1e-2, 1e-2]},
        pyc_decay:float = 0.1,
        pv_decay:float = 0.25,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        seed:int = 42,
        receives_context:tuple[bool, bool ] = (True, True),
        FFrule:Literal['anti-Hebbian', 'Hebbian'] = 'anti-Hebbian',
        FBrule:Literal["dampened-anti-Hebbian", "Hebbian"] = "dampened-anti-Hebbian",
        use_FF_connection:bool = True,
        FF_plasticity:bool = True,
        use_FB_connection:bool = True,
        FB_plasticity:bool = True,
        use_lat_connection:bool = True,
        lat_plasticity:bool = True,
        use_pv_connection:bool = True,
        pv_plasticity:bool = True
    ):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if weight_decay < 0 or weight_decay > 1:
            raise ValueError("weight_decay must be 0 <= wd <= 1.")

        torch.manual_seed(seed) # set random seed for weight initialization
        assert FFrule in ["anti-Hebbian", "Hebbian"], "FFrule must be either 'anti-Hebbian' or 'Hebbian'."
        self.FFrule = FFrule
        assert FBrule in ["dampened-anti-Hebbian", "Hebbian"], "FBrule must be either 'dampened-anti-Hebbian' or 'Hebbian'."
        self.FBrule = FBrule
        assert len(receives_context) == 2, "receives_context must be a tuple of two booleans indicating whether the neuron receives context input for familiar and novel conditions respectively."
        self.receives_context = torch.tensor(receives_context, dtype=torch.bool)

        self.n_features = n_features
        self.n_pv = n_pv
        self.n_context = n_context
        self.activation = activation if activation is not None else nn.ReLU()

        # Learnable weights updated manually via local rules
        self.w_ff = nonnegative(randn_reparam(size=(1,), **w_ff_init))
        self.w_fb = nonnegative(randn_reparam(size=(1,), **w_fb_init)) * self.receives_context
        self.w_lat = nonnegative(randn_reparam(size=(1,), **w_lat_init))
        self.W_pv = torch.cat((
            nonnegative(randn_reparam(size=(1,), mu = W_pv_init['mu'][0],sigma = W_pv_init['sigma'][0])).unsqueeze(0),
            nonnegative(randn_reparam(size=(1,), mu = W_pv_init['mu'][1],sigma = W_pv_init['sigma'][1])).unsqueeze(0)), 
                             dim=0)
        # Hyperpatameters
        self.lr_ff = lr_ff
        self.lr_fb = lr_fb
        self.lr_lat = lr_lat
        self.lr_pv = lr_pv
        self.alpha = alpha
        self.weight_decay = weight_decay

        # State variables for PV and pyramidal neurons, implemented as EMAs.
        self.pv = EMA(shape=(n_pv,), alpha=pv_decay)
        self.pyramidal = EMA(shape=(), alpha=pyc_decay)
        self.adapt = EMA(shape=(), alpha=pyc_decay*0.2)

        # EMA of weights to implement decay towards baseline in absence of input (optional)
        # Baselines
        self.w_ff_baseline = self.w_ff.detach().clone()
        self.w_fb_baseline = self.w_fb.detach().clone()
        self.w_lat_baseline = self.w_lat.detach().clone()
        self.W_pv_baseline = self.W_pv.detach().clone()

        # Feedback specificity (decoding image identity with 60% accuracy)
        self.fb_specificity = torch.tensor([[0.3, 0.2],
                                            [0.2, 0.3]])
        
        # Ablation parameters
        self.use_FF_connection = use_FF_connection
        self.FF_plasticity = FF_plasticity
        self.use_FB_connection = use_FB_connection
        self.FB_plasticity = FB_plasticity
        self.use_lat_connection = use_lat_connection
        self.lat_plasticity = lat_plasticity
        self.use_pv_connection = use_pv_connection
        self.pv_plasticity = pv_plasticity

    def forward(self, x: torch.Tensor, c: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: bottom-up input, shape (n_features,)
            c: contextual input, shape (n_context,)
        Returns:
            y: pyramidal activity, scalar tensor shape ()
            p: PV activity, shape (n_pv,)
        """
        assert x.shape == (self.n_features,) and c.shape == (self.n_context,)

        # feedforward excitation to PV neurons
        pv_ff = self.W_pv @ x * self.use_pv_connection
        pv_lat = self.pyramidal.ema * self.w_lat * self.use_lat_connection
        p = self.pv(self.activation(
            pv_ff + pv_lat 
            + randn_reparam(size=self.pv.ema.shape, mu=0, sigma=0.06) # small random baseline input
            )) 
        
        a = self.adapt(self.pyramidal.ema) # update adaptation variable 

        y_ff  = torch.dot(self.w_ff, x) * self.use_FF_connection # feedforward excitation
        y_fb = torch.dot(self.w_fb, c * self.receives_context) * self.use_FB_connection # feedback excitation 
        y_lat = torch.dot(self.w_lat, p) * self.use_lat_connection # "lateral" inhibition 
        y = self.pyramidal(self.activation(
            y_ff + y_fb - y_lat
            + randn_reparam(size=(), mu=0, sigma=0.01) # small random baseline input
            - a # adaptation
            ))
        
        return x, y, p, c

    @torch.no_grad()
    def update(self, x_t: torch.Tensor, y_next:torch.Tensor, 
               pv_t:torch.Tensor, c_t: torch.Tensor):
        """
        One local update step using current inputs (x_t, c_t).
        Returns y_{t+1}, p_t as computed for this step.
        """
        dw_ff, dw_fb, dw_lat, dw_W_pv = (torch.zeros_like(self.w_ff), torch.zeros_like(self.w_fb), 
                                         torch.zeros_like(self.w_lat), torch.zeros_like(self.W_pv))
        
        # 1) Anti-Hebbian update for w_ff
        if self.FF_plasticity:
            match self.FFrule:
                case "anti-Hebbian":
                    dw_ff = - self.lr_ff * (y_next * x_t)
                case "Hebbian":
                    dw_ff = self.lr_ff * (y_next * x_t)

        # 2) Dampened-Hebbian update for w_fb
        damp = self.alpha / (y_next + self.alpha)

        if self.FB_plasticity:
            match self.FBrule:
                # contextual strengthening general (not only the experienced context, also novel)
                case "dampened-anti-Hebbian":
                    # dw_fb = self.lr_fb * (damp * y_next * c_t)
                    dw_fb = self.lr_fb * (damp * self.fb_specificity @ c_t) * self.receives_context
                case "Hebbian":
                    dw_fb = self.lr_fb * (y_next * self.fb_specificity @ c_t) * self.receives_context

        # 3) Hebbian update for w_lat
        if self.lat_plasticity:
            dw_lat = self.lr_lat * (y_next * pv_t)

        # 4) Hebbian update for W_pv
        if self.pv_plasticity:
            dw_W_pv = self.lr_pv * torch.outer(pv_t, x_t)

        # Apply updates
        self.w_ff += dw_ff
        self.w_fb += dw_fb
        self.w_lat += dw_lat
        self.W_pv += dw_W_pv
        
        # Decay towards baseline
        if 0.0 < self.weight_decay < 1.0:
            self.w_ff -= (self.w_ff - self.w_ff_baseline) * self.weight_decay * self.FF_plasticity
            self.w_fb -= (self.w_fb - self.w_fb_baseline) * self.weight_decay * self.FB_plasticity
            self.w_lat -= (self.w_lat - self.w_lat_baseline) * self.weight_decay * self.lat_plasticity
            self.W_pv -= (self.W_pv - self.W_pv_baseline) * self.weight_decay * self.pv_plasticity
        
        # Ensure non-negativity of weights
        self.w_ff = nonnegative(self.w_ff)
        self.w_fb = nonnegative(self.w_fb) * self.receives_context
        self.w_lat = nonnegative(self.w_lat)
        self.W_pv = nonnegative(self.W_pv)


    def _reset_state(self):
        self.pv.reset_state()
        self.pyramidal.reset_state()

if __name__ == "__main__":
    # Example usage:
    model = CCNeuron()
    n_steps = 50
    X = torch.randn((n_steps, model.n_features)) # random input sequence
    C = torch.randn((n_steps, model.n_context)) # random context sequence

    for step in range(n_steps):
        x, y, p, c = model(X[step], C[step])
        update = model.update(x, y, p, c)
        print(f"Step {step}: y={y.item():.4f}, p={p.detach().numpy()}")