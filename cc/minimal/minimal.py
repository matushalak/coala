# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
from cc.utils import EMA, nonnegative, randn_reparam

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
      dw_fb  ~ (alpha / (y + alpha)) * (y * c)   (dampened-Hebbian)
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
        lr_fb: float = 0.01,
        lr_lat: float = 0.01,
        lr_pv: float = 0.01,
        pyc_decay:float = 0.1,
        pv_decay:float = 0.25,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        seed:int = 42,
    ):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")

        torch.manual_seed(seed) # set random seed for weight initialization

        self.n_features = n_features
        self.n_pv = n_pv
        self.n_context = n_context
        self.activation = activation if activation is not None else nn.ReLU()

        # Learnable weights (stored as Parameters; updated manually via local rules).
        self.w_ff = nn.Parameter(randn_reparam(size=(n_features,), mu=0.0, sigma=1.0), requires_grad=False)
        self.w_fb = nn.Parameter(randn_reparam(size=(n_context,), mu=0.0, sigma=0.1), requires_grad=False)
        self.w_lat = nn.Parameter(randn_reparam(size=(n_pv,), mu=0.0, sigma=0.3), requires_grad=False)
        self.W_pv = nn.Parameter(randn_reparam(size=(n_pv, n_features), mu=0.0, sigma=0.1), requires_grad=False)
        # breakpoint()
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

        p = self.pv(self.activation(nonnegative(self.W_pv) @ x)) # feedforward excitation to PV neurons
        y = self.pyramidal(self.activation(
            torch.dot(nonnegative(self.w_ff), x) # feedforward excitation
            + torch.dot(nonnegative(self.w_fb), c) # feedback excitation
            - torch.dot(nonnegative(self.w_lat), p) # "lateral" inhibition 
                            )) 
        
        return x, y, p, c

    @torch.no_grad()
    def update(self, x_t: torch.Tensor, y_next:torch.Tensor, 
               pv_t:torch.Tensor, c_t: torch.Tensor):
        """
        One local update step using current inputs (x_t, c_t).
        Returns y_{t+1}, p_t as computed for this step.
        """
        # 1) Anti-Hebbian update for w_ff
        self.w_ff -= self.lr_ff * (y_next * x_t)

        # 2) Dampened-Hebbian update for w_fb
        damp = self.alpha / (y_next + self.alpha)
        self.w_fb += self.lr_fb * (damp * y_next * c_t)

        # 3) Hebbian update for w_lat
        self.w_lat += self.lr_lat * (y_next * pv_t)

        # 4) Hebbian update for W_pv
        self.W_pv += self.lr_pv * torch.outer(pv_t, x_t)

        if self.weight_decay > 0.0:
            decay = 1.0 - self.weight_decay
            self.w_ff *= decay
            self.w_fb *= decay
            self.w_lat *= decay
            self.W_pv *= decay

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