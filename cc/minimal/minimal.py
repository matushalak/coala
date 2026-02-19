import torch
import torch.nn as nn
from cc.utils import EMA, nonnegative

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

    def __init__(
        self,
        n_features: int = 2,
        activation: nn.Module | None = None,
        lr_ff: float = 0.01,
        lr_fb: float = 0.01,
        lr_lat: float = 0.01,
        lr_pv: float = 0.01,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        if n_features != 2:
            raise ValueError("This minimal model expects n_features=2.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")

        self.n_features = n_features
        self.activation = activation if activation is not None else nn.ReLU()

        # Learnable weights (stored as Parameters; updated manually via local rules).
        self.w_ff = nn.Parameter(torch.randn(n_features), requires_grad=False)
        self.w_fb = nn.Parameter(0.1 * torch.randn(n_features), requires_grad=False)
        self.w_lat = nn.Parameter(0.3 * torch.randn(n_features), requires_grad=False)
        self.W_pv = nn.Parameter(0.1 * torch.randn(n_features, n_features), requires_grad=False)

        self.lr_ff = lr_ff
        self.lr_fb = lr_fb
        self.lr_lat = lr_lat
        self.lr_pv = lr_pv
        self.alpha = alpha
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: bottom-up input, shape (2,)
            c: contextual input, shape (2,)
        Returns:
            y: pyramidal activity, scalar tensor shape ()
            p: PV activity, shape (2,)
        """
        if x.shape != (self.n_features,) or c.shape != (self.n_features,):
            raise ValueError(
                f"x and c must each have shape ({self.n_features},), "
                f"got x={tuple(x.shape)}, c={tuple(c.shape)}."
            )

        p = self.activation(self.W_pv @ x)
        y = self.activation(torch.dot(self.w_ff, x) + torch.dot(self.w_fb, c) - torch.dot(self.w_lat, p))
        return y, p

    @torch.no_grad()
    def update(self, x_t: torch.Tensor, c_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One local update step using current inputs (x_t, c_t).
        Returns y_{t+1}, p_t as computed for this step.
        """
        y_next, p_t = self.forward(x_t, c_t)

        # 1) Anti-Hebbian update for w_ff
        self.w_ff -= self.lr_ff * (y_next * x_t)

        # 2) Dampened-Hebbian update for w_fb
        damp = self.alpha / (y_next + self.alpha)
        self.w_fb += self.lr_fb * (damp * y_next * c_t)

        # 3) Hebbian update for w_lat
        self.w_lat += self.lr_lat * (y_next * p_t)

        # 4) Hebbian update for W_pv
        self.W_pv += self.lr_pv * torch.outer(p_t, x_t)

        if self.weight_decay > 0.0:
            decay = 1.0 - self.weight_decay
            self.w_ff *= decay
            self.w_fb *= decay
            self.w_lat *= decay
            self.W_pv *= decay

        return y_next, p_t

if __name__ == "__main__":
    # Example usage:
    model = CCNeuron()
    x = torch.tensor([1.0, 0.5])
    c = torch.tensor([0.5, 1.0])
    for step in range(10):
        y, p = model.update(x, c)
        print(f"Step {step}: y={y.item():.4f}, p={p.detach().numpy()}")