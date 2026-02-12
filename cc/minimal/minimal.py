import torch
from utils import nonnegative

class Minimal(torch.nn.Module):
    def __init__(self,
                 n_inputs:int = 6,
                 n_pyramidal :int = 3,
                 n_pv:int = 2,
                 n_hva:int = 2,
                 HVA_tuning:torch.Tensor | None = None):
        super().__init__()
        # Store parameters
        self.n_inputs = n_inputs
        self.n_pyramidal = n_pyramidal
        self.n_pv = n_pv
        self.n_hva = n_hva

        # Define receptive fields
        # Pyramidal neuron receptive fields (non-overlapping)
        self.RF_y_i = torch.arange(n_inputs).view(n_pyramidal, -1)  
        # Pyramidal neuron Parvalbumin inputs (overlapping)
        self.RF_y_pv = [[p for p, p_in in enumerate(
            zip(list(range(n_pyramidal)), list(range(n_pyramidal))[1:]))
                        if y in p_in] for y in range(n_pyramidal)]
        # Parvalbumin (PV) neuron receptive fields (non-overlapping)
        self.RF_pv = torch.arange(n_inputs).view(n_pv, -1)
        # HVA neuron receptive fields (overlapping)
        self.RF_hva = torch.arange(n_pyramidal).tile(n_hva).view(n_hva, -1)

        # Define layers weights
        # Layer 1 W_FFpv: Input-Parvalbumin (PV) neuron weights (PV x I)
        self.W_FFpv = torch.nn.Parameter(torch.ones(n_pv, n_inputs, requires_grad=False))
        # Layer 1 W_FFy: Feedforward Input-Pyramidal neuron (Y) weights (Y x I)
        self.W_FFy = torch.nn.Parameter(torch.ones(n_pyramidal, n_inputs), requires_grad=False)
        # Layer 1 W_Iy: Inhibitory PV-Pyramidal neuron weights (Y x PV)
        self.W_Iy = torch.nn.Parameter(0.4*torch.ones(n_pyramidal, n_pv), requires_grad=False)
        # Layer 2 W_FFh: Feedforward Pyramidal-HVA neuron weights (HVA x Y)
        if HVA_tuning is not None: # initialize with specific tuning pattern
            assert HVA_tuning.shape == (n_hva, n_pyramidal), "HVA_tuning must have shape (n_hva, n_pyramidal)"
            self.W_FFh = torch.nn.Parameter(HVA_tuning, requires_grad=False)
        else: # initialize with random weights
            self.W_FFh = torch.nn.Parameter(torch.ones(n_hva, n_pyramidal), requires_grad=False)
        # Layer 2 W_FBy: Feedback HVA-Pyramidal neuron weights (Y x HVA)
        self.W_FBy = torch.nn.Parameter(0.1*torch.ones(n_pyramidal, n_hva), requires_grad=False)

        # Create boolean masks for local weights based on receptive fields
        # Mask for W_FFpv (PV x I): each PV neuron connects to inputs in its RF
        self.mask_FFpv = self._create_mask_RF(n_pv, n_inputs, self.RF_pv)
        # Mask for W_Iy (Y x PV): inhibition from PV to pyramidal
        self.mask_Iy = self._create_mask_RF(n_pyramidal, n_pv, self.RF_y_pv)
        # Mask for W_FFy (Y x I): each pyramidal neuron connects to inputs in its RF
        self.mask_FFy = self._create_mask_RF(n_pyramidal, n_inputs, self.RF_y_i)
        # Mask for W_FFh (HVA x Y): each HVA neuron connects to pyramidal neurons in its RF
        self.mask_FFh = torch.ones(n_hva, n_pyramidal, dtype=torch.bool)  # all-to-all for simplicity
        # Mask for W_FBy (Y x HVA): feedback from HVA to pyramidal (all-to-all)
        self.mask_FBy = torch.ones(n_pyramidal, n_hva, dtype=torch.bool)

        # Neuron nonlinear activation function (e.g., sigmoid, tanh, ReLU)
        self.activation = torch.nn.ReLU() # benefit of ReLU, stay 0 at 0 input

    def forward(self, I:torch.Tensor, hva_ini:torch.Tensor | None = None, train:bool = False) -> torch.Tensor:
        # To store pyramidal, PV, and HVA activations over time
        out = {'Pyramidal': torch.zeros(self.n_pyramidal, I.shape[0]), 
               'PV': torch.zeros(self.n_pv, I.shape[0]), 
               'HVA': torch.zeros(self.n_hva, I.shape[0]),
               'Time': torch.arange(I.shape[0])}
        # Initialize HVA neuron activations
        hva = hva_ini if hva_ini is not None else torch.zeros(self.n_hva)
        for t, stim in enumerate(I):
            # PV neuron activations based on current stimulus
            pv = self.activation((nonnegative(self.W_FFpv)*self.mask_FFpv) @ stim)
            # pyramidal neuron activations based on 
            pyramidal = (nonnegative(self.W_FFy)*self.mask_FFy) @ stim  # feedforward input (current stimulus)
            pyramidal -= (nonnegative(self.W_Iy)*self.mask_Iy) @ pv  # PV inhibition
            pyramidal += (nonnegative(self.W_FBy)*self.mask_FBy) @ hva  # HVA feedback based on previous timestep HVA activations
            pyramidal = self.activation(pyramidal)  # apply nonlinearity
            # HVA neuron activations based on current pyramidal activations
            hva = self.activation(nonnegative(self.W_FFh)*self.mask_FFh @ pyramidal)
            if train:
                self.update(pyramidal, pv, hva) # update weights based on local learning rules
            # store activations for this timestep
            out['Pyramidal'][:, t] = pyramidal  # store all pyramidal activations
            out['PV'][:, t] = pv  # store all PV activations
            out['HVA'][:, t] = hva  # store all HVA activations
        return out

    @torch.no_grad()
    def update(self, pyramidal:torch.Tensor, pv:torch.Tensor, hva:torch.Tensor):
        # custom update rules can be implemented here
        pass

    def _create_mask_RF(self, n_out: int, n_in: int, RF_indices: torch.Tensor|list) -> torch.Tensor:
        """
        Create mask for weight matrix based on receptive field indices.
        
        Args:
            n_out: Number of output neurons (rows of weight matrix)
            n_in: Number of input neurons (columns of weight matrix)
            RF_indices: Receptive field indices tensor / nested list of shape (n_out, RF_size)
                        where each row contains the input indices for that neuron
        
        Returns:
            Boolean mask of shape (n_out, n_in) where True indicates allowed connections
        """
        mask = torch.zeros(n_out, n_in, dtype=torch.bool)
        for i in range(n_out):
            mask[i, RF_indices[i]] = True
        return mask