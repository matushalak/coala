from typing import Literal
import torch
from cc.utils import EMA, nonnegative

class Circuit(torch.nn.Module):
    def __init__(self,
                 n_inputs:int = 6,
                 n_pyramidal :int = 3,
                 n_pv:int = 2,
                 n_hva:int = 2,
                 HVA_tuning:torch.Tensor | None = None,
                 cell_type_alphas:dict[str: float] = {'PV': 0.25, 'Pyramidal': 0.1},
                 feedback_rule:Literal['Hebbian', 'Anti-Hebbian'] = 'Hebbian'
                 ): 
        '''
        Minimal model of cortical circuit implementing context contrasting with local learning rules.
        
        TODO: implement weight decay / forgetting using EMA of weights, 
            where in absence of input, weights decay towards baseline (initial) values;
        
        NOTE: Current FB learning rule is Hebbian
        TODO: try anti-Hebbian FB learning rule:
            - general strengthening (or Hebbian wrt other HVA neuron activations)
            (apical dendrite behave like small neuron, LTP-like process when multiple apical inputs at once)
            - anti-Hebbian wrt PyC receiving FB

        TODO: model parameters as input config dict
        '''
        super().__init__()
        # Store parameters
        self.n_inputs = n_inputs
        self.n_pyramidal = n_pyramidal
        self.n_pv = n_pv
        self.n_hva = n_hva
        self.feedback_rule = feedback_rule

        # Define receptive fields
        # Pyramidal neuron receptive fields (non-overlapping)
        self.RF_y_i = torch.arange(n_inputs).view(n_pyramidal, -1)  
        # Pyramidal neuron Parvalbumin inputs (overlapping)
        self.RF_y_pv = [[pv for pv, y_in in enumerate(
            zip(list(range(n_pyramidal)), list(range(n_pyramidal))[1:]))
                        if y in y_in] for y in range(n_pyramidal)]
        
        # Parvalbumin (PV) neuron FF receptive fields (non-overlapping)
        self.RF_pv_ff = torch.arange(n_inputs).view(n_pv, -1)
        # Parvalbumin (PV) neuron lateral receptive fields (overlapping)
        self.RF_pv_lateral = [[y_in for pv, y_in in enumerate(
            zip(list(range(n_pyramidal)), list(range(n_pyramidal))[1:]))
                        if Pv == pv] for Pv in range(n_pv)]

        # HVA neuron receptive fields (overlapping)
        self.RF_hva = torch.arange(n_pyramidal).tile(n_hva).view(n_hva, -1)

        # Define layers weights
        # Layer 1 W_FFpv: Input-Parvalbumin (PV) neuron weights (PV x I)
        self.W_FFpv = torch.nn.Parameter(0.1*torch.ones(n_pv, n_inputs, requires_grad=False))
        # Layer 1 W_LatPV: Lateral Y-PV weights (PV x Y)
        self.W_LatPV = torch.nn.Parameter(0.75*torch.ones(n_pv, n_pyramidal), requires_grad=False)
        # Layer 1 W_FFy: Feedforward Input-Pyramidal neuron (Y) weights (Y x I)
        self.W_FFy = torch.nn.Parameter(torch.ones(n_pyramidal, n_inputs), requires_grad=False)
        # Layer 1 W_Iy: Inhibitory PV-Pyramidal neuron weights (Y x PV)
        self.W_Iy = torch.nn.Parameter(0.3*torch.ones(n_pyramidal, n_pv), requires_grad=False)
        # Layer 2 W_FFh: Feedforward Pyramidal-HVA neuron weights (HVA x Y)
        if HVA_tuning is not None: # initialize with specific tuning pattern
            assert HVA_tuning.shape == (n_hva, n_pyramidal), "HVA_tuning must have shape (n_hva, n_pyramidal)"
            self.W_FFh = torch.nn.Parameter(HVA_tuning, requires_grad=False)
        else: # initialize with constant weights
            self.W_FFh = torch.nn.Parameter(torch.ones(n_hva, n_pyramidal), requires_grad=False)
        # Layer 2 W_FBy: Feedback HVA-Pyramidal neuron weights (Y x HVA)
        self.W_FBy = torch.nn.Parameter(0.1*torch.ones(n_pyramidal, n_hva), requires_grad=False)

        # Create boolean masks for local weights based on receptive fields
        # Mask for W_FFpv (PV x I): each PV neuron connects to inputs in its RF
        self.mask_FFpv = self._create_mask_RF(n_pv, n_inputs, self.RF_pv_ff)
        # Mask for W_LatPV (PV x Y): each PV neuron connects to pyramidal neurons in its lateral RF
        self.mask_LatPV = self._create_mask_RF(n_pv, n_pyramidal, self.RF_pv_lateral)
        # Mask for W_Iy (Y x PV): inhibition from PV to pyramidal
        self.mask_Iy = self._create_mask_RF(n_pyramidal, n_pv, self.RF_y_pv)
        # Mask for W_FFy (Y x I): each pyramidal neuron connects to inputs in its RF
        self.mask_FFy = self._create_mask_RF(n_pyramidal, n_inputs, self.RF_y_i)
        # Mask for W_FFh (HVA x Y): each HVA neuron connects to pyramidal neurons in its RF
        self.mask_FFh = torch.ones(n_hva, n_pyramidal, dtype=torch.bool)  # all-to-all for simplicity
        # Mask for W_FBy (Y x HVA): feedback from HVA to pyramidal (all-to-all)
        self.mask_FBy = torch.ones(n_pyramidal, n_hva, dtype=torch.bool)

        # Neuron nonlinear activation function (e.g., sigmoid, tanh, ReLU)
        self.activation = torch.nn.ReLU() # benefit of ReLU & Tanh, stay 0 at 0 input

        # Define exponential moving averages (decay) for neuron activations
        self.ema_pv = EMA(shape=self.n_pv, alpha=cell_type_alphas['PV'])
        self.ema_pyramidal = EMA(shape=self.n_pyramidal, alpha=cell_type_alphas['Pyramidal'])
        self.ema_hva = EMA(shape=self.n_hva, alpha=cell_type_alphas['Pyramidal'])

        # learning rates for local learning rules
        self.lr_Iy = torch.nn.Parameter(torch.tensor(0.0025))  # learning rate for inhibitory PV-Pyramidal weights
        self.lr_FFy = torch.nn.Parameter(torch.tensor(0.005))  # learning rate for feedforward Input-Pyramidal weights
        self.lr_FBy = torch.nn.Parameter(torch.tensor(0.0035))  # learning rate for feedback HVA-Pyramidal weights
        self.lr_FFpv = torch.nn.Parameter(torch.tensor(0.0015))  # learning rate for feedforward Input-PV weights

        # Define exponential moving averages (decay) for weights (to enable weight decay / forgetting)
        # TODO: use this in update()
        self.ema_FFy = EMA(shape=(n_pyramidal, n_inputs), alpha=1e-4, baseline=self.W_FFy.detach().clone())  # for feedforward Input-Pyramidal weights
        self.ema_Iy = EMA(shape=(n_pyramidal, n_pv), alpha=1e-4, baseline=self.W_Iy.detach().clone())  # for inhibitory PV-Pyramidal weights
        self.ema_FBy = EMA(shape=(n_pyramidal, n_hva), alpha=1e-4, baseline=self.W_FBy.detach().clone())  # for feedback HVA-Pyramidal weights
        self.ema_LatPV = EMA(shape=(n_pv, n_pyramidal), alpha=1e-4, baseline=self.W_LatPV.detach().clone())  # for lateral PV-Pyramidal weights
        self.ema_Ipv = EMA(shape=(n_pv, n_inputs), alpha=1e-4, baseline=self.W_FFpv.detach().clone())  # for feedforward Input-PV weights

    def forward(self, I:torch.Tensor, train:bool = False) -> torch.Tensor:
        # To store pyramidal, PV, and HVA activations over time
        out = {'Pyramidal': torch.zeros(self.n_pyramidal, I.shape[0]), 
               'PV': torch.zeros(self.n_pv, I.shape[0]), 
               'HVA': torch.zeros(self.n_hva, I.shape[0]),
               'Time': torch.arange(I.shape[0])}
        # Initialize HVA neuron activations
        hva = self.ema_hva.ema
        pyramidal = self.ema_pyramidal.ema
        for t, stim in enumerate(I):
            # PV neuron activations based on current stimulus
            pv = (nonnegative(self.W_FFpv)*self.mask_FFpv) @ stim
            # Add lateral activation of PV by pyramidal
            pv += (nonnegative(self.W_LatPV)*self.mask_LatPV) @ pyramidal
            pv = self.ema_pv(self.activation(pv))  # apply EMA to PV activations

            # pyramidal neuron activations based on 
            pyramidal = (nonnegative(self.W_FFy)*self.mask_FFy) @ stim  # feedforward input (current stimulus)
            pyramidal -= (nonnegative(self.W_Iy)*self.mask_Iy) @ pv  # PV inhibition
            pyramidal += (nonnegative(self.W_FBy)*self.mask_FBy) @ hva  # HVA feedback based on previous timestep HVA activations
            pyramidal = self.ema_pyramidal(self.activation(pyramidal))  # apply nonlinearity
            
            # Update with current timestep HVA activations
            if train:
                self.update(stim, pyramidal, pv, hva) # update weights based on local learning rules
            
            # store activations for this timestep
            out['Pyramidal'][:, t] = pyramidal  # store all pyramidal activations
            out['PV'][:, t] = pv  # store all PV activations
            out['HVA'][:, t] = hva  # store all HVA activations
            
            # Next timestep HVA neuron activations based on current pyramidal activations
            hva = self.ema_hva(self.activation(nonnegative(self.W_FFh)*self.mask_FFh @ pyramidal))
        self._reset_state()
        return out

    @torch.no_grad()
    def update(self, stim:torch.Tensor, pyramidal:torch.Tensor, pv:torch.Tensor, hva:torch.Tensor):
        '''
        To start with, only update
            W_Iy (Hebbian in Y & PC)
            W_FFy (Anti-Hebbian in Y & I)
            W_FBy (Hebbian in Y & HVA)
            W_FFpv (Hebbian in PV & Input)
        
        Input args:
            stim: current input stimulus (n_inputs)
            pyramidal: current activations of pyramidal neurons (n_pyramidal)
            pv: current activations of PV neurons (n_pv)
            hva: current activations of HVA neurons (n_hva)
        '''
        # Hebbian delta W_Iy (masked)
        self.W_Iy += self.lr_Iy * torch.outer(pyramidal, pv) * self.mask_Iy
        # Hebbian delta W_FFpv (masked) - inhibition becomes more specific to inputs that drive PV activation
        self.W_FFpv += self.lr_FFpv * torch.outer(pv, stim) * self.mask_FFpv
        
        # Anti-Hebbian delta W_FFy (masked) - adaptation
        self.W_FFy -= self.lr_FFy * torch.outer(pyramidal, stim) * self.mask_FFy
        
        if self.feedback_rule == 'Hebbian':
            # Hebbian delta W_FBy (masked)
            context = 0.8 * torch.outer(torch.ones_like(pyramidal), hva) # general context
            context += 0.2 * torch.outer(pyramidal, hva) # synapse-specific context
            self.W_FBy += self.lr_FBy * context * self.mask_FBy
        else:
            # Anti-Hebbian delta W_FBy (masked)
            context = torch.outer(pyramidal, hva) # feature-specific context
            context /= (pyramidal[:, None] + 1) # modulated by postsynaptic activity
            self.W_FBy += self.lr_FBy * context * self.mask_FBy

        
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
    
    def _reset_state(self):
        self.ema_pv.reset_state()
        self.ema_pyramidal.reset_state()
        self.ema_hva.reset_state()