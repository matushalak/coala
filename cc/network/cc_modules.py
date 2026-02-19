import torch
import torch.nn as nn
import torch.nn.functional as F

class CCModule(nn.Module):
    '''
    Contextual Contrasting RNN module
    '''
    def __init__(self, spatial_dims:tuple[int, int], FF_conv:nn.Conv2d, FB_conv:nn.Conv2d, 
                 LAT_ksize:tuple[int, int] = (3,3), activation_fn:nn.Module = nn.ReLU()):
        super().__init__()
        # Initialize convolution layers for feedforward, feedback, and lateral inhibition
        self.FF_conv = FF_conv # pre-trained feedforward convolution layer (fixed weights)
        self.FB_conv = FB_conv # pre-trained feedback convolution layer (fixed weights)
        # Lateral inhibition implemented as a convolution with fixed weights (sum over local neighborhood)
        self.LAT_conv = lambda y: F.conv2d(
            y, weight=torch.ones(size=(FF_conv.out_channels, FF_conv.out_channels, *LAT_ksize),requires_grad=False),
            padding = (LAT_ksize[0]//2, LAT_ksize[1]//2))
        
        # Define dynamic alphas
        self.Lambda_FF = nn.Parameter(torch.zeros(1, FF_conv.out_channels, *spatial_dims), requires_grad=False)
        self.Lambda_FB = nn.Parameter(torch.zeros(1, FF_conv.out_channels, *spatial_dims), requires_grad=False)
        self.Lambda_LAT = nn.Parameter(torch.zeros(1, FF_conv.out_channels, *spatial_dims), requires_grad=False)

        # activation function (e.g., ReLU)
        self.activation_fn = activation_fn

        self.Y_LAT, self.Y_FF, self.Y_FB = torch.zeros_like(self.Lambda_FF), torch.zeros_like(self.Lambda_FF), torch.zeros_like(self.Lambda_FF) 
    
    def forward(self, x:torch.Tensor, context:torch.Tensor, train:bool = False)->torch.Tensor:
        # Compute feedforward, feedback, and lateral inhibition contributions
        self.Y_LAT = self.LAT_conv(self.Y_FF)
        self.Y_FF = self.FF_conv(x)
        self.Y_FB = self.FB_conv(context)
        
        # Combine contributions with dynamic lambdas
        Y = (1+self.Lambda_FF) * self.Y_FF + (1+self.Lambda_FB) * self.Y_FB - (1+self.Lambda_LAT) * self.Y_LAT

        # Apply activation function
        Y = self.activation_fn(Y)

        if train:
            self.update_lambdas(Y)
        
        return Y
    
    @torch.no_grad()
    def update_lambdas(self, Y:torch.Tensor, lr_FF:float = 0.01, lr_FB:float = 0.01, lr_LAT:float = 0.01):
        self.Lambda_FF += lr_FF * (-self.Lambda_FF - (Y * self.Y_FF))
        self.Lambda_FB += lr_FB * (-self.Lambda_FB + ((1/(1+Y))) * (Y * self.Y_FB))
        self.Lambda_LAT += lr_LAT * (-self.Lambda_LAT + (Y * self.Y_LAT))


if __name__ == "__main__":
    # Example usage
    spatial_dims = (32, 32) # example spatial dimensions
    FF_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    FB_conv = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
    
    cc_module = CCModule(spatial_dims, FF_conv, FB_conv)
    input_tensor = torch.randn(1, 3, *spatial_dims) # example input
    hidden_tensor = torch.randn(1, 32, *spatial_dims) # example hidden state
    output = cc_module(input_tensor, hidden_tensor)
    print(output.shape)