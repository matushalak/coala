# author: Matúš Halák (@matushalak)
import torch
import matplotlib.pyplot as plt
from utils import randn_reparam
def normal_vs_lognormal():
    normal = torch.randn(10000)
    lognorml = torch.exp(normal)
    lognorml = lognorml[lognorml < 3]
    normal2 = torch.randn(20000)
    lognorml2 = torch.exp(normal2)
    lognorml2= lognorml2[lognorml2 < 3]
    lognorml3 = lognorml * 0.5
    plt.hist(normal.numpy(), bins=100, label='Standard Normal')
    plt.hist(lognorml.numpy(), alpha=0.5, bins=100, label='Log-Normal')
    plt.hist(lognorml2.numpy(), alpha=0.5, bins=100, label='Log-Normal 2')
    plt.hist(lognorml3.numpy(), alpha=0.5, bins=100, label='Log-Normal 3')
    plt.title("Histogram of 10000 samples from torch.randn")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def reparametrization_trick(mu, sigma):
    mu = torch.tensor(mu)
    sigma = torch.tensor(sigma)
    x = randn_reparam((10000,), mu, sigma)
    if x.shape[1] == 2:
        plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.5, label='Reparameterized Normal')
        plt.title("Scatter plot of 10000 samples from reparameterization trick")
        plt.ylim(-3, 3)
        plt.xlim(-3, 3)
    elif x.shape[1] == 1:
        plt.hist(x.numpy(), bins=100, label='Reparameterized Normal')
        plt.title("Histogram of 10000 samples from reparameterization trick")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    # normal_vs_lognormal()
    # reparametrization_trick([3/2, 3/2], [[1/10, 1/2], 
    #                                      [1/2, 1/10]])
    
    reparametrization_trick([34/59, 2], [ 2])