# builds vae from cnn blocks
# same as AE, except with same parameters, latent dimension is 2*latent dimension
# and parametrizes both mean and variance of latent distribution; 
# latent activaations then are samples from this distribution, same
# dimension as latent dimension param in AE