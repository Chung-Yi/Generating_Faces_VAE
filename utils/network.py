import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        return self.block(x)
    
class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)
    
class CelebVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims=None) -> None:
        super(CelebVAE, self).__init__()
        self.laten_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.encoder = nn.Sequential(
            *[
                ConvBlock(in_f, out_f) for in_f, out_f in zip([in_channels] + hidden_dims[:-1], hidden_dims)
            ]
        )
        
        # fully connected layer for the mean of the latent space
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        # fully connected layer for the variance of the latent space
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # build the decoder using transposed convolutional blocks
        # fully connected layer to expand the latent space
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        self.decoder = nn.Sequential(
            *[
                # create a convtblock for each pair of input and output channels
                ConvTBlock(in_f, out_f)
                for in_f, out_f in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3, 
                stride=2,
                padding=1,
                output_padding=1,
            ),

            # normalize the activations of the layer
            nn.BatchNorm2d(hidden_dims[-1]),
            # apply leaky relu activation
            nn.LeakyReLU(),
            # final convolution to match the output channels
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            # apply tanh activation to scale the output
            nn.Tanh()
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return std * eps + mu
    
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
        