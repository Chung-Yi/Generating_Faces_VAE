import matplotlib
import numpy as np
import torch
from PIL import Image
from configuration import config

matplotlib.use("Agg")

import imageio
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def loss_function(VAELossParams, kld_weight):
    recons, input, mu, log_var = VAELossParams
    recons_loss = F.mse_loss(recons, input)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )

    loss = recons_loss + kld_weight * kld_loss
    return {
        "loss": loss,
        "Reconstruction_Loss": recons_loss.detach(),
        "KLD": -kld_loss.detach()
    }

def validate(model, val_dataloader, device):
    running_loss = 0.0

    for i, x in enumerate(val_dataloader):

        x = x.to(device)
        predictions = model(x)

        total_loss = loss_function(predictions, config.KLD_WEIGHT)

        running_loss += total_loss['loss'].item()

    return running_loss