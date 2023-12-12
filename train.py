import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from configuration import config
from utils import dataset_utils, network, model_utils


# create output director
output_dir = "output"
# os.makedirs("output", exist_ok=True)

# create the training_progress directory inside the output directory
training_progress_dir = os.path.join(output_dir, "training_progress")
# os.makedirs(training_progress_dir, exist_ok=True)

# create the model_weights directory inside the output directory
# for storing autoencoder weights
model_weights_dir = os.path.join(output_dir, "model_weights")
# os.makedirs(model_weights_dir, exist_ok=True)

# define model_weights path including best weights
MODEL_BEST_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_vae_celeba.pt")
MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "vae_celeba.pt")


# Define the transformations
def train_test_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148),
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(148), 
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor()
    ])

    return train_transforms, val_transforms


def main():
    
    os.makedirs("output", exist_ok=True)
    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(model_weights_dir, exist_ok=True)

    train_transforms, val_transforms = train_test_transforms()
    celeba_dataset = dataset_utils.CelebADataset(config.DATASET_PATH, transform=train_transforms)

    val_size = int(len(celeba_dataset) * 0.1)
    train_size = len(celeba_dataset) - val_size

    train_dataset, val_dataset = random_split(celeba_dataset, [train_size, val_size])

    # Define the data loaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = network.CelebVAE(config.CHANNELS, config.EMBEDDING_DIM)
    model = model.to(config.DEVICE)

    # instantiate optimizer, and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    best_val_loss = float("inf")

    print("Training Started!!")


    # print("train_dataloader length: ", len(train_dataloader))
    # print("val_dataloader length: ", len(val_dataloader))

    # os._exit(0)
    

    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        for i, x in enumerate(train_dataloader):

            print("i: ", i)
            
            x =  x.to(config.DEVICE)
            optimizer.zero_grad()

            predictions = model(x)

            total_loss = model_utils.loss_function(predictions, config.KLD_WEIGHT)

            # backward pass
            total_loss['loss'].backward()

            # Optimizer variable updates
            optimizer.step()

            running_loss += total_loss['loss'].item()

        # compute average loss for the epoch
        training_loss = running_loss / len(train_dataloader)

    

        # compute validation loss for the epoch
        val_loss = model_utils.validate(model, val_dataloader, config.DEVICE)        

        # save best vae model weights based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"vae-celeba": model.state_dict()},
                MODEL_BEST_WEIGHTS_PATH
            )
        
        torch.save(
            {"vae-celeba": model.state_dict()},
            MODEL_WEIGHTS_PATH,
        )

        print(
            f"Epoch {epoch+1}/{config.EPOCHS}, Batch {i+1}/{len(train_dataloader)}, "
            f"Total Loss: {total_loss['loss'].detach().item():.4f}, "
            f"Reconstruction Loss: {total_loss['Reconstruction_Loss']:.4f}, "
            f"KL Divergence Loss: {total_loss['KLD']:.4f}",
            f"Val Loss: {val_loss:.4f}",
        )

        scheduler.step()


if __name__ == '__main__':
    main()