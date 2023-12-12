import torch


# define device
DEVICE = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# define training parameter
LR = 0.005
IMG_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 64
EMBEDDING_DIM = 128
EPOCHS = 50
KLD_WEIGHT = 0.00025

# define the dataset path
DATASET_PATH = "data/img_align_celeba"
DATASET_ATTRS_PATH = "data/list_attr_celeba.csv"

NUM_FRAMES = 50
FPS = 5

LABELS = ["Eyeglasses", "Smiling", "Attractive", "Male", "Blond_Hair"]