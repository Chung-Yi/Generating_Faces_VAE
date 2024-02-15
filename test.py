import torch
import cv2
from utils import network
from configuration import config
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def main():
    model_path = 'output/model_weights/best_vae_celeba.pt'
    image_path = 'data/test/test1.jpeg'

    model = network.CelebVAE(config.CHANNELS, config.EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps'))['vae-celeba'])

    
    image_transforms = transforms.Compose([
        transforms.ToPILImage()
    ])

    image = Image.open(image_path)
    image = image_transforms(image)

    image = image.view(1, 3, config.IMG_SIZE, config.IMG_SIZE)

    output = model.encode(image)[0]
    # output = output.squeeze(0)

    print(output.shape)

    pil_image = transforms.ToPILImage()(output)
    pil_image.show()

    # output = output.detach().numpy() 

    



    # cv2.imshow('My Image', output)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.imshow(output.detach().numpy())
    # plt.show()


if __name__ == "__main__":
    main()