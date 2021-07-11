import torch
from torch import nn 
from torchvision import transforms 
from PIL import Image
from matplotlib import pyplot as plt
import argparse
import time 
from network import Generator

if __name__ == "__main__":

    # Parse commandline argument
    # example command: python3 drawify.py --input thispersondoesnotexist.jpg --model_path pretrained_model/cycleGAN_52000.pth
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='thispersondoesnotexist.jpg', type=str, help='input face image. e.g. abc.jpg')
    parser.add_argument('--model_path', default='pretrained_model/cycleGAN_52000.pth', type=str, help='path to model weights (file pth)')
    args = parser.parse_args()

    # input image resolution
    load_shape = 256
    target_shape = 256
    
    # as input is RGB image. dim = 3
    image_dimension = 3
    # necessary image transformations 
    transform = transforms.Compose([
        transforms.Resize(load_shape),
        transforms.CenterCrop(load_shape),
        transforms.ToTensor(),
    ])

    timeit = time.time()

    # Load backbone of cycle GAN generator and restore model weights
    device = 'cpu'
    generator = Generator(image_dimension, image_dimension).to(device=device)
    pretrained_model = torch.load(args.model_path, map_location=torch.device(device))
    generator.load_state_dict(pretrained_model['gen_AB'])

    # load_image to preprocess image. 
    # Note: output of generator (last activation function is a Tanh) carries value between -1 to 1.
    # We wants to normalize gan input to -1 to 1. Hence the following codes
    face = transform(Image.open(args.input)).to(device)
    face = (face - 0.5) * 2
    face = torch.unsqueeze(face, 0)
    face = nn.functional.interpolate(face, size=target_shape)
    
    with torch.no_grad():
        fake_draw = generator(face)
        fake_draw = (fake_draw + 1) / 2

    print(f'drawify took {time.time() - timeit} seconds')
    # show image
    plt.imshow(fake_draw[0].permute(1, 2, 0).squeeze(), interpolation='nearest', aspect='auto')
    plt.axis('off')
    plt.show()
