from utils import config
import numpy as np
import itertools
import os
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import utils
from model.target import Target
from imageio import imread


def restore_model(model, checkpoint_dir, inp_epoch):
    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    # print(checkpoint["state_dict"])
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        print("=> Checkpoint not successfully restored")
        raise
    return model

def normalized(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std

def prediction(logits):
    return torch.argmax(torch.squeeze(softmax(logits, dim=1)))

def predict(image, epoch):
    model = Target()
    model = restore_model(model, config("target.checkpoint"), epoch)
    image = normalized(image)
    image = torch.tensor(np.expand_dims(image, axis=(0,1)), dtype=torch.float)
    output = model(image)
    final_val = prediction(output.data).item()
    return final_val

def predictDict(image, epoch):
    model = Target()
    model = restore_model(model, config("target.checkpoint"), epoch)
    image = normalized(image)
    image = torch.tensor(np.expand_dims(image, axis=(0,1)), dtype=torch.float)
    output = model(image)
    final_dict = torch.squeeze(softmax(output.data, dim=1))
    return final_dict

# print(imread("data/mnist_png/Hnd/Sample4/53.png"))






