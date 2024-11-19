import random
import numpy as np
import math
import torch
import torch.nn as nn
from torchnet import FeatNet, MaskNet
import cv2


if __name__ == '__main__':
    featnet = FeatNet()
    masknet = MaskNet()
    featnet.load_state_dict(torch.load('featnet.pth'))
    masknet.load_state_dict(torch.load('masknet.pth'))

    img = cv2.imread('3.bmp',cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    img = img/255 - 0.25
    iriscode = featnet(img)
    iriscode = iriscode.detach().numpy().squeeze(0).squeeze(0)
    irismask = masknet(img)
    irismask = (irismask.detach().squeeze(0))[1,:,].squeeze(0).numpy()  
    
    cv2.imshow('iriscode.jpg',iriscode)
    cv2.imshow('irismask.jpg',irismask)
    cv2.waitKey()
