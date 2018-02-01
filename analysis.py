import pickle
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, RADIUS, epochs, batch_size
from src.model import Keypoints
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# model
keypoints = Keypoints(NUM_CLASSES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('../character_checkpoints/model_1_18_4.pth'))

# cuda
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    torch.cuda.set_device(2)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

img = Image.open('../data/test_cropped_humans/0.jpg')
img = np.array(img)
img_t = transform(img)
img_t = img_t.cuda()
result, keypoints = prediction.predict(img_t)

keypoints = keypoints.cpu().numpy()
prediction.plot(img, result, keypoints[0])
