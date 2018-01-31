import pickle
import torch
from torch.utils.data import DataLoader
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, RADIUS, epochs, batch_size
from src.model import Keypoints
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

# model
keypoints = Keypoints(NUM_CLASSES)
keypoints.load_state_dict(torch.load('../checkpoints/model_3.pth'))

# cuda
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    torch.cuda.set_device(2)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, use_cuda)

img = Image.open('../data/test_cropped_humans/0.jpg')
img = transform(img)
img = img.cuda()

print(img.shape)
time1 = datetime.now()
result, keypoints = prediction.predict(img)
time2 = datetime.now()
print(time2 - time1)
print(keypoints)
img = img.cpu().numpy().transpose((1,2,0))
prediction.plot(img, result, keypoints)
