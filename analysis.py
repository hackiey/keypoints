import pickle
import torch
from torch.utils.data import DataLoader
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, RADIUS, epochs, batch_size
from src.model import Keypoints
from src.dataset import KeypointsDataset
from src.prediction import Prediction
from datetime import datetime
import matplotlib.pyplot as plt

# dataset
with open('../data/annotation/annotation_test_cropped_humans.pkl', 'rb') as f:
    labels = pickle.load(f)

# dataset
dataset = KeypointsDataset('../data/test_cropped_humans/', labels, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

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
img, label = dataset[221]
img = img.cuda()
time1 = datetime.now()
result, keypoints = prediction.predict(img)
time2 = datetime.now()
print(time2 - time1)
print(keypoints)
img = img.cpu().numpy().transpose((1,2,0))
prediction.plot(img, result, keypoints)
