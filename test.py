import pickle
import torch
from torch.utils.data import DataLoader
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, epochs, batch_size
from src.model import Keypoints
from src.dataset import KeypointsDataset
from src.prediction import Prediction

# dataset
with open('../data/annotation/annotation_test_cropped_humans.pkl', 'rb') as f:
    labels = pickle.load(f)

# dataset
dataset = KeypointsDataset('../data/test_cropped_humans/', labels, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# model
keypoints = Keypoints(NUM_CLASSES)

# cuda
use_cuda = torch.cuda.is_available()
# use_cuda = False
if use_cuda:
    torch.cuda.set_device(2)
    keypoints = keypoints.cuda()

prediction = Prediction(deep_keypoints, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, IMG_SMALL_HEIGHT, IMG_SMALL_WIDTH, use_cuda)

error = 0
for i_batch, sample_batched in enumerate(dataloader):

    X = sample_batched[0]
    label_batched = sample_batched[2]

    X = X.cuda()
    result, keypoints = prediction.predict(img)

    break