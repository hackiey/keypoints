import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, epochs, batch_size
from src.model import Keypoints
from src.dataset import KeypointsDataset, transform

def custom_loss(predictions_maps, maps, predictions_offsets_x, offsets_x, predictions_offsets_y, offsets_y):
    
    loss_h = bceLoss(predictions_maps, maps)

    distance_x = predictions_offsets_x[maps==1] - offsets_x[maps==1]
    distance_y = predictions_offsets_y[maps==1] - offsets_y[maps==1]
    distances = torch.sqrt(distance_x * distance_x + distance_y * distance_y)
    zero_distances = Variable(
        torch.zeros(distance_x.shape).cuda() if use_cuda else torch.zeros(distance_x.shape))
    loss_o = smoothL1Loss(distances, zero_distances)
    loss = 4 * loss_h + loss_o
    return loss

def forward(sample_batched, model):
    X = sample_batched[0]
    maps, offsets_x, offsets_y = sample_batched[1]

    maps = Variable(maps.cuda() if use_cuda else maps)
    offsets_x = Variable(offsets_x.cuda() if use_cuda else offsets_x)
    offsets_y = Variable(offsets_y.cuda() if use_cuda else offsets_y)

    # forward
    X = Variable(X.cuda() if use_cuda else X)
    (predictions_maps, predictions_offsets_x, predictions_offsets_y), pred = model.forward(X)
    
    return custom_loss(predictions_maps, maps, predictions_offsets_x, offsets_x, predictions_offsets_y, offsets_y)

def fit(train_data, test_data, model, loss_function, epochs, checkpoint_path = ''):
    for epoch in range(epochs):
        # training 
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(train_data):
            optimizer.zero_grad()
            
            loss = forward(sample_batched, model)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]

            print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.data[0]), end='')
            print('\r', end='')
        print('train loss:', train_loss / i_batch)
        
        test_loss = 0.0
        for i_batch, sample_batched in enumerate(test_data):
            loss = forward(sample_batched, model)
            test_loss += loss.data[0]
        print('test loss:', test_loss / i_batch)
        
        torch.save(keypoints.state_dict(), checkpoint_path + 'model_2_1_' + str(epoch)+'.pth')

# dataset
with open('../data/annotation/annotation_train_cropped_humans.pkl', 'rb') as f:
    train_labels = pickle.load(f)
train_dataset = KeypointsDataset('../data/train_cropped_humans/',
                           train_labels, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, transform=transform)
train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

with open('../data/annotation/annotation_test_cropped_humans.pkl', 'rb') as f:
    test_labels = pickle.load(f)
test_dataset = KeypointsDataset('../data/test_cropped_humans/',
                           test_labels, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS, transform=transform)
test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

use_cuda = torch.cuda.is_available()
# use_cuda = False
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if use_cuda:
    torch.cuda.set_device(2)

# loss
smoothL1Loss = nn.SmoothL1Loss()
bceLoss = nn.BCELoss()
# model
keypoints = Keypoints(NUM_CLASSES)
keypoints = keypoints.cuda() if use_cuda else keypoints
# optimizer
optimizer = optim.Adam(keypoints.parameters(), lr=0.0001)

fit(train_data, test_data, keypoints, custom_loss, epochs=200, checkpoint_path='../checkpoints/')
