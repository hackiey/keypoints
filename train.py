from torch.utils.data import Dataset, DataLoader
from config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, epochs, batch_size
from src.model import Keypoints
from src.dataset import KeypointsDataSet

# dataset
with open('../data/annotation/annotation_train_cropped_humans.pkl', 'rb') as f:
    labels = pickle.load(f)
dataset = KeypointsDataSet('../data/train_cropped_humans/', labels, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, RADIUS)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# model
keypoints = Keypoints(NUM_CLASSES)
for epoch in range(epochs):
    running_loss = 0.0
    for i_batch, sample_batched in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        X = sample_batched[0]
        maps, offsets_x, offsets_y = sample_batched[1]

        maps = Variable(maps.cuda() if use_cuda else maps)
        offsets_x = Variable(offsets_x.cuda() if use_cuda else offsets_x)
        offsets_y = Variable(offsets_y.cuda() if use_cuda else offsets_y)
        
        # forward
        X = Variable(X.cuda() if use_cuda else X)
        (predictions_maps, predictions_offsets_x, predictions_offsets_y), pred = keypoints.forward(X)
        
        # loss
        loss_h = bceLoss(predictions_maps, maps)
        
        distance_x = predictions_offsets_x[maps==1] - offsets_x[maps==1]
        distance_y = predictions_offsets_y[maps==1] - offsets_y[maps==1]
        distances = torch.sqrt(distance_x * distance_x + distance_y * distance_y)
        zero_distances = Variable(
            torch.zeros(distance_x.shape).cuda() if use_cuda else torch.zeros(distance_x.shape))
        loss_o = smoothL1Loss(distances, zero_distances)
        loss = 4 * loss_h + loss_o
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i_batch + 1, loss.data[0]), end='')
        print('\r', end='')
        # torch.save(keypoints.state_dict(), 'model_'+str(epoch)+'.pth')
    print(running_loss / i_batch)
    
    torch.save(keypoints.state_dict(), '../checkpoints/model_1_15_'+str(epoch)+'.pth')

