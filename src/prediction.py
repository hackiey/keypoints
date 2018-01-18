import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
class Prediction:
    def __init__(self, model, num_classes, img_height, img_width, img_small_height, img_small_width, use_cuda):
        self.model = model
        
        self.num_classes = num_classes
        self.img_height  = img_height
        self.img_width   = img_width
        
        self.img_small_height = img_small_height
        self.img_small_width  = img_small_width
        
        self.use_cuda = use_cuda
        
        self.offset_x_ij = torch.arange(0, self.img_small_width) \
            .repeat(self.img_small_height).view(1,1,self.img_small_height, self.img_small_width)
        self.offset_y_ij = torch.arange(0, self.img_small_height) \
            .repeat(self.img_small_width).view(self.img_small_width, self.img_small_height).t().contiguous() \
            .view(1,1,self.img_small_height, self.img_small_width)
        
        if self.use_cuda:
            self.offset_x_ij = self.offset_x_ij.cuda()
            self.offset_y_ij = self.offset_y_ij.cuda()
        
        self.offset_x_add = (0 - self.offset_x_ij).view(self.img_small_height, self.img_small_width, 1, 1)
        self.offset_y_add = (0 - self.offset_y_ij).view(self.img_small_height, self.img_small_width, 1, 1)
        
        self.offset_x_ij = (self.offset_x_ij + self.offset_x_add) * self.img_width / self.img_small_width
        self.offset_y_ij = (self.offset_y_ij + self.offset_y_add) * self.img_height/ self.img_small_height
        
    def predict(self, imgs):
        # img: torch.Tensor(3, height, width)
        imgs = imgs.view(-1, imgs.shape[0], imgs.shape[1], imgs.shape[2])     
        result, (maps_pred, offsets_x_pred, offsets_y_pred) = self.model.forward(Variable(imgs))
        maps_pred = maps_pred.data
        offsets_x_pred = offsets_x_pred.data
        offsets_y_pred = offsets_y_pred.data
        keypoints = []
        
        for i in range(maps_pred.shape[0]):
            for k in range(self.num_classes):
                offsets_x_ij = self.offset_x_ij + offsets_x_pred[i][k]
                offsets_y_ij = self.offset_y_ij + offsets_y_pred[i][k]
                distances_ij = torch.sqrt(offsets_x_ij * offsets_x_ij + offsets_y_ij * offsets_y_ij)

                distances_ij[distances_ij > 0.9] = 1
                distances_ij = 1 - distances_ij         
                score_ij = (distances_ij * maps_pred[i][k]).sum(3).sum(2)

                v1,index_y = score_ij.max(0)
                v2,index_x = v1.max(0)

                if self.use_cuda:
                    keypoints.append([index_y[index_x].cpu().numpy()[0], index_x.cpu().numpy()[0]])
                else:
                    keypoints.append([index_x.numpy()[0], index_y[index_x].numpy()[0]]) 

        if self.use_cuda:
            maps_array = result[0].cpu().data.numpy()
            offsets_x_array = result[1].cpu().data.numpy()
            offsets_y_array = result[2].cpu().data.numpy()
        else:
            maps_array = result[0].data.numpy()
            offsets_x_array = result[1].data.numpy()
            offsets_y_array = result[2].data.numpy()
            
        return (maps_array, offsets_x_array, offsets_y_array), keypoints
    
    def plot(self, plt_img, result, keypoints):
        
        maps_array = result[0]
        offsets_x_array = result[1]
        offsets_y_array = result[2]
        
        plt.imshow(plt_img)
        
        for i in range(NUM_CLASSES):

            heatmap = plt_img.copy()            
            plt.figure(figsize=(12, 9))
            
            # heatmap
            plt.subplot(1, 4, 1)
            plt.title(str(i))
            indexes = maps_array[0][i] > np.percentile(maps_array[0][i], 98.5)
            indexes = maps_array[0][i] > 0.5
            heatmap[indexes] = maps_array[0][i].repeat(3).reshape((self.img_height, self.img_width, 3))[indexes]
            plt.imshow(heatmap)
            
            # offsets
            
            offsets = np.sqrt(offsets_x_array[0][i] * offsets_x_array[0][i] + offsets_y_array[0][i] * offsets_y_array[0][i])
            offsets_repeated = offsets.repeat(3)
            
            plt.subplot(1, 4, 2)
            plt.title(str(i))
            offsets_array = offsets_repeated.reshape((self.img_height, self.img_width, 3))
            offsets_array = offsets_array / offsets_array.max()
            plt.imshow(offsets_array)
            
            # offsets disk
            plt.subplot(1, 4, 3)
            plt.title(str(i))
            offsets_array = np.zeros((self.img_height, self.img_width, 3))
            offsets_array[indexes] = offsets_repeated.reshape((self.img_height, self.img_width, 3))[indexes]
            offsets_array = offsets_array / offsets_array.max()
            plt.imshow(offsets_array)
            
            # final result
            plt.subplot(1, 4, 4)
            plt.imshow(plt_img)
            plt.scatter(keypoints[i][1] * self.img_height / self.img_small_height * 0.91, keypoints[i][0] * self.img_width / self.img_small_width * 1.1)
            plt.show()
