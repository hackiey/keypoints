import torch
import torchvision
import torch.nn as nn

class Keypoints(nn.Module):
    def __init__(self, num_classes, img_height=353, img_width=257, resnet=18):
        super(Keypoints, self).__init__()
        
        self.num_classes = num_classes
        self.num_outputs = num_classes * 3
        self.img_height = img_height
        self.img_width = img_width
        
        if resnet == 18:
            self.resnet = torchvision.models.resnet18()
            self.conv1by1 = nn.Conv2d(512, self.num_outputs, (1,1))
        elif resnet == 101:
            self.resnet = torchvision.models.resnet101()
            self.conv1by1 = nn.Conv2d(2048, self.num_outputs, (1,1))

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.resnet = self.resnet
            
        self.conv_transpose = nn.ConvTranspose2d(self.num_outputs, self.num_outputs, kernel_size=32, stride=8)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.conv1by1(x)
        x = self.conv_transpose(x)
        output = nn.Upsample(size=(self.img_height, self.img_width), mode='bilinear')(x)
        
        maps = self.sigmoid(output[:,:self.num_classes, :, :])
        offsets_x = output[:, self.num_classes:2*self.num_classes, :, :]
        offsets_y = output[:, 2*self.num_classes:3*self.num_classes, :, :]
        
        maps_pred = self.sigmoid(x[:,:self.num_classes, :, :])
        offsets_x_pred = x[:, self.num_classes:2*self.num_classes, :, :]
        offsets_y_pred = x[:, 2*self.num_classes:3*self.num_classes, :, :]

        return (maps, offsets_x, offsets_y), (maps_pred, offsets_x_pred, offsets_y_pred)
