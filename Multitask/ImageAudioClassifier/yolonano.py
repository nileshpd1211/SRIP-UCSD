import torch
import torch.nn as nn
import torch.nn.functional as F

from yolonanoutils import *


class YOLONano(nn.Module):
    def __init__(self, image_size=416):
        super(YOLONano, self).__init__()
        self.image_size = image_size

        
        # image:  416x416x3
        self.features1 = nn.Sequential(
        conv3x3(3, 12, stride=1), # output: 416x416x12
        conv3x3(12, 24, stride=2), # output: 208x208x24
        PEP(24, 24, 7, stride=1), # output: 208x208x24
        EP(24, 70, stride=2), # output: 104x104x70
        PEP(70, 70, 25, stride=1), # output: 104x104x70
        PEP(70, 70, 24, stride=1), # output: 104x104x70
        EP(70, 150, stride=2), # output: 52x52x150
        PEP(150, 150, 56, stride=1), # output: 52x52x150
        conv1x1(150, 150, stride=1), # output: 52x52x150
        FCA(150, 8), # output: 52x52x150
        PEP(150, 150, 73, stride=1), # output: 52x52x150
        PEP(150, 150, 71, stride=1), # output: 52x52x150
        )
        
        self.features2 = nn.Sequential(
        PEP(150, 150, 75, stride=1), # output: 52x52x150
        EP(150, 325, stride=2), # output: 26x26x325
        PEP(325, 325, 132, stride=1), # output: 26x26x325
        PEP(325, 325, 124, stride=1), # output: 26x26x325
        PEP(325, 325, 141, stride=1), # output: 26x26x325
        PEP(325, 325, 140, stride=1), # output: 26x26x325
        PEP(325, 325, 137, stride=1), # output: 26x26x325
        PEP(325, 325, 135, stride=1), # output: 26x26x325
        PEP(325, 325, 133, stride=1), # output: 26x26x325
        )
        
        self.features3 = nn.Sequential(
        PEP(325, 325, 140, stride=1), # output: 26x26x325
        EP(325, 545, stride=2), # output: 13x13x545
        PEP(545, 545, 276, stride=1), # output: 13x13x545
        conv1x1(545, 230, stride=1), # output: 13x13x230
        EP(230, 489, stride=1), # output: 13x13x489
        PEP(489, 469, 213, stride=1), # output: 13x13x469
        conv1x1(469, 189, stride=1), # output: 13x13x189
        conv1x1(189, 105, stride=1), # output: 13x13x105
        )
        
        
        
        
        # upsampling conv6 to 26x26x105
        # concatenating [conv6, pep15] -> pep18 (26x26x430)
        
        # upsampling conv8 to 52x52x47
        # concatenating [conv8, pep7] -> pep20 (52x52x197)
#         PEP(197, 122, 58, stride=1) # output: 52x52x122
#         PEP(122, 87, 52, stride=1) # output: 52x52x87
#         PEP(87, 93, 47, stride=1) # output: 52x52x93
# #         self.conv9 = conv1x1(93, self.yolo_channels, stride=1, bn=False) # output: 52x52x yolo_channels
# #         self.yolo_layer52 = YOLOLayer(anchors52, num_classes, img_dim=image_size)

#         # conv7 -> ep6
#         self.ep6 = EP(98, 183, stride=1) # output: 26x26x183
# #         self.conv10 = conv1x1(183, self.yolo_channels, stride=1, bn=False) # output: 26x26x yolo_channels
# #         self.yolo_layer26 = YOLOLayer(anchors26, num_classes, img_dim=image_size)

#         # conv5 -> ep7
#         self.ep7 = EP(189, 462, stride=1) # output: 13x13x462
# #         self.conv11 = conv1x1(462, self.yolo_channels, stride=1, bn=False) # output: 13x13x yolo_channels
# #         self.yolo_layer13 = YOLOLayer(anchors13, num_classes, img_dim=image_size)

        
                
        self.classifier1a = nn.Sequential(
            PEP(105, 50, 58, stride=1), # output: 52x52x122
            PEP(50, 25, 52, stride=1), # output: 52x52x87
            PEP(25, 5, 47, stride=1), # output: 52x52x93
        )
        
        self.classifier1b = nn.Sequential(
            nn.Linear(845, 30),
#             nn.Linear(64, 30),
        )
        
        self.classifier2a = nn.Sequential(
            PEP(105, 50, 58, stride=1), # output: 52x52x122
            PEP(50, 25, 52, stride=1), # output: 52x52x87
            PEP(25, 5, 47, stride=1), # output: 52x52x93
        )
        
        self.classifier2b = nn.Sequential(
            nn.Linear(845,10),
#             nn.Linear(64, 10),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x):
        AudioIsTrue=0
        if(x.shape[1]==1):
            x = torch.cat(3*[x],1)
            AudioIsTrue=1
                
        x = F.interpolate(x,(416))

        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        if(AudioIsTrue):
            x = self.classifier1a(x)
            x = x.view(x.size(0), -1)
            x = self.classifier1b(x)
            return  F.log_softmax(x, dim=1)
        else:
            x = self.classifier2a(x)
            x = x.view(x.size(0), -1)
            x = self.classifier2b(x)
            return x
    
        
    def name(self):
        return "YoloNano"

if __name__ == '__main__':

    # ---------test class
    a = 1