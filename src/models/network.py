import torch
import torch.nn as nn
from models.modules.resnet import encoder
from models.modules.resnet import res_block
from models.modules.unet_block import unet_block

class Net(nn.Module):
    def __init__(self,num_classes, encoder_path):
        super(Net, self).__init__()
        self.encoder = encoder(res_block,[3,4,6,3],3)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))
        self.in_channels=1024
        self.bottle_conv1=nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1)
        self.bottle_conv2=nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        self.unet_block1=unet_block(1024,256,256)
        self.unet_block2=unet_block(512,128,128)
        self.unet_block3=unet_block(256,64,64)
        self.unet_block4=unet_block(128,64,64)
        self.conv_transpose=nn.ConvTranspose2d(128,128,kernel_size=2,stride=2,padding=0)
        self.out1=nn.Conv2d(128,num_classes,kernel_size=1,stride=1,padding=0)
        
    def forward(self,x):
        x_skip,skip1,skip2,skip3,skip4=self.encoder(x)
        x=self.bottle_conv1(x_skip)
        x=self.relu(x)
        x=self.bottle_conv2(x)
        x=self.relu(x)
        x=nn.ReLU()(torch.cat([x,x_skip],dim=1))
        x=self.unet_block1(x,skip4)
        x=self.unet_block2(x,skip3)
        x=self.unet_block3(x,skip2)
        x=self.unet_block4(x,skip1)
        x=self.conv_transpose(x)
        out1=self.out1(x)
        return out1
