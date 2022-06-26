import torch
import torch.nn as nn
from models.modules.attention_net import attention
from fastai.layers import PixelShuffle_ICNR

class unet_block(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels):
        super(unet_block,self).__init__()
        final_channel=out_channels+skip_channels
        self.Pixelshuf=PixelShuffle_ICNR(in_channels,out_channels)
        self.conv=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.attention=attention(out_channels)
        self.up_conv1=nn.Conv2d(final_channel,final_channel,kernel_size=3,stride=1,padding=1)
        self.up_conv2=nn.Conv2d(final_channel,final_channel,kernel_size=3,stride=1,padding=1)
        self.relu=nn.ReLU()
        
    def forward(self,x,skip):
        x=self.Pixelshuf(x)
        x=self.conv(x)
        skip=self.attention(skip,x)
        x=torch.cat([x, skip],dim=1)
        x=self.up_conv1(x)
        x=self.relu(x)
        x=self.up_conv2(x)
        x=self.relu(x)
        return x