import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ImageDataset(Dataset):
  def __init__(self,df,w,h,tsfm):
    self.w = w
    self.h = h
    self.dataset=df.values
    self.tsfm = tsfm

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self,idx):
    img=cv2.imread(self.dataset[idx][0])
    msk=cv2.imread(self.dataset[idx][1])[:,:,0]
    img=cv2.resize(img,(self.w,self.h))
    msk=cv2.resize(msk,(self.image_size,self.image_size))
    img=np.transpose(img,(2,0,1))
    if self.tsfm is not None:
        X = self.tsfm(img)
    X=torch.cuda.FloatTensor(img)
    y=torch.cuda.LongTensor(msk)
    return X,y