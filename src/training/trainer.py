import os
from tkinter import W
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from zmq import device
from utils.utils import Utils
from metrics.metrics import Metrics
from data.dataset import ImageDataset
from torch.utils.data import DataLoader
from losses.loss import PerceptualLoss


class Trainer():
    def __init__(self,device, df_train, df_valid, w, h, batch_size, num_epochs,tsfm, model, lr,num_workers, outputs_dir, best_score=None):
        self.device = device
        self.df_train = df_train
        self.df_valid = df_valid
        self.epochs = num_epochs
        self.model = model
        self.model = model.to(self.device)
        self.w = w
        self.h = h
        self.batch_size = batch_size
        self.tsfm = tsfm
        self.lr = lr
        self.num_workers = num_workers
        self.output_dir = outputs_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.output_weights_dir = os.path.join(self.output_dir, "weights")
        if not os.path.exists(self.output_weights_dir):
            os.mkdir(self.output_weights_dir)
        self.train_dataset=ImageDataset(df=self.df_train, w=self.w, h=self.h, tsfm=self.tsfm)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,batch_size=batch_size,shuffle=True,
                                           num_workers=self.num_workers,pin_memory=False)
        self.valid_dataset=ImageDataset(df=self.df_valid, w=self.w, h=self.h, tsfm=self.tsfm)
        self.valid_dataloader = DataLoader(dataset=self.valid_dataset,batch_size=batch_size,shuffle=True,
                                           num_workers=self.num_workers,pin_memory=False)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.best_score = best_score if best_score is not None else 0.0
        
        
    def fit(self):
        for epoch in tqdm.tqdm(range(self.epochs)):
            train_loss = []
            train_acc = []
            valid_loss = []
            valid_acc = []
            
            self.model.train()
            for i, (images, masks) in enumerate(self.train_dataloader):
                images=images/255
                images.to(self.device)
                masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = Metrics.compute_loss(outputs,masks)
                print(loss)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                acc=Metrics.acc_camvid(outputs,masks)
                train_acc.append(acc.item())
            
            self.model.eval()
            for i, (images, masks) in enumerate(self.valid_dataloader):
                images=images/255

                images.to(self.device)
                masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.encoder_decoder(images)
               
                loss = Metrics.compute_loss(outputs,masks)
                valid_loss.append(loss.item())
                
                acc=Metrics.acc_camvid(outputs,masks)
                valid_acc.append(acc.item())
            
            avg_train_loss = np.average(train_loss)
            avg_train_acc = np.average(train_acc)
            Utils.print_metrics(avg_train_loss,avg_train_acc)
            
            avg_valid_loss = np.average(valid_loss)
            avg_valid_acc = np.average(valid_acc)
            Utils.print_metrics(avg_valid_loss,avg_valid_acc)
            
            if(avg_valid_loss<best_loss):
                torch.save(self.model.state_dict(),os.path.join(self.output_weights_dir,f"Deep_Drive_epoch{epoch}.pth"))
                best_loss = avg_valid_loss
            
        return self.model
  