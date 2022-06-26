import torch

void_code = 0

class Metrics:

    @classmethod
    def acc_camvid(cls,input, target):
      mask = target != void_code
      return (torch.argmax(input, dim=1)[mask]==target[mask]).float().mean()
        