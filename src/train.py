import os
import torch
import warnings
from models.network import Net
from torchvision import transforms
from training.trainer import Trainer

warnings.filterwarnings("ignore",category=UserWarning)

train_stage = 1
dataset_dir = "dataset/train"
outputs_dir = "outputs/results/training"
weights_dir = "weights"
model_filename = None
weight_type = None
encoder_path = "weights/encoder_decoder.pth"
df_train = ""
df_valid = ""
w = ""
h = ""
batch_size = ""
num_epochs = 1
num_worker = 1
lr = 0.001
best_score = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tsfm = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                          ])
model = Net(num_classes=8 , encoder_path=encoder_path)

if model_filename is not None:
    model_path = os.path.join(weights_dir,model_filename)
    if weight_type == "model":
        model = torch.load(model_path)
    else:
        model.load_state_dict(torch.load(model_path))

model.to(device)
train = Trainer(device=device, df_train= df_train, df_valid=df_valid,
                w=w, h=h, batch_size=batch_size, num_epochs=num_epochs, 
                tsfm=tsfm, lr=lr, num_workers=num_worker,
                outputs_dir=outputs_dir,best_score=best_score)
train.fit()

torch.save(train.G,f"Generator_model_Deep_Drive_stage_{train_stage}.pth")
torch.save(train.D,f"Discriminator_model_Deep_Drive_stage_{train_stage}.pth")