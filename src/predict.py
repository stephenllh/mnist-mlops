import numpy as np
import torch
from torchvision import transforms
from src.net import Net


def predict(img_numpy, ckpt_path):
    transform = transforms.Compose([transforms.ToTensor()])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_torch = transform(img_numpy.astype(np.float32))
    img_torch = img_torch.unsqueeze(dim=0).to(device)

    net = Net().to(device)
    net.eval()
    net.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))  
    log_preds = net(img_torch)
    return log_preds
