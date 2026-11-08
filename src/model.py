import torch
import torch.nn as nn
from pathlib import Path

def load_model():
    model=nn.Sequential(
        nn.Conv2d(1,16, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16,32,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*7*7,128),
        nn.ReLU(),
        nn.Linear(128,10)
    )

    BASE_DIR=Path(__file__).resolve().parents[1]
    model_path = BASE_DIR/"model"/"mnist_cnn_best.pth"

    state_dict=torch.load(model_path,
                          map_location="cpu",
                          weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model