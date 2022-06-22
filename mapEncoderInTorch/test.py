import torch
from model_def import ModelNN, GainPredictor, MapEncoder, PoseEncoder

model = torch.load("model.pt")