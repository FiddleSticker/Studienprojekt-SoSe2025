import os
import torch

print(torch.cuda.is_available())

SRC_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.dirname(SRC_DIR)
