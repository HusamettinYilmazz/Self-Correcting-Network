import os 
import sys

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

import xml.etree.ElementTree as ET

from utils.boxinfo import BoxInfo

class FullySuperVOCDataset(Dataset):
    def __init__(self, data_path, data_type="train", transform=None):
        super().__init__()
        
        self.data_path = data_path
        self.data_type = data_type
        self.transform = transform
