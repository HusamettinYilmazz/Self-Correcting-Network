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
        self.data = self._load_data(self.data_type)

    def _load_data(self, data_type):
        file_path = os.path.join(self.data_path, "ImageSets", "Segmentation", f"{data_type}.txt")
        annot_file_path = os.path.join(self.data_path, "Annotations")

        data = []
        bbox = {}
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip() 
                data.append(line)
                bbox[line] = self._parse_xml(os.path.join(annot_file_path, f"{line}.xml"))

        return data

    def _parse_xml(self, xml_path): ## be sure of its correctness and use it inside _load_data
        """
        1. Build scheme for bbox
        2. Be sure it parsed properly
        3. use it inside _load_data return: bbox with the path of the image
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root.find("filename").text

        boxes = []
        for obj in root.findall("object"):
            cls = obj.find("name").text

            # skip difficult objects if you want
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue

            bndbox = obj.find("bndbox")

            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)

            bbox = BoxInfo(cls, xmin, ymin, xmax, ymax, img_name)

            boxes.append(bbox)

        return boxes

    def __len__(self):
        return len(self.data)
