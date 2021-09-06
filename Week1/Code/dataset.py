"""
Creates a Pytorch dataset to load the Pascal VOC & MS COCO datasets
"""

from posixpath import join
import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir, label_dir,
        anchors,
        image_size = 416,
        S = [13,26,52],
        C = 20,
        transform = None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0],anchors[1],anchors[2]) # for all 3 scales
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index,1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndim=2),4,axis = 1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index,0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform
            image = augmentations["imahe"]
            bboxes = augmentations["bboxes"]
            
        # targets = 
        