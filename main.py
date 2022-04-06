import json
import os
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import pycocotools.mask as mask

with open("instances_attributes_train2020.json") as f:
    train_data = json.load(f)
    f.close()

with open("instances_attributes_val2020.json") as f:
    val_data = json.load(f)
    f.close()

with open("info_test2020.json") as f:
    test_data = json.load(f)
    f.close()

train_df = pd.DataFrame(train_data['annotations'])
val_df = pd.DataFrame(val_data['annotations'])
print(train_df.head())

id2filename_train = {i['id']: "train/" + i['file_name'] for i in train_data['images']}
id2filename_val = {i['id']: "test/" + i['file_name'] for i in val_data['images']} #le immagini di val sono nella cartella test - controllato
train_df['filename'] = [id2filename_train[row['image_id']] for i, row in train_df.iterrows()]
val_df['filename'] = [id2filename_val[row['image_id']] for i, row in val_df.iterrows()]

def polygonFromMask(maskedArr): # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    #contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours((maskedArr).astype(np.uint8), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    return segmentation

def get_fashionpedia_dicts(df, data):
    """
    Transforms dataframe into dictionary used to train using detectron
    """
    dataset_dicts = []
    for img in data['images']:
        record = {}

        record["file_name"] = str(df[df['image_id'] == img['id']]['filename'].unique()[0])
        record["image_id"] = int(img['id'])
        record["height"] = int(img['height'])
        record["width"] = int(img['width'])
        
        objs = []
        for i, row in df[(df['image_id'] == img['id'])].iterrows():
            #Convert and Check consistency of segmentations 
            if isinstance(row['segmentation'], dict): #if in RLE format convert
                m = mask.decode(row['segmentation'])  #decode binary mask
                segmentation = polygonFromMask(m)     #convert binary mask to polygon in x1, y1, x2, y2, ...
            else:                                     #if already as list do not convert
                segmentation = row['segmentation']
            if segmentation: #drop empty segmentations - NaN
                obj = {
                    "bbox": BoxMode.convert(row['bbox'], from_mode = BoxMode.XYWH_ABS, to_mode = BoxMode.XYXY_ABS), #Convert bbox format
                    "bbox_mode": BoxMode.XYXY_ABS, #bbox format - XYXY for compatibility with pre-trained model
                    "segmentation": segmentation,
                    "category_id": int(row['category_id']),
                    "attributes": row['attribute_ids'],
                    "iscrowd": int(row['iscrowd'])
                }
                objs.append(obj)
        
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Register the custom dataset to detectron2
categories = [i['name'] for i in train_data['categories']]
for d in ["train", "val"]:
    if d == "train":
        used_df = train_df.copy()
        used_data = train_data
    else:
        used_df = val_df.copy()
        used_data = val_data
    DatasetCatalog.register("fashionpedia_" + d, lambda df=used_df, data=used_data: get_fashionpedia_dicts(df, data))
    # DatasetCatalog.register("mat_" + d, lambda df=df_copy: get_materialist_dicts(df))
    MetadataCatalog.get("fashionpedia_" + d).set(thing_classes=categories)
fashionpedia_metadata = MetadataCatalog.get("fashionpedia_train")