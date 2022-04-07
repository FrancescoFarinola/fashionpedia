import json
import cv2
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import pycocotools.mask as mask
import numpy as np

def get_data():
    with open("instances_attributes_train2020.json") as f:
        train_data = json.load(f)
        f.close()

    with open("instances_attributes_val2020.json") as f:
        val_data = json.load(f)
        f.close()

    with open("info_test2020.json") as f:
        test_data = json.load(f)
        f.close()
    return train_data, val_data, test_data


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


def register_datasets(train_data, train_df, val_data, val_df):
    # Register the custom dataset to detectron2
    print("Registering dataset and metadata on Catalog")
    categories = [i['name'] for i in train_data['categories']]
    for d in ["train", "val"]:
        if d == "train":
            used_df = train_df.copy()
            used_data = train_data
        else:
            used_df = val_df.copy()
            used_data = val_data
        DatasetCatalog.register("fashionpedia_" + d, lambda df=used_df, data=used_data: get_fashionpedia_dicts(df, data))
        MetadataCatalog.get("fashionpedia_" + d).set(thing_classes=categories)
    fashionpedia_metadata = MetadataCatalog.get("fashionpedia_train")
    print("Registered!")
    return fashionpedia_metadata, categories



