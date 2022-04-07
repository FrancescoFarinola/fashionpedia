import json
import cv2
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
import pycocotools.mask as mask
import numpy as np

def get_data():
    # Load annotations from json files
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
    # Utility function to convert a binary mask to a list of polygons
    # To get the binary mask we need to pass the segmentation in RLE to the decode function of pycocotools.mask
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
    The dictionary needs to be in the following format:
        - image_id: unique id of the image : int
        - file_name: string containing the path to the image_id : string
        - height: height in pixel of the image : int
        - width: width in pixel of the image : int
        - annotations: list of annotations including one for each of it:
            - bbox: bounding box of the segmentation in format [x_min,y_min,x_max,y_max] : list of floats
                    in our dataset the default format is [x, y, width, height] so we convert it
            - bbox_mode: class indicating the format of the bounding box : class
                    for compatibility reasons we are forced to use XYXY_ABS format
            - segmentation: list of polygons for current annotation : list of lists in format [x1,y1, x2,y2 ... xn,yn]
                    some segmentations are in RLE format so we need to convert them and check its consistency
            - category_id: id of the category : int
            - attributes: list of attribute ids : list of ints
            - iscrowd: 0 or 1 according to COCO - already given in the dataset : bool/int
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
        #Register
        DatasetCatalog.register("fashionpedia_" + d, lambda df=used_df, data=used_data: get_fashionpedia_dicts(df, data))
        MetadataCatalog.get("fashionpedia_" + d).set(thing_classes=categories)
    fashionpedia_metadata = MetadataCatalog.get("fashionpedia_train")
    print("Registered!")
    return categories, fashionpedia_metadata


def override_parameters(cfg, args, n):
    # Override parameters affected by batch_size, epochs and num_workers and initialize evaluator
    cfg.dataloader.train.total_batch_size = args.batch_size #Number of images per batch
    #Number of workers to pre-load batches - this has nothing to do with the number of iterations
    cfg.dataloader.train.num_workers = cfg.dataloader.test.num_workers = args.workers 
    #Set the number of iterations according to epochs and number of batches and consequently 
    #change the log, evaluation and checkpoint period according to # of iterations (save every epoch)
    one_epoch = int(n / args.batch_size) + 1
    cfg.train.max_iter = one_epoch * int(args.epochs)
    cfg.train.log_period = int(one_epoch /100) #log ech 1% of each epoch - 100 logs per epoch
    cfg.train.eval_period = one_epoch #evaluate after each epoch
    cfg.checkpointer = {'max_to_keep': 100, 'period': one_epoch} #save checkpoint at each epoch

    #LR multiplier
    #Change milestones and number of updates for step learning rate schedule
    cfg.lr_multiplier.scheduler.num_updates = one_epoch * int(args.epochs) #shiould be equal to number of iterations
    # lower lr at 60% and 90% of training iterations
    cfg.lr_multiplier.scheduler.milestones = [int((one_epoch * int(args.epochs))/10*6) , int((one_epoch * int(args.epochs))/10*9)]
    cfg.lr_multiplier.warmup_length = 200/ (one_epoch * int(args.epochs))
    #Reinitialize evaluator to make it correctly work
    cfg.dataloader.evaluator = COCOEvaluator("fashionpedia_val", output_dir="./output") 
    return cfg