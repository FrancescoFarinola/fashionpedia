import pandas as pd
from detectron2.config import LazyConfig as L
from detectron2.engine import default_setup
from engine import do_train
from data import get_data, register_datasets

train_data, val_data, test_data = get_data()
print('Loaded data')

train_df = pd.DataFrame(train_data['annotations'])
val_df = pd.DataFrame(val_data['annotations'])
print(train_df.head())

print("Adding file paths to dataframe...")
id2filename_train = {i['id']: "train/" + i['file_name'] for i in train_data['images']}
id2filename_val = {i['id']: "test/" + i['file_name'] for i in val_data['images']} #le immagini di val sono nella cartella test - controllato
train_df['filename'] = [id2filename_train[row['image_id']] for i, row in train_df.iterrows()]
val_df['filename'] = [id2filename_val[row['image_id']] for i, row in val_df.iterrows()]
print("Added!")

fp_metadata, classes = register_datasets(train_data, train_df, val_data, val_df)


cfg = L.load("config.yaml")
a = {"num_machines":1, "num_gpus":1}
cfg.dataloader.train.total_batch_size = 1
cfg.dataloder.train.num_workers = 2
cfg.train.max_iter = 200
cfg.train.log_period = 20
default_setup(cfg, args = a)


import torch
torch.cuda.empty_cache()
import gc
gc.collect()

do_train(cfg)