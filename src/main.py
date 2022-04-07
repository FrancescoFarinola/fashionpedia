import pandas as pd
from detectron2.config import LazyConfig as L
from solver import do_train
from data import get_data, register_datasets, override_parameters
import argparse
import torch
import gc


def main(args):
    #Load json files
    train_data, val_data, test_data = get_data()
    print('Loaded data')

    #Create dataframes
    train_df = pd.DataFrame(train_data['annotations'])
    val_df = pd.DataFrame(val_data['annotations'])
    print(train_df.head())

    #Get paths of all image files
    print("Adding file paths to dataframe...")
    id2filename_train = {i['id']: "./train/" + i['file_name'] for i in train_data['images']}
    id2filename_val = {i['id']: "./test/" + i['file_name'] for i in val_data['images']} 
    train_df['filename'] = [id2filename_train[row['image_id']] for i, row in train_df.iterrows()]
    val_df['filename'] = [id2filename_val[row['image_id']] for i, row in val_df.iterrows()]
    print("Added!")

    #Register custom train and test datasets is Detectron2 DatasetCatalog and MetadataCatalog
    categories, fp_metadata = register_datasets(train_data, train_df, val_data, val_df)

    cfg = L.load("./config.yaml") #Lazy initialization of the model parameters - necessary for new baseline models
    a = {"num_machines":1, "num_gpus":1} #Dictionary with additional parameters to initialize the detectron2 environment

    #Ovverride parameters passed as args
    len_df = train_df['image_id'].unique().size #Number of unique images necessary to set max_iter, log
    cfg = override_parameters(cfg, args, len_df)

    #Clear cuda cache and use garbage collector before starting training 
    torch.cuda.empty_cache()
    gc.collect()
    do_train(cfg, a)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Depending on batch size and epochs several parameters needs to be adjusted but this is done automatically by override_parameters function
    # Changes in train.max_iter, train.log_period, train.eval_period and checkpointer : log is set at each 1% of each epoch, while eval and checkpoint for every epoch
    # Also affects the LR schedule : num_updates requires to be equal to max_iter, also need to adjust milestones (when to lower lr) to 60 and 90% of total
    # training iterations. Also, warmup_length.
    # LR is decreased at each milestone: starting from 0.001 then 0.0001 and finally 0.00001
    # Need to adjust the starting LR : start from 0.01 or 0.001? When milestones? How many epochs?
    parser.add_argument("-b", "--batch_size", action="store", default=2, type=int)
    parser.add_argument("-e", "--epochs", action="store", default=1)
    parser.add_argument("-w", "--workers", action="store", default=2, type=int) #useful just for preparing batches - doesnt affect other parameters
    args = parser.parse_args()
    main(args)