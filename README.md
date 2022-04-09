# Fashionpedia Image Segmentation

`config.yaml` contains the config parameters of each component necessary to fine-tune the RegNetY pre-trained model from Detectron2 on the Fashionpedia dataset

```
src
    ├── data.py
    ├── main.py
    └── solver.py
```

`data.py` contains the following functions:
  - `get_data()` loads annotations form already downloaded json files in the *data* folder
  - `polygonFromMask(maskedArr)` converts a binary mask to a list of polygons - used to convert RLE segmentations in list of 
      polygons and it is necessary to convert first the RLE segmentation using the pycocotools.mask.decode function
  - `get_fashionpedia_dicts(df, data)` creates a dictionary containing all images and its annotations in a compatible format 
      in order to fine-tune the model
  - `register_datasets(train_data, train_df, val_data, val_df)` register train and val dataset in the detectron2 DatasetCatalog and MetadataCatalog
  - `override_parameters(cfg, args, n)` the main script gets batch_size, num_workers, number of epochs as command line arguments and several model parameters needs to be
      adjusted according to these, so this functions automatically calculates them and overrides default ones.
      
`main.py` contains the script to be run. It accepts as command line arguments batch_size (number of images per batch), number of epochs and num_workers (affects
pre-training, in particular each worker prepares a batch to be trained next - i.e. for augmentations)

`solver.py` contains the functions `do_train` and `do_test` which respectively define the training routine and test routine
