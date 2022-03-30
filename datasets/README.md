# Base datasets

For a few datasets that FsDet natively supports,
the datasets are assumed to exist in a directory called
"datasets/", under the directory where you launch the program.
They need to have the following directory structure:


## Expected dataset structure for COCO:
```
coco/
  annotations/
    instances_{train,val}2014.json
  {train,val}2014/
    # image files that are mentioned in the corresponding json
```

## Expected dataset structure for LVIS:
```
coco/
  {train,val}2017/
lvis/
  lvis_v0.5_{train,val}.json
  lvis_v0.5_train_{freq,common,rare}.json
```

LVIS uses the same images and annotation format as COCO. You can use [split_lvis_annotation.py](split_lvis_annotation.py) to split `lvis_v0.5_train.json` into `lvis_v0.5_train_{freq,common,rare}.json`.

Install lvis-api by:
```
pip install git+https://github.com/lvis-dataset/lvis-api.git
```

# Few-shot datasets

For each dataset, we additionally create few-shot versions by sampling shots for each novel category. For better comparisons, we sample multiple groups of training shots in addition to the ones provided in previous works. We include the sampling scripts we used for better reproducibility and extensibility. The few-shot dataset files can be found [here](http://dl.yf.io/fs-det/datasets/). They should have the following directory structure:



## COCO:
```
cocosplit/
  datasplit/
    trainvalno5k.json
    5k.json
  full_box_{1,2,3,5,10,30}shot_{category}_trainval.json
  seed{1-9}/
    # shots
```

All but 5k images from the train/val sets are used for training, while 5k images from the val set are used as the test set. `trainvalno5k.json` denotes the training set and `5k.json` is the test set.

The sampling procedure is the same as for Pascal VOC, except we sample exactly _K_ instances for each category. For COCO, we use 10 groups.

See [prepare_coco_few_shot.py](prepare_coco_few_shot.py) for generating the seeds yourself.

Dataset names for config files:
```
coco_trainval_{base,all}                        # Train/val datasets with base categories or all
                                                  categories.
coco_trainval_all_{1,2,3,5,10,30}shot           # Balanced subsets containing 1, 2, 3, 5, 10, or 30
                                                  shots for each category.
coco_trainval_novel_{1,2,3,5,10,30}shot         # Same as previous datasets, but only contains data
                                                  of novel categories.
coco_test_{base,novel,all}                      # Test datasets with base categories, novel categories,
                                                  or all categories.
```

## LVIS:
```
lvissplit/
  lvis_shots.json
```

We treat the frequent and common categories as the base categories and the rare categories as the novel categories.

We sample up to 10 instances for each category to build a balanced subset for the few-shot fine-tuning stage. We include all shots in a single COCO-style annotation file.

See [prepare_lvis_few_shot.py](prepare_lvis_few_shot.py) for generating the seeds yourself.

Dataset names for config files:
```
lvis_v0.5_train_{freq,common}                   # Train datasets with freq categories or common
                                                  categories. These are used as the base datasets.
lvis_v0.5_train_rare_novel                      # Train datasets with rare categories.
lvis_v0.5_train_shots                           # Balanced subset containing up to 10 shots for
                                                  each category.
lvis_v0.5_val                                   # Validation set with all categories.
lvis_v0.5_val_novel                             # Validation set with only novel categories.
```
