# YOLO Modules

## Introduction

A general YOLO pipeline which gives option to choose from different yolo models and backbone for training on CelebA dataset for face detection. This repository also contains work on Speech classification and localization. This repository is maintained by Guoren Zhong, Hamed Omidvar and Nilesh Prasad Pandey as a part of the SRIP summer project, University of California San Diego(UCSD).

## Training on Celeba dataset
# agruments related to train.py

Note: celeba_train_small.txt and celeba_val_small.txt represents the dataset prepared in the format compatible with this yolo repository.
[train.py](https://github.com/david8862/keras-YOLOv3-model-set/blob/master/train.py)
```
# python train.py -h
usage: train.py [-h] [--model_type MODEL_TYPE] [--anchors_path ANCHORS_PATH]
                [--model_image_size MODEL_IMAGE_SIZE]
                [--weights_path WEIGHTS_PATH]
                [--annotation_file ANNOTATION_FILE]
                [--val_annotation_file VAL_ANNOTATION_FILE]
                [--val_split VAL_SPLIT] [--classes_path CLASSES_PATH]
                [--batch_size BATCH_SIZE] [--optimizer OPTIMIZER]
                [--learning_rate LEARNING_RATE] [--decay_type DECAY_TYPE]
                [--transfer_epoch TRANSFER_EPOCH]
                [--freeze_level FREEZE_LEVEL] [--init_epoch INIT_EPOCH]
                [--total_epoch TOTAL_EPOCH] [--multiscale]
                [--rescale_interval RESCALE_INTERVAL] [--model_pruning]
                [--label_smoothing LABEL_SMOOTHING] [--data_shuffle]
                [--gpu_num GPU_NUM] [--eval_online]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--save_eval_checkpoint]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        YOLO model type: yolo3_mobilenet_lite/tiny_yolo3_mobil
                        enet/yolo3_darknet/..., default=yolo3_mobilenet_lite
  --anchors_path ANCHORS_PATH
                        path to anchor definitions,
                        default=configs/yolo3_anchors.txt
  --model_image_size MODEL_IMAGE_SIZE
                        Initial model image input size as <num>x<num>, default
                        416x416
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --annotation_file ANNOTATION_FILE
                        train annotation txt file, default=trainval.txt
  --val_annotation_file VAL_ANNOTATION_FILE
                        val annotation txt file, default=None
  --val_split VAL_SPLIT
                        validation data persentage in dataset if no val
                        dataset provide, default=0.1
  --classes_path CLASSES_PATH
                        path to class definitions,
                        default=configs/voc_classes.txt
  --batch_size BATCH_SIZE
                        Batch size for train, default=16
  --optimizer OPTIMIZER
                        optimizer for training (adam/rmsprop/sgd),
                        default=adam
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.001
  --decay_type DECAY_TYPE
                        Learning rate decay type
                        (None/Cosine/Exponential/Polynomial), default=None
  --transfer_epoch TRANSFER_EPOCH
                        Transfer training (from Imagenet) stage epochs,
                        default=20
  --freeze_level FREEZE_LEVEL
                        Freeze level of the model in transfer training stage.
                        0:NA/1:backbone/2:only open prediction layer
  --init_epoch INIT_EPOCH
                        Initial training epochs for fine tune training,
                        default=0
  --total_epoch TOTAL_EPOCH
                        Total training epochs, default=250
  --multiscale          Whether to use multiscale training
  --rescale_interval RESCALE_INTERVAL
                        Number of iteration(batches) interval to rescale input
                        size, default=10
  --model_pruning       Use model pruning for optimization, only for TF 1.x
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing factor (between 0 and 1) for
                        classification loss, default=0
  --data_shuffle        Whether to shuffle train/val data for cross-validation
  --gpu_num GPU_NUM     Number of GPU to use, default=1
  --eval_online         Whether to do evaluation on validation dataset during
                        training
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Number of iteration(epochs) interval to do evaluation,
                        default=10
  --save_eval_checkpoint
                        Whether to save checkpoint with best evaluation result
```

