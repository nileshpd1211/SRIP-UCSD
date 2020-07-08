# YOLO Modules

## Introduction

A general YOLO pipeline which gives option to choose from different yolo models and backbone for training on CelebA dataset for face detection and librispeech dataset for voice localization. This repository also contains work on Speech classification and localization. This repository is maintained by Guoren Zhong, Hamed Omidvar and Nilesh Prasad Pandey as a part of the SRIP summer project, University of California San Diego(UCSD) under the guidance of Prof. Massimo Franceschetti.

## Training on Librispeech Dataset [./SpeechDetection]


## Training on Celeba dataset [./FaceDetection]
This FaceDetection folder contains three sub-folders:
1. Yolo_support: contains all the files for setting up yolo networks, configurations and prepared celebA data
2. YoloIpython: Contains Ipython notebooks for training yolo networks for face detection
3. pythonscripts: Contains python scripts for training yolo networks for face detection

Note: celeba_train_small.txt and celeba_val_small.txt in the Yolo_support represents the dataset prepared in the format compatible with this yolo repository.

