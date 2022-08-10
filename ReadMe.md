# Vision Cognitive Engineering Assignment

This repository includes the final assignment of Vision Cognitive Engineering. Semantic Segmentation is selected from three issues.
In this issue, students are required to complete a model mentioned in this courses. The model should be accomplished using PyTorch and then trained and tested on dataset Weizmann Horse. MIoU  and BIoU will be used to measure the performance of the model, the front should reach 0.9 possibly.

## Prepare Data
[Weizmann-Horse](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata) in Kaggle is available for our code. The dataset provides horses images and  mask images.
The path structure used in our code looks like this:
dataset
├──── horse
│    ├──── horse001.png
│    ├──── horse002.png
│    ├──── ...
│    └──── horse327.png
├──── mask
│    ├──── horse001.png
│    ├──── horse002.png
│    ├──── ...
│    └──── horse327.png

## Train and Test
The file model.py defines the models used in EfficientDet, and utils.py is the complementation for it. The file main.py builds a BiFPN model and training it on the dataset loaded before, then test the model with mIoU.
In fact, you can only execute semanticsegmentation.ipynb which is available in Kaggle to train and test the model. If you want to only test or change the parameters, please modify the code according to the annotation.
In addition, a trained model is available in the [release](https://github.com/LebmontG/Semantic-Segmentation/releases/tag/BiFPN) of this repository. You can download it and use the model without training.

## Citation
BiFPN:
Tan, M. , R. Pang , and Q. V. Le . ”EﬀicientDet: Scalable and Eﬀicient Object Detection.”2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) IEEE,2020.

Boundary IoU:
Cheng, B. , et al. ”Boundary IoU: Improving Object-Centric Image Segmentation Evaluation.” 2021.

## Permission and Disclaimer
This code can be used/distributed for any purpose, though it is probably only used for scoring for course Vision Cognitive Engineering.