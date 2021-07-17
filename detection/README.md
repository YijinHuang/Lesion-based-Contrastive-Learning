# Lesion Detection Network

A wonderful object detection toolbox [MMDetection](https://github.com/open-mmlab/mmdetection) are used for lesion detection. A trained model and predicted results can be downloaded [here](https://github.com/YijinHuang/Lesion-based-Contrastive-Learning/releases/tag/v1.0). Note that the model has a relatively poor generalization ability and cannot precisely predict lesions of fundus images from EyePACS because of the limited training samples of IDRiD.



## Usage

### Installation

Please refer to the instruction of MMDetection for installation.

### Training

1. Download IDRiD dataset [[Link](https://idrid.grand-challenge.org/ )].
2. Use `scripts/mask2bbox.py` to convert the lesion mask of images to bounding boxes. The output is a csv file.
3. Use `scripts/csv2coco.py` to convert the csv file to coco annotation format. The requirements for the format of new datasets can be found in [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/2_new_data_model.md).
4. Replace the configuration files in MMDetection with the files in `configs`. Please remember to update the path to dataset.

5. Follow the instruction of MMDetection to training the model.

### Dataset for Representation Learning

1. Download EyePACS dataset [[Link](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)]. Then delete line 77 in `../tools/crop.py` and run it to remove the black border of images of EyePACS. Line 77 in `../tools/crop.py` is for resizing and we do not resize images for lesion detection.

2. Use `scripts/detect.py` to generate lesion predictions of EyePACS.

3. Use `scripts/build_dataset.py` to select lesions with high confidence threshold. The output is a pickle file for training the lesion-based contrastive learning models.

