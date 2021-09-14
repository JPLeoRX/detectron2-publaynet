# Models overview

To provide you with some options, and to experiment with different models I have trained 6 versions for this dataset. 3 of them are from basic object detection, and 3 contain segmentation masks. You can choose which better suits your task by comparing accuracy scores and inference times of these models.

### Models from [COCO Object Detection](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-object-detection-baselines):

| Name  | Train time (s/iter) | Inference time (s/im) | Folder name | Model zoo config | Trained model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| R50-FPN  | 0.209 | 0.038 | faster_rcnn_R_50_FPN_3x | [Click me!](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml) | [Click me!]() |
| R101-FPN  | 0.286 | 0.051 | faster_rcnn_R_101_FPN_3x | [Click me!](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml) | [Click me!]() |
| X101-FPN  | 0.638 | 0.098 | faster_rcnn_X_101_32x8d_FPN_3x | [Click me!](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml) | [Click me!]() |


### Models from [COCO Instance Segmentation with Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn):

| Name  | Train time (s/iter) | Inference time (s/im) | Folder name | Model zoo config | Trained model |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| R50-FPN  | 0.261 | 0.043 | mask_rcnn_R_50_FPN_3x | [Click me!](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) | [Click me!]() |
| R101-FPN  | 0.340 | 0.056 | mask_rcnn_R_101_FPN_3x | [Click me!](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) | [Click me!]() |
| X101-FPN  | 0.690 | 0.103 | mask_rcnn_X_101_32x8d_FPN_3x | [Click me!](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml) | [Click me!]() |



# Model folder

Each model's directory will contain these files:

| File name  | Description |
| ------------- | ------------- |
| download.txt  | A text file that contains a link from where you can download the trained model |
| train.py  | Training script, that trains the model found in `training_output` sub-folder for a given number of epochs |
| test.py  | Testing script, that runs the model found in `training_output` in inference mode for 6 randomly preselected images (3 from training, 3 from evaluation datasets) and displays all predictions on each image in a pop-up window |
| eval.py  | Testing script, that runs the model found in `training_output` in inference mode for all images found in dataset's `val` folder and `val.json`. Evaluation is performed through Detectron's `COCOEvaluator` |

Generally all models should be available through [here](https://keybase.pub/jpleorx/detectron2-publaynet/), but if not - refer to individual `download.txt` links
