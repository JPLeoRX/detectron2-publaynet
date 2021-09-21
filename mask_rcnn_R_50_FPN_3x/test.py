from detectron2.engine import DefaultPredictor
from utils import build_config, register_publaynet_datasets, visual_test

# Register datasets
dataset_train_name, dataset_test_name = register_publaynet_datasets()

# Model parameters
model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
trained_model_output_dir = '/home/leo/tekleo/detectron2-publaynet/mask_rcnn_R_50_FPN_3x/training_output'
prediction_score_threshold = 0.7
base_lr = 0
max_iter = 0
batch_size = 0

# Detectron config
cfg = build_config(model_zoo_config_name, dataset_train_name, dataset_test_name, trained_model_output_dir, prediction_score_threshold, base_lr, max_iter, batch_size)

# Detectron predictor
predictor = DefaultPredictor(cfg)
visual_test(cfg, predictor)
