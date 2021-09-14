from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from utils import build_config, register_publaynet_datasets

# Register datasets
dataset_train_name, dataset_test_name = register_publaynet_datasets()

# Model parameters
model_zoo_config_name = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
trained_model_output_dir = '/home/leo/tekleo/detectron2-publaynet/faster_rcnn_R_50_FPN_3x/training_output'
prediction_score_threshold = 0.7
base_lr = 0.0025
max_iter = 1000
batch_size = 64

# Detectron config
cfg = build_config(model_zoo_config_name, dataset_train_name, dataset_test_name, trained_model_output_dir, prediction_score_threshold, base_lr, max_iter, batch_size)

# Detectron trainer with evaluation mode
model = build_model(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.test(cfg, model, [COCOEvaluator(dataset_test_name)])
