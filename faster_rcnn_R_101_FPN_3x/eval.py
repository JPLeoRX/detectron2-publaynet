from detectron2.data import DatasetCatalog, DatasetMapper
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from utils import build_config, register_publaynet_datasets, visualize_dataset_dict, visualize_outputs

# Register datasets
dataset_train_name, dataset_test_name = register_publaynet_datasets()

# Model parameters
model_zoo_config_name = 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
trained_model_output_dir = '/home/leo/tekleo/detectron2-publaynet/faster_rcnn_R_101_FPN_3x/training_output'
prediction_score_threshold = 0.7
base_lr = 0.001
max_iter = 500
batch_size = 128

# Detectron config
cfg = build_config(model_zoo_config_name, dataset_train_name, dataset_test_name, trained_model_output_dir, prediction_score_threshold, base_lr, max_iter, batch_size)

# Detectron predictor
predictor = DefaultPredictor(cfg)

# Load test dataset, and evaluate over it
dataset = DatasetCatalog.get(dataset_test_name)
dataset_mapper = DatasetMapper(cfg, is_train=False)
data_loader = build_detection_test_loader(dataset=dataset, mapper=dataset_mapper)
inference_on_dataset(
    model=predictor.model,
    data_loader=data_loader,
    evaluator=COCOEvaluator(dataset_test_name),
)
