import os
from typing import List
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy
from PIL.Image import Image
from PIL import Image as image_main
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances


def open_image_pil(image_path: str) -> Image:
    return image_main.open(image_path)


def convert_pil_to_cv(pil_image: Image):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)


def debug_image_cv(cv_image):
    cv2.namedWindow('Debug Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Debug Image', cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def register_publaynet_datasets() -> (str, str):
    dataset_train_name = 'publaynet_dataset_train'
    dataset_test_name = 'publaynet_dataset_test'
    class_labels = ['text', 'title', 'list', 'table', 'figure']
    register_coco_instances(
        dataset_train_name,
        {},
        "/home/leo/datasets/publaynet/train.json",
        "/home/leo/datasets/publaynet/train"
    )
    register_coco_instances(
        dataset_test_name,
        {},
        "/home/leo/datasets/publaynet/val.json",
        "/home/leo/datasets/publaynet/val"
    )

    # Make sure the datasets got registered
    metadata_train = MetadataCatalog.get(dataset_train_name)
    metadata_test = MetadataCatalog.get(dataset_test_name)
    print(metadata_train)
    print(metadata_test)

    # Set labels
    MetadataCatalog.get(dataset_train_name).thing_classes = class_labels
    MetadataCatalog.get(dataset_test_name).thing_classes = class_labels

    return dataset_train_name, dataset_test_name


def build_config(
        model_zoo_config_name: str,
        dataset_train_name: str, dataset_test_name: str,
        trained_model_output_dir: str,
        prediction_score_threshold: float,
        base_lr: float, max_iter: int, batch_size: int
) -> CfgNode:
    trained_model_weights_path = trained_model_output_dir + "/model_final.pth"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
    cfg.DATASETS.TRAIN = (dataset_train_name,)
    cfg.DATASETS.TEST = (dataset_test_name,)
    cfg.OUTPUT_DIR = trained_model_output_dir
    cfg.DATALOADER.NUM_WORKERS = 8
    if os.path.exists(trained_model_weights_path):
        cfg.MODEL.WEIGHTS = trained_model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100
    return cfg


def visualize_dataset_dict(dataset_name: str, dataset_dict):
    image_pil = open_image_pil(dataset_dict["file_name"])
    image_cv = convert_pil_to_cv(image_pil)
    visualizer = Visualizer(image_cv[:, :, ::-1], MetadataCatalog.get(dataset_name), scale=1.2)
    out = visualizer.draw_dataset_dict(dataset_dict)
    output_image_cv = out.get_image()[:, :, ::-1]
    debug_image_cv(output_image_cv)


def visualize_outputs(cfg, image_cv, outputs):
    v = Visualizer(image_cv[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_image_cv = out.get_image()[:, :, ::-1]
    debug_image_cv(output_image_cv)


def visual_test(cfg: CfgNode, predictor: DefaultPredictor):
    image_paths = [
        "/home/leo/datasets/publaynet/train/PMC1500815_00002.jpg",
        "/home/leo/datasets/publaynet/train/PMC3162874_00002.jpg",
        "/home/leo/datasets/publaynet/train/PMC4203354_00000.jpg",
        "/home/leo/datasets/publaynet/val/PMC1247188_00003.jpg",
        "/home/leo/datasets/publaynet/val/PMC2829689_00004.jpg",
        "/home/leo/datasets/publaynet/val/PMC4520132_00000.jpg",
    ]

    for image_path in image_paths:
        print('Testing on ' + image_path)
        image_pil = open_image_pil(image_path)
        image_cv = convert_pil_to_cv(image_pil)
        outputs = predictor(image_cv)
        visualize_outputs(cfg, image_cv, outputs)
