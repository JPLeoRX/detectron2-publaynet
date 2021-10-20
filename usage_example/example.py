import cv2
import numpy
from PIL import Image as image_main
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# Model, threshold score, class labels, and example image - be sure to replace with your own
image_path = 'PMC1500815_00002.jpg'
model_path = '/home/leo/tekleo/detectron2-publaynet/mask_rcnn_R_50_FPN_3x/training_output/model_final.pth'
model_zoo_config_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
prediction_score_threshold = 0.7
class_labels = ['text', 'title', 'list', 'table', 'figure']

# Detectron config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model_zoo_config_name))
cfg.MODEL.WEIGHTS = model_path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_score_threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

# Detectron predictor
predictor = DefaultPredictor(cfg)

# Open image, and run predictor
image_pil = image_main.open(image_path)
image_cv = cv2.cvtColor(numpy.array(image_pil), cv2.COLOR_RGB2BGR)
outputs = predictor(image_cv)

# Debug outputs
instances = outputs["instances"].to("cpu")
pred_boxes = instances.pred_boxes
scores = instances.scores
pred_classes = instances.pred_classes
for i in range(0, len(pred_boxes)):
    box = pred_boxes[i].tensor.numpy()[0]
    score = round(float(scores[i].numpy()), 4)
    label_key = int(pred_classes[i].numpy())
    label = class_labels[label_key]
    x = int(box[0])
    y = int(box[1])
    w = int(box[2] - box[0])
    h = int(box[3] - box[1])

    print('Detected object of label=' + str(label) + ' with score=' + str(score) + ' and in box={x=' + str(x) + ', y=' + str(y) + ', w=' + str(w) + ', h=' + str(h) + '}')
