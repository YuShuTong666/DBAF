"""
Module containing estimators for object detection.
"""
from attacks.art.estimators.object_detection.object_detector import ObjectDetectorMixin

from attacks.art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
from attacks.art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from attacks.art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from attacks.art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
