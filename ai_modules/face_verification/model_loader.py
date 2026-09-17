from keras_facenet import FaceNet
from mtcnn import MTCNN
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
import numpy as np

_facenet_model = None
_resnet_model = None
_mtcnn_detector = None

def load_facenet():
    global _facenet_model
    if _facenet_model is None:
        _facenet_model = FaceNet()
    return _facenet_model

def load_resnet():
    global _resnet_model
    if _resnet_model is None:
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        _resnet_model = Model(inputs=base_model.input, outputs=base_model.output)
    return _resnet_model

def load_mtcnn():
    global _mtcnn_detector
    if _mtcnn_detector is None:
        _mtcnn_detector = MTCNN()
    return _mtcnn_detector