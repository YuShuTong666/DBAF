"""
This module contains the Estimator API.
"""
from attacks.art.estimators.estimator import (
    BaseEstimator,
    LossGradientsMixin,
    NeuralNetworkMixin,
    DecisionTreeMixin,
)

from attacks.art.estimators.keras import KerasEstimator
from attacks.art.estimators.mxnet import MXEstimator
from attacks.art.estimators.pytorch import PyTorchEstimator
from attacks.art.estimators.scikitlearn import ScikitlearnEstimator
#from attacks.art.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator

from attacks.art.estimators import certification
from attacks.art.estimators import classification
from attacks.art.estimators import encoding
from attacks.art.estimators import generation
from attacks.art.estimators import object_detection
from attacks.art.estimators import poison_mitigation
from attacks.art.estimators import regression
from attacks.art.estimators import speech_recognition
