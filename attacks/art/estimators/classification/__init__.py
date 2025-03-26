"""
Classifier API for applying all attacks. Use the :class:`.Classifier` wrapper to be able to apply an attack to a
preexisting model.
"""
from attacks.art.estimators.classification.classifier import (
    ClassifierMixin,
    ClassGradientsMixin,
)

from attacks.art.estimators.classification.blackbox import BlackBoxClassifier, BlackBoxClassifierNeuralNetwork
from attacks.art.estimators.classification.catboost import CatBoostARTClassifier
from attacks.art.estimators.classification.deep_partition_ensemble import DeepPartitionEnsemble
from attacks.art.estimators.classification.detector_classifier import DetectorClassifier
from attacks.art.estimators.classification.ensemble import EnsembleClassifier
from attacks.art.estimators.classification.GPy import GPyGaussianProcessClassifier
from attacks.art.estimators.classification.keras import KerasClassifier
from attacks.art.estimators.classification.lightgbm import LightGBMClassifier
from attacks.art.estimators.classification.mxnet import MXClassifier
from attacks.art.estimators.classification.pytorch import PyTorchClassifier
from attacks.art.estimators.classification.query_efficient_bb import QueryEfficientGradientEstimationClassifier
from attacks.art.estimators.classification.scikitlearn import SklearnClassifier
from attacks.art.estimators.classification.tensorflow import (
    TFClassifier,
    TensorFlowClassifier,
    TensorFlowV2Classifier,
)
from attacks.art.estimators.classification.xgboost import XGBoostClassifier
