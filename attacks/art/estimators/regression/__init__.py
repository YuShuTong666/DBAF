"""
This module implements all regressors in ART.
"""
from attacks.art.estimators.regression.regressor import RegressorMixin

from attacks.art.estimators.regression.scikitlearn import ScikitlearnRegressor

from attacks.art.estimators.regression.keras import KerasRegressor

from attacks.art.estimators.regression.pytorch import PyTorchRegressor
