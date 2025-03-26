"""
Module providing evasion attacks under a common interface.
"""
from attacks.art.attacks.evasion.adversarial_patch.adversarial_patch import AdversarialPatch
from attacks.art.attacks.evasion.adversarial_patch.adversarial_patch_numpy import AdversarialPatchNumpy
from attacks.art.attacks.evasion.adversarial_patch.adversarial_patch_tensorflow import AdversarialPatchTensorFlowV2
from attacks.art.attacks.evasion.adversarial_patch.adversarial_patch_pytorch import AdversarialPatchPyTorch
from attacks.art.attacks.evasion.adversarial_texture.adversarial_texture_pytorch import AdversarialTexturePyTorch
from attacks.art.attacks.evasion.adversarial_asr import CarliniWagnerASR
from attacks.art.attacks.evasion.auto_attack import AutoAttack
from attacks.art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from attacks.art.attacks.evasion.brendel_bethge import BrendelBethgeAttack
from attacks.art.attacks.evasion.boundary import BoundaryAttack
from attacks.art.attacks.evasion.carlini import CarliniL2Method, CarliniLInfMethod, CarliniL0Method
from attacks.art.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from attacks.art.attacks.evasion.deepfool import DeepFool
from attacks.art.attacks.evasion.dpatch import DPatch
from attacks.art.attacks.evasion.dpatch_robust import RobustDPatch
from attacks.art.attacks.evasion.elastic_net import ElasticNet
from attacks.art.attacks.evasion.fast_gradient import FastGradientMethod
from attacks.art.attacks.evasion.frame_saliency import FrameSaliencyAttack
from attacks.art.attacks.evasion.feature_adversaries.feature_adversaries_numpy import FeatureAdversariesNumpy
from attacks.art.attacks.evasion.feature_adversaries.feature_adversaries_pytorch import FeatureAdversariesPyTorch
from attacks.art.attacks.evasion.feature_adversaries.feature_adversaries_tensorflow import FeatureAdversariesTensorFlowV2
from attacks.art.attacks.evasion.geometric_decision_based_attack import GeoDA
from attacks.art.attacks.evasion.hclu import HighConfidenceLowUncertainty
from attacks.art.attacks.evasion.hop_skip_jump import HopSkipJump
from attacks.art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
from attacks.art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
from attacks.art.attacks.evasion.iterative_method import BasicIterativeMethod
from attacks.art.attacks.evasion.laser_attack.laser_attack import LaserAttack
from attacks.art.attacks.evasion.lowprofool import LowProFool
from attacks.art.attacks.evasion.momentum_iterative_method import MomentumIterativeMethod
from attacks.art.attacks.evasion.newtonfool import NewtonFool
from attacks.art.attacks.evasion.pe_malware_attack import MalwareGDTensorFlow
from attacks.art.attacks.evasion.pixel_threshold import PixelAttack
from attacks.art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from attacks.art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import (
    ProjectedGradientDescentNumpy,
)
from attacks.art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import (
    ProjectedGradientDescentPyTorch,
)
from attacks.art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import (
    ProjectedGradientDescentTensorFlowV2,
)
from attacks.art.attacks.evasion.over_the_air_flickering.over_the_air_flickering_pytorch import OverTheAirFlickeringPyTorch
from attacks.art.attacks.evasion.saliency_map import SaliencyMapMethod
from attacks.art.attacks.evasion.shadow_attack import ShadowAttack
from attacks.art.attacks.evasion.shapeshifter import ShapeShifter
from attacks.art.attacks.evasion.simba import SimBA
from attacks.art.attacks.evasion.spatial_transformation import SpatialTransformation
from attacks.art.attacks.evasion.square_attack import SquareAttack
from attacks.art.attacks.evasion.pixel_threshold import ThresholdAttack
from attacks.art.attacks.evasion.universal_perturbation import UniversalPerturbation
from attacks.art.attacks.evasion.targeted_universal_perturbation import TargetedUniversalPerturbation
from attacks.art.attacks.evasion.virtual_adversarial import VirtualAdversarialMethod
from attacks.art.attacks.evasion.wasserstein import Wasserstein
from attacks.art.attacks.evasion.zoo import ZooAttack
from attacks.art.attacks.evasion.sign_opt import SignOPTAttack
