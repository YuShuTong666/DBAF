"""
Module providing attribute inference attacks.
"""
from attacks.art.attacks.inference.attribute_inference.black_box import AttributeInferenceBlackBox
from attacks.art.attacks.inference.attribute_inference.baseline import AttributeInferenceBaseline
from attacks.art.attacks.inference.attribute_inference.true_label_baseline import AttributeInferenceBaselineTrueLabel
from attacks.art.attacks.inference.attribute_inference.white_box_decision_tree import AttributeInferenceWhiteBoxDecisionTree
from attacks.art.attacks.inference.attribute_inference.white_box_lifestyle_decision_tree import (
    AttributeInferenceWhiteBoxLifestyleDecisionTree,
)
from attacks.art.attacks.inference.attribute_inference.meminf_based import AttributeInferenceMembership
