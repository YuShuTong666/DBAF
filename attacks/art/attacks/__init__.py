"""
Module providing adversarial attacks under a common interface.
"""
from attacks.art.attacks.attack import Attack, EvasionAttack, PoisoningAttack, PoisoningAttackBlackBox, PoisoningAttackWhiteBox
from attacks.art.attacks.attack import PoisoningAttackTransformer, ExtractionAttack, InferenceAttack, AttributeInferenceAttack
from attacks.art.attacks.attack import ReconstructionAttack

from attacks.art.attacks import evasion
from attacks.art.attacks import extraction
from attacks.art.attacks import inference
from attacks.art.attacks import poisoning
