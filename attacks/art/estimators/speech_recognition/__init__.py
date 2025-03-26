"""
Module containing estimators for speech recognition.
"""
from attacks.art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin

from attacks.art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from attacks.art.estimators.speech_recognition.pytorch_espresso import PyTorchEspresso
from attacks.art.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
