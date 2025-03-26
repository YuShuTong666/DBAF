"""
Module implementing preprocessing defences against adversarial attacks.
"""
from attacks.art.defences.preprocessor.feature_squeezing import FeatureSqueezing
from attacks.art.defences.preprocessor.gaussian_augmentation import GaussianAugmentation
from attacks.art.defences.preprocessor.inverse_gan import DefenseGAN, InverseGAN
from attacks.art.defences.preprocessor.jpeg_compression import JpegCompression
from attacks.art.defences.preprocessor.label_smoothing import LabelSmoothing
from attacks.art.defences.preprocessor.mp3_compression import Mp3Compression
from attacks.art.defences.preprocessor.mp3_compression_pytorch import Mp3CompressionPyTorch
from attacks.art.defences.preprocessor.pixel_defend import PixelDefend
from attacks.art.defences.preprocessor.preprocessor import Preprocessor
from attacks.art.defences.preprocessor.resample import Resample
from attacks.art.defences.preprocessor.spatial_smoothing import SpatialSmoothing
from attacks.art.defences.preprocessor.spatial_smoothing_pytorch import SpatialSmoothingPyTorch
from attacks.art.defences.preprocessor.spatial_smoothing_tensorflow import SpatialSmoothingTensorFlowV2
from attacks.art.defences.preprocessor.thermometer_encoding import ThermometerEncoding
from attacks.art.defences.preprocessor.variance_minimization import TotalVarMin
from attacks.art.defences.preprocessor.video_compression import VideoCompression
from attacks.art.defences.preprocessor.video_compression_pytorch import VideoCompressionPyTorch
