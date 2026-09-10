"""Extensions to QSPRpred needed by these experiments.

Everything here exists so the experiments can run against the dev version
of QSPRpred on github.

  randomnets      the Randomnets model, wrapper around the original RandomNets
  neural_network  HiddenSizesFullyConnected: DNN with explicit hidden layer widths
  descriptors     PCMDataFrameDescriptorSet: descriptor join for PCM row structure
"""

from qsprpred_ext.descriptors import PCMDataFrameDescriptorSet
from qsprpred_ext.neural_network import HiddenSizesFullyConnected
from qsprpred_ext.randomnets import Randomnets

__all__ = [
    "PCMDataFrameDescriptorSet",
    "HiddenSizesFullyConnected",
    "Randomnets",
]
