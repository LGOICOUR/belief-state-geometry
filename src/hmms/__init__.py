"""Data-generating processes (HMMs) in computational-mechanics labeled-transition form.

The base :class:`Process` derives all belief-state machinery generically from the
labeled transition tensor, so concrete processes only declare their tensor.
"""

from .base import Process
from .mess3 import Mess3, mess3_tensor, PAPER_X, PAPER_ALPHA
from .rrxor import RRXOR, RRXOR_TENSOR
from .mixture import MixtureProcess, mixture_tensor, GEN_A, GEN_B  # Phase 2

__all__ = [
    "Process",
    "Mess3",
    "mess3_tensor",
    "PAPER_X",
    "PAPER_ALPHA",
    "RRXOR",
    "RRXOR_TENSOR",
    "MixtureProcess",
    "mixture_tensor",
    "GEN_A",
    "GEN_B",
]
