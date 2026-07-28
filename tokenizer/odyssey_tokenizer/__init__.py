"""OdysseyTokenizer — reusable byte-level BPE library.

Public interface
----------------
>>> from odyssey_tokenizer import OdysseyTokenizer
>>> tokenizer = OdysseyTokenizer.load("assets/tokenizer/bpe/odyssey.model")
>>> ids = tokenizer.encode("Build authentication API")
>>> text = tokenizer.decode(ids)
"""

from odyssey_tokenizer.config import BPEConfig, load_bpe_config
from odyssey_tokenizer.tokenizer import InspectionResult, OdysseyTokenizer
from odyssey_tokenizer.trainer import BPETrainer, TrainResult, train_bpe

__all__ = [
    "BPEConfig",
    "BPETrainer",
    "InspectionResult",
    "OdysseyTokenizer",
    "TrainResult",
    "load_bpe_config",
    "train_bpe",
]

__version__ = "0.2.0"
