from dataclasses import dataclass
from odeon.core.data import InputDataKeys

InputDataFields = {InputDataKeys.INPUT: "image",
                   InputDataKeys.TARGET: "mask",
                   InputDataKeys.PREDS: "preds",
                   InputDataKeys.METADATA: "metadata"}

