from pydantic.dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AdvancedProfilerConf:
    _target_: str = "pytorch_lightning.profiler.AdvancedProfiler"
    output_filename: Optional[str] = None
    line_count_restriction: float = 1.0


@dataclass
class PassThroughProfilerConf:
    _target_: str = "pytorch_lightning.profiler.PassThroughProfiler"


@dataclass
class SimpleProfilerConf:
    _target_: str = "pytorch_lightning.profiler.SimpleProfiler"
    dirpath: Optional[str] = None
    extended: Any = True
