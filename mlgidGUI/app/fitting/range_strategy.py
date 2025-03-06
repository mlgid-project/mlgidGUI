from enum import Enum
from dataclasses import dataclass


class RangeStrategyType(Enum):
    fixed = 'Fixed'
    adjust = 'Adjust'


@dataclass
class RangeStrategy:
    range_factor: float = 1
    strategy_type: RangeStrategyType = RangeStrategyType.adjust
    is_default: bool = True
