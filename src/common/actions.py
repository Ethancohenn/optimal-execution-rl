from __future__ import annotations

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ActionSpec:
    fractions: List[float]

DEFAULT_ACTION_SPEC = ActionSpec(fractions=[0.0, 0.25, 0.5, 0.75, 1.0])

def action_to_fraction(action: int, spec: ActionSpec = DEFAULT_ACTION_SPEC) -> float:
    if action < 0 or action >= len(spec.fractions):
        raise ValueError(f"Action {action} out of range [0, {len(spec.fractions)-1}]")
    return float(spec.fractions[action])

def action_to_qty(action: int, remaining_inventory: float, max_trade_size: float,
                  spec: ActionSpec = DEFAULT_ACTION_SPEC) -> float:
    fraction = action_to_fraction(action, spec)
    qty = fraction * float(max_trade_size)
    return float(min(float(remaining_inventory), qty))
