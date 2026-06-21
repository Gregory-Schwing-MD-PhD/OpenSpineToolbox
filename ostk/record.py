"""ostk.record — the per-case measurement output contract (SPEC §4)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class Measurement:
    case_id: str
    parameter: str
    value: Optional[float]
    units: str = "degrees"
    landmarks_world_mm: Dict[str, list] = field(default_factory=dict)
    fit_residuals: Dict[str, float] = field(default_factory=dict)
    qc_flags: List[str] = field(default_factory=lambda: ["ok"])
    method_version: str = ""
    supine_ct: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
