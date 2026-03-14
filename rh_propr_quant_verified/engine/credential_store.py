from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class CredentialStore:
    def __init__(self, root: Path):
        self.path = root / '.runtime_settings.json'

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def save(self, payload: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
