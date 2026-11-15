"""Bind a confirmation mapping to the actual development normalization bytes."""
import hashlib
import json
from pathlib import Path

from .dataset_contract import load_contract_dataset


def check_frozen_normalization(root: Path, compare_roots) -> list[str]:
    descriptor, _ = load_contract_dataset(root)
    name = descriptor.get("normalization")
    if not name:
        return ["Confirmation dataset has no normalization artifact."]
    try:
        own = json.loads((root / name).read_text(encoding="utf-8"))
        for other_root in compare_roots:
            other_root = Path(other_root).resolve()
            other_descriptor, _ = load_contract_dataset(other_root)
            source = other_root / str(other_descriptor.get("normalization", "topo_normalization.json"))
            content = source.read_bytes()
            reference = json.loads(content)
            if own.get("source_sha256") != hashlib.sha256(content).hexdigest():
                continue
            # Compare every source field, allowing only application provenance to differ.
            application_fields = {"application_policy", "fit_performed_on_this_dataset", "frozen_from_dataset", "source_sha256"}
            if any(own.get(key) != value for key, value in reference.items() if key not in application_fields):
                return ["Frozen normalization parameters differ from their declared source bytes."]
            if reference.get("source_split") != "train" or reference.get("fit_performed_on_this_dataset") is False:
                return ["Normalization reference must be an original development training fit."]
            return []
    except (OSError, ValueError, TypeError) as exc:
        return [f"Cannot verify frozen normalization: {exc}"]
    return ["Frozen normalization source SHA256 does not match any comparison development dataset."]
