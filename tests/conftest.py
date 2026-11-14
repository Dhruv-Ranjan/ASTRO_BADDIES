"""Make the copied source package importable from a repository checkout."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "FINAL_PROJECT_NASA" / "src"))
sys.path.insert(0, str(ROOT))

# These legacy validation scripts execute immediately and require the excluded
# multi-gigabyte sequence dataset. Run them directly after restoring inputs.
collect_ignore = ["test_end_to_end.py", "test_inference_validation.py"]
