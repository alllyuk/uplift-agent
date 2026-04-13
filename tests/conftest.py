"""Test configuration and path setup.

Ensures the repository root is on sys.path so that `import sme_causal`
works consistently across different IDE and runner setups.
"""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

