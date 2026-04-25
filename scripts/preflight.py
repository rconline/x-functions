"""Re-exports `spark_ai_functions.preflight.run_preflight` as a script entry."""

from __future__ import annotations

import sys

from spark_ai_functions.preflight import run_preflight


if __name__ == "__main__":
    sys.exit(0 if run_preflight() else 1)
