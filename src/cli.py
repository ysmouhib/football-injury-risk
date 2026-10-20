"""Task runner: python -m src.cli {data,features,test,train,evaluate,report}."""

from __future__ import annotations

import argparse
import subprocess
import sys

from .constants import REPO_ROOT


def _data() -> int:
    from . import data_audit

    data_audit.main()
    return 0


def _features() -> int:
    from . import panel

    panel.main()
    return 0


def _test() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT)


def _not_wired(stage: str):
    def run() -> int:
        print(f"not available yet: wired in {stage}", file=sys.stderr)
        return 2
    return run


TASKS = {
    "data": _data,
    "features": _features,
    "test": _test,
    "train": _not_wired("Stage 1"),
    "evaluate": _not_wired("Stage 1"),
    "report": _not_wired("Stage 1"),
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("task", choices=sorted(TASKS))
    sys.exit(TASKS[parser.parse_args().task]())


if __name__ == "__main__":
    main()
