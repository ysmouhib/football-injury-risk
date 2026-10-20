"""Task runner: python -m src.cli {data,features,test,train,evaluate,report} [config]."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .constants import REPO_ROOT


def _data(_: Path | None) -> int:
    from . import data_audit

    data_audit.main()
    return 0


def _features(_: Path | None) -> int:
    from . import panel

    panel.main()
    return 0


def _test(_: Path | None) -> int:
    return subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=REPO_ROOT)


def _train(config: Path | None) -> int:
    from . import train

    train.train(_require(config))
    return 0


def _evaluate(config: Path | None) -> int:
    from . import evaluate

    evaluate.evaluate(_require(config))
    return 0


def _report(_: Path | None) -> int:
    from . import report

    report.main()
    return 0


def _require(config: Path | None) -> Path:
    if config is None:
        raise SystemExit("this task needs an experiment config, e.g. configs/exp-0001.yaml")
    return config


TASKS = {
    "data": _data,
    "features": _features,
    "test": _test,
    "train": _train,
    "evaluate": _evaluate,
    "report": _report,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("config", nargs="?", type=Path)
    args = parser.parse_args()
    sys.exit(TASKS[args.task](args.config))


if __name__ == "__main__":
    main()
