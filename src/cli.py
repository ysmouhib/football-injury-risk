"""Task runner: python -m src.cli {data,features,test,train,evaluate,report,freeze,analyse} [config]
and python -m src.cli prospective --date YYYY-MM-DD."""

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


def _freeze(_: Path | None) -> int:
    from . import champion

    champion.freeze()
    return 0


def _analyse(config: Path | None) -> int:
    from . import practice

    practice.run(_require(config))
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
    "freeze": _freeze,
    "analyse": _analyse,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("task", choices=sorted([*TASKS, "prospective"]))
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--date", help="prospective: score as of this date (YYYY-MM-DD)")
    args = parser.parse_args()
    if args.task == "prospective":
        from . import prospective

        if not args.date:
            raise SystemExit("prospective needs --date YYYY-MM-DD")
        prospective.main(args.date)
        sys.exit(0)
    sys.exit(TASKS[args.task](args.config))


if __name__ == "__main__":
    main()
