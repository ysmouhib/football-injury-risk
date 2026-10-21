"""Interpretability helpers: shape criterion and encoded-feature mapping."""

import numpy as np
import pandas as pd

from src import interpret


def test_shape_verdict_detects_u_and_monotone():
    rows = pd.DataFrame({"x": np.linspace(0, 100, 1001)})
    grid = np.linspace(1, 99, 25)
    u = interpret.shape_verdict(rows, "x", grid, 0.0005 * (grid - 50) ** 2)
    assert u["shape"] == "U/J-shaped"
    rising = interpret.shape_verdict(rows, "x", grid, 0.01 * grid)
    assert rising["shape"] == "monotone/flat" and rising["end_to_end"] > 0


def test_encoded_names_map_back_to_features():
    numeric, categorical = ["age", "prev1_apps"], ["position_group", "season_month"]
    assert interpret._original_feature("num__age", numeric, categorical) == "age"
    assert interpret._original_feature("num__missingindicator_age", numeric, categorical) == "age"
    assert interpret._original_feature("cat__position_group_Defender", numeric, categorical) == "position_group"
    assert interpret._original_feature("cat__season_month_m07", numeric, categorical) == "season_month"
