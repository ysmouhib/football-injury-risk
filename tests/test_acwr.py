"""The 7:28-day acute:chronic workload ratio: window edges, spikes and no future information."""

import numpy as np
import pandas as pd

from src.acwr import acwr

T = pd.Timestamp("2025-10-19")  # a Sunday


def _minutes(*days_and_minutes, player=1):
    return pd.DataFrame({"player_id": player, "date": [T - pd.Timedelta(days=d) for d, _ in days_and_minutes],
                         "minutes": [m for _, m in days_and_minutes]})


def _origin(player=1, t=T):
    return pd.DataFrame({"player_id": [player], "t": [t]})


def test_steady_load_gives_one():
    out = acwr(_minutes((0, 90), (7, 90), (14, 90), (21, 90)), _origin())
    assert out.loc[0, "acute"] == 90 and out.loc[0, "chronic"] == 90
    assert out.loc[0, "acwr"] == 1.0 and not out.loc[0, "flag"]


def test_spike_is_flagged():
    out = acwr(_minutes((2, 90)), _origin())
    assert out.loc[0, "chronic"] == 22.5 and out.loc[0, "acwr"] == 4.0 and out.loc[0, "flag"]


def test_window_edges():
    # day t-6 is acute, t-7 is not; t-27 is chronic, t-28 is not
    out = acwr(_minutes((6, 10), (7, 20), (27, 40), (28, 80)), _origin())
    assert out.loc[0, "acute"] == 10
    assert out.loc[0, "chronic"] == (10 + 20 + 40) / 4


def test_future_minutes_are_ignored():
    past = _minutes((3, 90), (10, 60))
    future = pd.concat([past, pd.DataFrame({"player_id": [1, 1], "date": [T + pd.Timedelta(days=1),
                                                                           T + pd.Timedelta(days=5)],
                                            "minutes": [90, 90]})])
    pd.testing.assert_frame_equal(acwr(past, _origin()), acwr(future, _origin()))


def test_undefined_without_chronic_minutes_and_other_players_ignored():
    out = acwr(_minutes((40, 90)).pipe(lambda m: pd.concat([m, _minutes((1, 90), player=2)])),
               pd.concat([_origin(1), _origin(3)], ignore_index=True))
    assert np.isnan(out["acwr"]).all() and not out["flag"].any()
    assert (out["acute"] == 0).all()
