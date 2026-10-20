"""Unit tests on synthetic inputs for labels, absences, mapping, membership, exposure and splits."""

import numpy as np
import pandas as pd
import pytest

from src import exposure, labels, membership, splits
from src.panel import last_completed_season

T0 = pd.Timestamp("2021-03-07")  # a Sunday
OBSERVED_UNTIL = pd.Timestamp("2026-09-28")


def _events(rows):
    """rows: (player_id, onset offset days from T0, days_missed, category, tissue)."""
    ev = pd.DataFrame(rows, columns=["player_id", "offset", "days_missed", "category", "tissue"])
    ev["from_date"] = T0 + pd.to_timedelta(ev["offset"], unit="D")
    ev["end_date"] = ev["from_date"] + pd.to_timedelta(ev["days_missed"] - 1, unit="D")
    ev["soft_tissue"] = ev["tissue"].isin(["muscle", "tendon"])
    return ev.drop(columns="offset")


def _labelled(rows, t=T0, observed_until=OBSERVED_UNTIL):
    ev = _events(rows)
    panel = pd.DataFrame({"player_id": sorted(ev["player_id"].unique()), "t": t})
    return labels.attach_labels(panel, ev, observed_until).set_index("player_id")


def test_outcome_window_boundaries():
    out = _labelled([
        (1, 0, 10, "injury", "muscle"),   # onset on t itself belongs to the previous row
        (2, 1, 10, "injury", "muscle"),
        (3, 7, 10, "injury", "muscle"),
        (4, 8, 10, "injury", "muscle"),
        (5, 28, 10, "injury", "muscle"),
        (6, 29, 10, "injury", "muscle"),
    ])
    assert out["y7"].tolist() == [0, 1, 1, 0, 0, 0]
    assert out["y28"].tolist() == [0, 1, 1, 1, 1, 0]
    assert out.loc[4, "onset_y28"] == T0 + pd.Timedelta(days=8)
    assert pd.isna(out.loc[4, "onset_y7"])


def test_severity_categories_and_masking():
    out = _labelled([
        (1, 3, 5, "injury", "muscle"),       # below 7 days, counts at >= 3
        (2, 3, 10, "unknown", "n/a"),        # ambiguous -> masked
        (3, 3, 10, "unknown", "n/a"),
        (3, 5, 10, "injury", "ligament_joint"),  # a known injury wins over the unknown one
        (4, 3, 10, "illness", "n/a"),        # not an injury
        (5, 10, 10, "injury", "muscle"),
        (6, 10, 10, "injury", "unspecified"),  # could be soft tissue
    ])
    assert out.loc[1, "y7"] == 0 and out.loc[1, "y7_ge3"] == 1
    assert np.isnan(out.loc[2, "y7"]) and np.isnan(out.loc[2, "y28"])
    assert out.loc[3, "y7"] == 1
    assert out.loc[4, "y7"] == 0 and out.loc[4, "y28"] == 0
    assert out.loc[5, "y28_soft"] == 1
    assert out.loc[6, "y28"] == 1 and np.isnan(out.loc[6, "y28_soft"])


def test_unobservable_windows_are_masked():
    out = _labelled([(1, 1, 10, "injury", "muscle")], observed_until=T0 + pd.Timedelta(days=10))
    assert out.loc[1, "y7"] == 1
    assert np.isnan(out.loc[1, "y28"])


def test_absence_weeks():
    ev = pd.DataFrame({
        "player_id": [1, 2, 3],
        "from_date": pd.to_datetime(["2021-03-03", "2021-03-07", "2021-03-08"]),
        "end_date": pd.to_datetime(["2021-03-14", "2021-03-08", "2021-03-10"]),
    })
    absent = labels.absence_weeks(ev, pd.Timestamp("2021-02-28"), pd.Timestamp("2021-03-21"))
    got = sorted(zip(absent["player_id"], absent["t"]))
    # player 1 is back on Mar 15 (end_date = last day missed), so not absent on Sunday Mar 14
    assert got == [(1, pd.Timestamp("2021-03-07")), (2, pd.Timestamp("2021-03-07"))]


@pytest.mark.parametrize("reason, category, body_part, tissue", [
    ("Hamstring injury", "injury", "hamstring", "muscle"),
    ("Whiplash", "injury", "head_neck", "other"),
    ("Achilles heel problems", "injury", "lower_leg", "tendon"),
    ("Torn lateral ankle ligament", "injury", "ankle", "ligament_joint"),
    ("Inflammation of the biceps tendon in the thigh", "injury", "hamstring", "tendon"),
    ("Outer ligament tear", "injury", "ankle", "ligament_joint"),
    ("Inner ligament injury", "injury", "knee", "ligament_joint"),
    ("Groin surgery", "injury", "groin_hip", "other"),
    ("Coccyx bruise", "injury", "trunk", "contusion"),
    ("Bruise on shinbone", "injury", "lower_leg", "contusion"),
    ("Fracture of the humeral head", "injury", "upper_limb", "bone"),
    ("Knock", "injury", "unspecified", "contusion"),
    ("Ill", "illness", "n/a", "n/a"),
    ("Corona virus", "illness", "n/a", "n/a"),
    ("Tonsillitis", "illness", "n/a", "n/a"),
    ("Toothache", "illness", "n/a", "n/a"),
    ("unknown injury", "unknown", "n/a", "n/a"),
    ("surgery", "unknown", "n/a", "n/a"),
    ("Rest", "non_injury", "n/a", "n/a"),
    ("Fitness", "non_injury", "n/a", "n/a"),
    ("traffic accident", "non_sport", "n/a", "n/a"),
])
def test_reason_mapping(reason, category, body_part, tissue):
    row = labels.map_reasons(pd.Series([reason]), labels.load_label_mapping()).iloc[0]
    assert (row["category"], row["body_part"], row["tissue"]) == (category, body_part, tissue)


def test_last_completed_season_respects_overrides():
    t = pd.Series(pd.to_datetime(["2020-07-05", "2020-08-23", "2020-08-30", "2021-06-27", "2021-07-04"]))
    got = last_completed_season(t, {2019: "2020-08-23"}).tolist()
    assert got == [2018, 2018, 2019, 2019, 2020]


def test_membership_applies_loan_return_before_same_day_move():
    transfers = pd.DataFrame({
        "player_id": [1, 1, 1, 1],
        "transfer_date": pd.to_datetime(["2020-07-01", "2021-01-15", "2021-06-30", "2021-06-30"]),
        "to_team_id": [10, 20, 30, 10],
        "transfer_type": ["Transfer", "Loan", "Loan", "Return from loan"],
    })
    top = pd.DataFrame({"club_id": [10, 20, 10], "season": [2020, 2020, 2021], "competition_id": "GB1"})
    weeks = membership.weekly_membership(
        membership.club_spells(transfers), top, pd.Timestamp("2020-07-05"), pd.Timestamp("2021-08-01"))
    by_t = weeks.set_index("t")["club_id"]
    assert by_t[pd.Timestamp("2020-07-05")] == 10
    assert by_t[pd.Timestamp("2021-01-10")] == 10
    assert by_t[pd.Timestamp("2021-01-17")] == 20
    assert weeks["t"].max() == pd.Timestamp("2021-06-27")  # then on loan at non-top club 30
    assert not weeks.duplicated(["player_id", "t"]).any()


def test_appearance_components_and_minutes_reconstruction():
    perf = pd.DataFrame({
        "nb_on_pitch": [10, 1, 5],
        "subed_in": [3, 1, 0],
        "subed_out": [2, 1, 0],
        "goals": [2.0, 0.0, np.nan],
        "minutes_played": [450.0, np.nan, 450.0],
    })
    comps = exposure.appearance_components(perf)
    assert comps.loc[0, ["full_games", "subbed_off_starts", "sub_apps"]].tolist() == [5, 2, 3]
    # subbed on and off in the same game: one substitute appearance, no start
    assert comps.loc[1, ["full_games", "subbed_off_starts", "sub_apps"]].tolist() == [0, 0, 1]
    assert (comps[["full_games", "subbed_off_starts", "sub_apps"]].sum(axis=1) == perf["nb_on_pitch"]).all()
    total = exposure.reconstruct_total_minutes(perf)
    assert total[0] == 900 and np.isnan(total[1]) and total[2] == 450


def test_player_roles_deterministic_and_proportional():
    ids = pd.Series(np.arange(1, 20001))
    fractions = {"train": 0.6, "validation": 0.2, "test": 0.2}
    roles = splits.player_roles(ids, fractions, seed=7)
    again = splits.player_roles(ids.iloc[::-1], fractions, seed=7).sort_index()
    assert roles.equals(again)
    shares = roles.value_counts(normalize=True)
    for role, frac in fractions.items():
        assert abs(shares[role] - frac) < 0.02


def test_holdout_is_guarded(cfg):
    panel = pd.DataFrame({"player_id": [1], "t": [T0], "season": [2020], "y7": [0.0]})
    with pytest.raises(ValueError):
        splits.origin_split(panel, cfg, cfg["splits"]["holdout_season"], "y7")
    with pytest.raises(PermissionError):
        splits.final_split(panel, cfg, "y7", 7)


def test_history_windows_and_recency():
    from src import history

    t = pd.Timestamp("2022-03-06")  # a Sunday
    ev = pd.DataFrame({
        "player_id": [1, 1, 1, 1, 2],
        "from_date": [t - pd.Timedelta(days=d) for d in (365, 364, 10, 800)] + [t - pd.Timedelta(days=50)],
        "days_missed": [20, 5, 8, 30, 3],
        "category": ["injury", "unknown", "injury", "injury", "illness"],
        "body_part": ["hamstring", "n/a", "knee", "hamstring", "n/a"],
        "tissue": ["muscle", "n/a", "ligament_joint", "muscle", "n/a"],
    })
    ev["end_date"] = ev["from_date"] + pd.to_timedelta(ev["days_missed"] - 1, unit="D")
    panel = pd.DataFrame({"player_id": [1, 2], "t": [t, t]})
    out = history.history_features(panel, ev).set_index(panel["player_id"])
    assert out.loc[1, "injuries_365d"] == 2  # onset exactly 365 days ago is outside (t-365, t]
    assert out.loc[1, "days_lost_365d"] == 13
    assert out.loc[1, "injuries_730d"] == 3 and out.loc[1, "career_injuries"] == 4
    assert out.loc[1, "hamstring_injuries_730d"] == 1 and out.loc[1, "muscle_injuries_730d"] == 1
    assert out.loc[1, "days_since_return"] == 3 and out.loc[1, "returned_within_28d"] == 1
    assert out.loc[1, "history_ts"] == t - pd.Timedelta(days=3)
    # illness is not injury history
    assert out.loc[2, "no_prior_injury"] == 1 and out.loc[2, "career_injuries"] == 0


def test_history_refuses_unfinished_injuries():
    from src import history

    t = pd.Timestamp("2022-03-06")
    ev = pd.DataFrame({"player_id": [1], "from_date": [t - pd.Timedelta(days=2)], "end_date": [t + pd.Timedelta(days=5)],
                       "days_missed": [8], "category": ["injury"], "body_part": ["knee"], "tissue": ["unspecified"]})
    with pytest.raises(ValueError):
        history.history_features(pd.DataFrame({"player_id": [1], "t": [t]}), ev)
