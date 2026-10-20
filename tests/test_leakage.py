"""Mandatory leakage tests on the built feature store."""

import numpy as np
import pandas as pd

from src import splits
from src.labels import LABEL_SPECS, PRIMARY_LABELS, label_horizon
from src.panel import FEATURE_SETS

MAX_ABS_CORRELATION = 0.95


def _feature_columns(panel):
    numeric = [c for num, _ in FEATURE_SETS.values() for c in num if c in panel.columns]
    categorical = [c for _, cat in FEATURE_SETS.values() for c in cat if c in panel.columns]
    return list(dict.fromkeys(numeric)), list(dict.fromkeys(categorical))


def test_feature_timestamps_precede_outcomes(panel):
    outcome_start = panel["t"] + pd.Timedelta(days=1)
    assert (panel["feat_ts"] < outcome_start).all()
    assert (panel["feat_ts"] <= panel["t"]).all()
    for label, (horizon, _, _) in LABEL_SPECS.items():
        onset = panel[f"onset_{label}"]
        has = onset.notna()
        assert (onset[has] > panel.loc[has, "t"]).all(), label
        assert (onset[has] <= panel.loc[has, "t"] + pd.Timedelta(days=horizon)).all(), label
        assert (panel.loc[has, "feat_ts"] < onset[has]).all(), label
        labelled = panel[label].notna()
        assert (panel.loc[labelled, label].eq(1) == has[labelled]).all(), label


def test_feature_list_excludes_outcome_columns():
    for numeric, categorical in FEATURE_SETS.values():
        for name in numeric + categorical:
            assert not name.startswith(("y", "onset")), name
            assert name not in LABEL_SPECS


def test_player_sets_disjoint(panel, cfg):
    roles = splits.player_roles(panel["player_id"], cfg["splits"]["group_fractions"], cfg["seed"])
    assert roles.groupby(panel["player_id"]).nunique().eq(1).all()
    development_players = set()
    for label in PRIMARY_LABELS:
        for split in splits.rolling_origins(panel, cfg, label):
            train = set(panel["player_id"].iloc[split.train_idx])
            evaluate = set(panel["player_id"].iloc[split.eval_idx])
            assert train and evaluate
            assert train.isdisjoint(evaluate), split.name
            development_players |= train | evaluate
    final = splits.final_split(panel, cfg, None, label_horizon("y28"), allow_holdout=True)
    holdout_players = set(panel["player_id"].iloc[final.eval_idx])
    assert holdout_players
    assert holdout_players.isdisjoint(panel["player_id"].iloc[final.train_idx])
    assert holdout_players.isdisjoint(development_players)


def test_labels_and_risk_set_match_bruteforce_recount(panel, cfg):
    """Recount sampled development rows directly from the raw injury table."""
    from src import datasets, labels

    roles = splits.player_roles(panel["player_id"], cfg["splits"]["group_fractions"], cfg["seed"])
    dev = panel[(panel["season"] < cfg["splits"]["holdout_season"]) & roles.ne("test")]
    sample = pd.concat([
        dev[dev["y28"].eq(1)].sample(1000, random_state=0),
        dev.sample(2000, random_state=1),
    ])
    events = labels.prepare_events(datasets.load_injuries(), labels.load_label_mapping())
    events = events[events["player_id"].isin(sample["player_id"])]
    by_player = dict(tuple(events.groupby("player_id")))
    empty = events.iloc[:0]
    has_history = "injuries_365d" in panel.columns
    for row in sample.itertuples():
        ev = by_player.get(row.player_id, empty)
        assert not ((ev["from_date"] <= row.t) & (ev["end_date"] > row.t)).any(), "absent player in risk set"
        if has_history:
            past = ev[ev["category"].isin(["injury", "unknown"]) & (ev["from_date"] <= row.t)]
            recent = past[past["from_date"] > row.t - pd.Timedelta(days=365)]
            assert row.injuries_365d == len(recent), (row.player_id, row.t)
            assert row.days_lost_365d == recent["days_missed"].sum(), (row.player_id, row.t)
            assert row.career_injuries == len(past), (row.player_id, row.t)
            if len(past):
                assert row.days_since_return == (row.t - past["end_date"].max()).days
                assert row.feat_ts >= past["end_date"].max()
            else:
                assert row.no_prior_injury == 1
        for label, (horizon, min_days, kind) in LABEL_SPECS.items():
            window = ev[(ev["from_date"] > row.t) & (ev["from_date"] <= row.t + pd.Timedelta(days=horizon))]
            long_enough = window["days_missed"] >= min_days
            injury, unknown = window["category"].eq("injury"), window["category"].eq("unknown")
            if kind == "soft":
                pos = (injury & window["soft_tissue"] & long_enough).any()
                amb = (long_enough & (unknown | (injury & window["tissue"].eq("unspecified")))).any()
            else:
                pos = (injury & long_enough).any()
                amb = (unknown & long_enough).any()
            got = getattr(row, label)
            if pos:
                assert got == 1, (row.player_id, row.t, label)
            elif amb:
                assert np.isnan(got), (row.player_id, row.t, label)
            else:
                assert got == 0, (row.player_id, row.t, label)


def _assert_chronological(panel, split, embargo_days):
    train = panel.iloc[split.train_idx]
    evaluate = panel.iloc[split.eval_idx]
    last_train_outcome = (train["t"] + pd.Timedelta(days=split.horizon_days)).max()
    assert last_train_outcome + pd.Timedelta(days=embargo_days) <= evaluate["t"].min(), split.name
    assert train["season"].max() < split.eval_season, split.name
    assert set(evaluate["season"]) == {split.eval_season}, split.name


def test_split_boundaries_chronological_with_embargo(panel, cfg):
    embargo = cfg["splits"]["embargo_days"]
    for label in PRIMARY_LABELS:
        origins = splits.rolling_origins(panel, cfg, label)
        assert len(origins) >= 3
        assert [s.eval_season for s in origins] == sorted(s.eval_season for s in origins)
        for split in origins:
            assert split.eval_season < cfg["splits"]["holdout_season"]
            _assert_chronological(panel, split, embargo)
    final = splits.final_split(panel, cfg, None, label_horizon("y28"), allow_holdout=True)
    _assert_chronological(panel, final, embargo)


def test_no_feature_has_impossible_label_correlation(panel, cfg):
    # development rows only: holdout-season and test-group rows are never inspected
    roles = splits.player_roles(panel["player_id"], cfg["splits"]["group_fractions"], cfg["seed"])
    dev = panel[(panel["season"] < cfg["splits"]["holdout_season"]) & roles.ne("test")]
    numeric, categorical = _feature_columns(panel)
    X = pd.concat([dev[numeric].astype(float), pd.get_dummies(dev[categorical], dtype=float)], axis=1)
    for label in LABEL_SPECS:
        y = dev[label]
        keep = y.notna().to_numpy()
        Xl, yl = X.to_numpy()[keep], y.to_numpy()[keep]
        for j, name in enumerate(X.columns):
            col = Xl[:, j]
            ok = ~np.isnan(col)
            if col[ok].std() == 0:
                continue
            r = np.corrcoef(col[ok], yl[ok])[0, 1]
            assert abs(r) < MAX_ABS_CORRELATION, f"{name} vs {label}: r={r:.3f}"
