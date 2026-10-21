"""Prospective harness: prediction time, snapshot guard, append-only log, frozen scoring and
feature parity with the training panel."""

import numpy as np
import pandas as pd
import pytest

from src import prospective
from src.calibration import PlattCalibrator
from src.champion import Frozen, model_hash
from src.constants import DATALAKE_DIR
from src.models import build_model, fit_categories, prepare_X

NUMERIC, CATEGORICAL = ["a", "b"], ["c"]
SPEC = {"kind": "logistic", "params": {"C": 1.0, "class_weight": "balanced", "max_iter": 500}}


def _frozen(label="y7", seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400), "c": rng.choice(["x", "y"], 400)})
    y = (rng.random(400) < 1 / (1 + np.exp(-(df["a"] - 2)))).astype(int)
    categories = fit_categories(df, CATEGORICAL)
    model = build_model(SPEC, NUMERIC, CATEGORICAL, 1).fit(prepare_X(df, NUMERIC, CATEGORICAL, categories), y)
    calibrator = PlattCalibrator().fit(model.predict_proba(prepare_X(df, NUMERIC, CATEGORICAL, categories))[:, 1], y)
    calibrator.a += shift
    return Frozen(label, NUMERIC, CATEGORICAL, categories, model, calibrator), df


def test_prediction_sunday():
    assert prospective.prediction_sunday("2025-10-22") == pd.Timestamp("2025-10-19")
    assert prospective.prediction_sunday("2025-10-19") == pd.Timestamp("2025-10-19")
    assert prospective.prediction_sunday("2025-10-25") == pd.Timestamp("2025-10-19")


def test_snapshot_guard():
    cfg = {"labels": {"observed_until": "2026-09-28"}, "competitions": ["GB1", "ES1"]}
    tcs = pd.DataFrame({"season_id": [2025, 2025, 2024], "competition_id": ["GB1", "ES1", "GB1"]})
    prospective.check_coverage(pd.Timestamp("2025-10-19"), cfg, tcs)
    with pytest.raises(SystemExit, match="after the data snapshot"):
        prospective.check_coverage(pd.Timestamp("2026-10-04"), cfg, tcs)
    with pytest.raises(SystemExit, match="no 2024/25 club table"):
        prospective.check_coverage(pd.Timestamp("2024-10-20"), cfg, tcs)


def test_fingerprint_is_deterministic_and_parameter_sensitive():
    a, _ = _frozen()
    b, _ = _frozen()
    c, _ = _frozen(shift=1e-9)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    assert model_hash({"y7": a, "y28": b}) == model_hash({"y28": b, "y7": a})


def test_score_and_append_only(tmp_path):
    m7, df = _frozen("y7")
    m28, _ = _frozen("y28", shift=1.0)
    rows = df.head(5).assign(player_id=np.arange(5), t=pd.Timestamp("2025-10-19"))
    scored = prospective.score(rows, {"y7": m7, "y28": m28}).assign(
        model_hash="m", git_hash="g", scored_at="2026-10-21T10:00:00Z", features_sha256="f")
    assert list(scored.columns[:4]) == ["player_id", "week", "p7", "p28"]
    assert np.allclose(scored["p7"], m7.predict(rows)[1]) and (scored["p28"] > scored["p7"]).all()
    path = tmp_path / "2025-10-22.csv"
    prospective.append(scored, path)
    first = path.read_text(encoding="utf-8")
    prospective.append(scored.assign(scored_at="2026-10-21T11:00:00Z"), path)
    second = path.read_text(encoding="utf-8")
    assert second.startswith(first)
    lines = second.splitlines()
    assert lines[0] == ",".join(prospective.COLUMNS) and len(lines) == 11
    assert sum(line.startswith("player_id") for line in lines) == 1
    back = pd.read_csv(path, float_precision="round_trip")
    assert np.array_equal(back["p7"].to_numpy()[:5], scored["p7"].to_numpy())
    assert (back["week"] == "2025-10-19").all()


def test_append_refuses_foreign_file(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("something,else\n1,2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unexpected header"):
        prospective.append(pd.DataFrame(columns=prospective.COLUMNS), path)


@pytest.mark.skipif(not DATALAKE_DIR.exists(), reason="datalake not available")
def test_features_match_the_training_panel(panel):
    """Rows built for one historical Sunday equal the feature store's rows of that Sunday."""
    t = pd.Timestamp("2023-10-15")
    built = prospective.features_at(t).sort_values("player_id", ignore_index=True)
    stored = panel[panel["t"].eq(t)].sort_values("player_id", ignore_index=True)
    columns = [c for c in built.columns if c in stored.columns]
    assert len(built) == len(stored) > 1000
    assert set(stored.columns) - set(built.columns) == {
        c for c in stored.columns if c.startswith(("y7", "y28", "onset_", "surv"))}
    pd.testing.assert_frame_equal(built[columns], stored[columns], check_dtype=False)
