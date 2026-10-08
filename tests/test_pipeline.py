import pandas as pd
from datetime import datetime
from src.pipeline import make_synthetic, add_labels, build_panel, evaluate, load_config, safe_corr


def test_execution_and_label_timing():
    cfg = load_config()
    d = make_synthetic(cfg).sort_values(["symbol", "date"])
    x = add_labels(d, cfg)
    one = x[x.symbol == x.symbol.iloc[0]].reset_index(drop=True)
    i = 100
    expected = one.loc[i + 6, "open"] / one.loc[i + 1, "open"] - 1
    assert abs(one.loc[i, "forward_return"] - expected) < 1e-12
    assert one.loc[i, "label_exit_date"] == one.loc[i + 6, "date"]


def test_synthetic_is_marked():
    cfg = load_config()
    assert make_synthetic(cfg)["data_source"].eq("synthetic_validation_only").all()


def test_turnover_tracks_symbols_not_row_ids():
    cfg = load_config()
    panel = build_panel(make_synthetic(cfg), cfg)
    metrics, _ = evaluate(panel, "benchmark", "2019-01-01", "2019-03-31", cfg["quantiles"])
    assert 0 <= metrics["mean_top_group_turnover"] < 1


def test_rank_correlation_without_scipy_runtime_dependency():
    a = pd.Series([10, 20, 30, 40, 50])
    b = pd.Series([1, 4, 2, 3, 5])
    assert abs(safe_corr(a, b, rank=True) - 0.7) < 1e-12


def test_factor_shift_does_not_cross_symbols():
    from src.factors_v1 import candidate_v1
    dates = pd.date_range("2020-01-01", periods=70)
    d = pd.concat([
        pd.DataFrame({"date": dates, "symbol": "A", "close": range(1, 71)}),
        pd.DataFrame({"date": dates, "symbol": "B", "close": range(1001, 1071)})
    ], ignore_index=True).sort_values(["symbol", "date"])
    f = candidate_v1(d)
    assert f[d.symbol.eq("B")].iloc[:60].isna().all()


def test_hac_mean_ci_and_development_guard():
    from src.posthoc_extension import hac_mean_ci, assert_development_bounds
    ci = hac_mean_ci(pd.Series([0.1, -0.1, 0.2, -0.2, 0.0]), maxlags=4)
    assert abs(ci["mean"]) < 1e-12
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert_development_bounds("2020-04-01", "2023-12-29")


def test_revised_exploration_is_cross_sectionally_orthogonal_in_levels():
    from src.factor_exploration_revised import turnover_shock_reversal_v2
    dates = pd.date_range("2020-01-01", periods=70)
    frames = []
    for j in range(12):
        frames.append(pd.DataFrame({"date": dates, "symbol": f"S{j:02d}",
            "close": 10 + j + pd.Series(range(70)) * (0.01 + j / 1000),
            "turnover": 0.01 + ((pd.Series(range(70)) + j) % 9) / 1000}))
    d = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    revised = turnover_shock_reversal_v2(d)
    reversal = -d.groupby("symbol")["close"].pct_change(5, fill_method=None)
    last = d.date.eq(d.date.max())
    # The synthetic cross-section is nearly constant, so correlation is numerically
    # ill-conditioned; the residual should still be effectively orthogonal.
    assert abs(revised[last].corr(reversal[last])) < 1e-3


def test_posthoc_run_directories_are_isolated(tmp_path):
    from src.posthoc_extension import allocate_run_directory
    fixed = datetime(2026, 10, 8, 12, 30, 45, 123456)
    first = allocate_run_directory(tmp_path, fixed)
    marker = first / "immutable_audit_marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    second = allocate_run_directory(tmp_path, fixed)
    assert first != second
    assert first.name == "run_20261008_123045_123456"
    assert second.name == "run_20261008_123045_123456_001"
    assert marker.read_text(encoding="utf-8") == "preserve"
