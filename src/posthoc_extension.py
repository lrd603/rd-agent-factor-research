"""Bounded post-hoc diagnostics and one development-only research interaction.

This module never evaluates the configured 2024-2025 holdout. Original outputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .factor_exploration_initial import turnover_shock_reversal_v1
from .factors_v1 import candidate_v1
from .factors_v2 import candidate_v2
from .pipeline import add_labels, cs_process, safe_corr
from .real_pipeline import cfg, load_real

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "outputs" / "posthoc_runs"
OUT: Path | None = None
HYPOTHESIS_SOURCE = ROOT / "provenance" / "model_interactions" / "20261008_hypothesis_generation.json"
REVISION_SOURCE = ROOT / "provenance" / "model_interactions" / "20261008_revision_feedback.json"
INTERACTION_MANIFEST = ROOT / "provenance" / "model_interactions" / "MANIFEST.json"
DEV_START, DEV_END = "2020-04-01", "2023-12-29"
DESIGN_START, DESIGN_END = "2020-04-01", "2022-06-30"
VALID_START, VALID_END = "2022-07-01", "2023-12-29"
Q = 5


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def allocate_run_directory(base: Path = RUNS_ROOT, timestamp: datetime | None = None) -> Path:
    """Atomically allocate a new audit directory; never reuse or overwrite one."""
    stamp = (timestamp or datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S_%f")
    base.mkdir(parents=True, exist_ok=True)
    for suffix in range(1000):
        name = f"run_{stamp}" if suffix == 0 else f"run_{stamp}_{suffix:03d}"
        candidate = base / name
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to allocate unique run directory below {base}")


def require_out() -> Path:
    if OUT is None:
        raise RuntimeError("Run directory has not been allocated")
    return OUT


def validate_historical_interactions() -> None:
    manifest = json.loads(INTERACTION_MANIFEST.read_text(encoding="utf-8"))
    root = INTERACTION_MANIFEST.parent
    for record in manifest["records"]:
        path = root / record["path"]
        actual = sha(path)
        if actual.lower() != record["sha256"].lower():
            raise RuntimeError(f"Historical interaction source hash mismatch: {path}")


def assert_development_bounds(start: str, end: str) -> None:
    c = cfg()
    assert pd.Timestamp(start) >= pd.Timestamp(c["development"]["start"])
    assert pd.Timestamp(end) <= pd.Timestamp(c["development"]["end"])
    assert pd.Timestamp(end) < pd.Timestamp(c["holdout"]["start"])


def build_development_panel(include_revision: bool = False) -> pd.DataFrame:
    c = cfg()
    x = add_labels(load_real(c), c)
    g = x.groupby("symbol", sort=False)["close"]
    x["momentum_20d_raw"] = g.pct_change(20, fill_method=None)
    x["reversal_5d_raw"] = -g.pct_change(5, fill_method=None)
    x["candidate_v1_raw"] = candidate_v1(x)
    x["candidate_v2_raw"] = candidate_v2(x)
    x["exploration_initial_raw"] = turnover_shock_reversal_v1(x)
    names = ["momentum_20d", "reversal_5d", "candidate_v1", "candidate_v2", "exploration_initial"]
    if include_revision:
        from .factor_exploration_revised import turnover_shock_reversal_v2
        x["exploration_revised_raw"] = turnover_shock_reversal_v2(x)
        names.append("exploration_revised")
    for name in names:
        x[name] = cs_process(x[f"{name}_raw"], x, c)
    # Hard boundary: nothing at or after holdout start survives in memory.
    x = x[x["date"] < pd.Timestamp(c["holdout"]["start"])].copy()
    assert x["date"].max() <= pd.Timestamp(DEV_END)
    return x


def slice_period(x: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    assert_development_bounds(start, end)
    end_ts = pd.Timestamp(end)
    return x[(x.date >= pd.Timestamp(start)) & (x.date <= end_ts) &
             (x.label_exit_date <= end_ts) & x.forward_return.notna()].copy()


def daily_evaluation(x: pd.DataFrame, factor: str, start: str, end: str):
    z = slice_period(x, start, end)
    daily, groups, prev_top = [], [], None
    for date, d in z.groupby("date", sort=True):
        d = d[["symbol", factor, "forward_return"]].dropna()
        if len(d) < Q * 2:
            continue
        ranks = d[factor].rank(method="first")
        buckets = pd.qcut(ranks, Q, labels=False) + 1
        top = set(d.loc[buckets == Q, "symbol"])
        turnover = np.nan if prev_top is None else 1 - len(top & prev_top) / max(len(top), 1)
        prev_top = top
        ic = safe_corr(d[factor], d.forward_return)
        rank_ic = safe_corr(d[factor], d.forward_return, rank=True)
        daily.append({"date": date, "factor": factor, "ic": ic, "rank_ic": rank_ic,
                      "turnover_top_group": turnover, "n_symbols": len(d)})
        means = d.assign(group=buckets).groupby("group").forward_return.mean()
        for group, value in means.items():
            groups.append({"date": date, "factor": factor, "group": int(group),
                           "mean_forward_return": float(value)})
    return pd.DataFrame(daily), pd.DataFrame(groups)


def hac_mean_ci(s: pd.Series, maxlags: int = 4) -> dict:
    """Newey-West/Bartlett CI for a mean, implemented without an extra dependency."""
    a = s.dropna().to_numpy(float)
    n = len(a)
    if n < 3:
        return {"mean": np.nan, "hac_se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}
    u = a - a.mean()
    long_run = float(u @ u / n)
    for lag in range(1, min(maxlags, n - 1) + 1):
        gamma = float(u[lag:] @ u[:-lag] / n)
        long_run += 2 * (1 - lag / (maxlags + 1)) * gamma
    se = np.sqrt(max(long_run, 0) / n)
    critical = norm.ppf(0.975)
    return {"mean": float(a.mean()), "hac_se": float(se),
            "ci_low": float(a.mean() - critical * se), "ci_high": float(a.mean() + critical * se),
            "n": n, "maxlags": maxlags}


def summarize(daily: pd.DataFrame, groups: pd.DataFrame, factor: str, period: str, start: str, end: str) -> dict:
    gm = groups.groupby("group").mean(numeric_only=True)["mean_forward_return"]
    a, r = hac_mean_ci(daily.ic), hac_mean_ci(daily.rank_ic)
    return {"period": period, "start": start, "end": end, "factor": factor, "n_dates": len(daily),
            "mean_ic": a["mean"], "ic_hac_se": a["hac_se"], "ic_ci_low": a["ci_low"], "ic_ci_high": a["ci_high"],
            "mean_rank_ic": r["mean"], "rank_ic_hac_se": r["hac_se"],
            "rank_ic_ci_low": r["ci_low"], "rank_ic_ci_high": r["ci_high"],
            "q1_return": gm.get(1, np.nan), "q2_return": gm.get(2, np.nan), "q3_return": gm.get(3, np.nan),
            "q4_return": gm.get(4, np.nan), "q5_return": gm.get(5, np.nan),
            "q5_minus_q1": gm.get(5, np.nan) - gm.get(1, np.nan),
            "mean_top_group_turnover": daily.turnover_top_group.mean()}


def factor_correlations(x: pd.DataFrame, factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = slice_period(x, DEV_START, DEV_END)
    rows = []
    for date, d in z.groupby("date", sort=True):
        for factor in factors:
            for benchmark in ["momentum_20d", "reversal_5d"]:
                rows.append({"date": date, "factor": factor, "benchmark": benchmark,
                             "spearman": safe_corr(d[factor], d[benchmark], rank=True)})
    daily = pd.DataFrame(rows)
    dist = daily.groupby(["factor", "benchmark"]).spearman.agg(
        n="count", mean="mean", std="std", min="min",
        p05=lambda s: s.quantile(.05), p25=lambda s: s.quantile(.25), median="median",
        p75=lambda s: s.quantile(.75), p95=lambda s: s.quantile(.95), max="max").reset_index()
    return daily, dist


def periods() -> list[tuple[str, str, str]]:
    return [("development", DEV_START, DEV_END), ("year_2020", "2020-04-01", "2020-12-31"),
            ("year_2021", "2021-01-01", "2021-12-31"), ("year_2022", "2022-01-01", "2022-12-30"),
            ("year_2023", "2023-01-03", "2023-12-29"), ("design_early", DESIGN_START, DESIGN_END),
            ("validation_late", VALID_START, VALID_END)]


def prepare() -> None:
    out = require_out()
    validate_historical_interactions()
    interaction = HYPOTHESIS_SOURCE
    shutil.copy2(interaction, out / "00_historical_model_interaction_source.json")
    shutil.copy2(ROOT / "src" / "factor_exploration_initial.py", out / "code_before_revision.py")
    write_json(out / "01_exploration_protocol_frozen.json", {
        "frozen_at": now(), "classification": "事后探索；已知历史开发数据上的有限扩展；不是新的完全未见样本确认",
        "forbidden": "No reading or evaluation of 2024-2025 holdout in this extension",
        "development": [DEV_START, DEV_END], "design_early": [DESIGN_START, DESIGN_END],
        "validation_late": [VALID_START, VALID_END], "max_revision": 1, "selected_hypothesis": "E1",
        "initial_code_sha256": sha(ROOT / "src" / "factor_exploration_initial.py"),
        "model_interaction_sha256": sha(interaction),
        "model_interaction_semantics": "Immutable historical source from 2026-10-08; referenced for reproduction, not generated by this run.",
        "selection_of_final_exploration_version": "Choose once after early design diagnostics; then evaluate late validation without further changes. Preserve signed direction.",
        "hac": "Newey-West/Bartlett, maxlags=4, reflecting 5-day overlapping forward returns."
    })
    print(out)


def early_design() -> None:
    out = require_out()
    if not (out / "01_exploration_protocol_frozen.json").exists():
        raise RuntimeError("Run prepare first")
    if (out / "02_early_design_diagnostics.csv").exists():
        raise RuntimeError("Early design already run")
    x = build_development_panel(False)
    factors = ["momentum_20d", "reversal_5d", "candidate_v1", "candidate_v2", "exploration_initial"]
    rows, all_daily, all_groups = [], [], []
    for factor in factors:
        d, g = daily_evaluation(x, factor, DESIGN_START, DESIGN_END)
        rows.append(summarize(d, g, factor, "design_early", DESIGN_START, DESIGN_END))
        all_daily.append(d); all_groups.append(g)
    pd.DataFrame(rows).to_csv(out / "02_early_design_diagnostics.csv", index=False)
    pd.concat(all_daily).to_csv(out / "02_early_design_daily.csv", index=False)
    pd.concat(all_groups).to_csv(out / "02_early_design_groups.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def final_analysis() -> None:
    out = require_out()
    feedback = out / "03_historical_revision_interaction_source.json"
    revised = ROOT / "src" / "factor_exploration_revised.py"
    if not feedback.exists() or not revised.exists():
        raise RuntimeError("Actual feedback record and revised code are required")
    if (out / "10_run_record.json").exists():
        raise RuntimeError("Final analysis already completed")
    x = build_development_panel(True)
    factors = ["momentum_20d", "reversal_5d", "candidate_v1", "candidate_v2",
               "exploration_initial", "exploration_revised"]
    summary_rows, daily_parts, group_parts = [], [], []
    for name, start, end in periods():
        for factor in factors:
            d, g = daily_evaluation(x, factor, start, end)
            summary_rows.append(summarize(d, g, factor, name, start, end))
            d["period"] = name; g["period"] = name
            daily_parts.append(d); group_parts.append(g)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "04_period_diagnostics.csv", index=False)
    pd.concat(daily_parts).to_csv(out / "05_daily_ic_turnover.csv", index=False)
    pd.concat(group_parts).to_csv(out / "06_daily_group_returns.csv", index=False)
    corr_daily, corr_dist = factor_correlations(x, factors)
    corr_daily.to_csv(out / "07_daily_factor_benchmark_spearman.csv", index=False)
    corr_dist.to_csv(out / "07_factor_benchmark_spearman_distribution.csv", index=False)
    shutil.copy2(revised, out / "code_after_revision.py")
    write_json(out / "08_exposure_and_alignment_audit.json", {
        "historical_market_cap": "Unavailable in cached daily files; no size exposure diagnostic or synthetic backfill.",
        "industry": "Unavailable in cached daily files. Any current classification would not establish point-in-time membership, so industry exposure diagnostic is omitted.",
        "signal_timing": "All rolling inputs end at signal-date t; label is qfq open[t+6]/open[t+1]-1.",
        "boundary": "Every period requires label_exit_date <= period end; holdout rows are removed before evaluation.",
        "turnover_formula": "1 - |top_quintile_t intersect top_quintile_previous_signal_date| / |top_quintile_t|.",
        "turnover_limitation": "Membership turnover on adjacent available signal dates; not weight turnover and excludes costs.",
        "five_day_overlap": "Adjacent daily labels share four of five return sessions, inducing serial correlation. HAC maxlags=4 CIs are therefore primary; daily observations are not treated as iid.",
        "original_selection_rule": "Existing project chose V2 because its signed development mean Rank IC (-0.01848) exceeded V1 (-0.02047). It did not choose by absolute IC. No post-holdout sign flip is permitted or performed.",
        "holdout_accessed": False
    })
    dev = summary[summary.period.eq("development")]
    late = summary[summary.period.eq("validation_late")]
    corr = corr_dist[corr_dist.factor.isin(["candidate_v1", "candidate_v2", "exploration_revised"])]
    report = make_report(dev, late, corr)
    (out / "REPORT_CN.md").write_text(report, encoding="utf-8")
    (out / "CLAIMS_CHECKLIST_CN.md").write_text(make_claims(), encoding="utf-8")
    write_json(out / "10_run_record.json", {"completed_at": now(), "status": "completed",
        "original_files_modified": False, "holdout_read_or_run": False,
        "run_directory": str(out),
        "historical_interactions": {"hypothesis_source": str(HYPOTHESIS_SOURCE), "hypothesis_sha256": sha(HYPOTHESIS_SOURCE),
          "revision_source": str(REVISION_SOURCE), "revision_sha256": sha(REVISION_SOURCE),
          "generated_in_this_run": False},
        "before_sha256": sha(out / "code_before_revision.py"), "after_sha256": sha(out / "code_after_revision.py")})
    print(summary[summary.period.isin(["development", "validation_late"])].to_string(index=False))


def f(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x:.4f}"


def make_report(dev: pd.DataFrame, late: pd.DataFrame, corr: pd.DataFrame) -> str:
    out = require_out()
    def row(tab, factor): return tab.set_index("factor").loc[factor]
    v2, new, new_late = row(dev, "candidate_v2"), row(dev, "exploration_revised"), row(late, "exploration_revised")
    years = pd.read_csv(out / "04_period_diagnostics.csv")
    yr = years[(years.factor == "exploration_revised") & years.period.str.startswith("year_")]
    concentration = ", ".join(f"{r.period[-4:]} RankIC={f(r.mean_rank_ic)}" for r in yr.itertuples())
    rc = corr[corr.factor.eq("exploration_revised")].set_index("benchmark")
    return f"""# 有限事后扩展报告

## 边界

原 V1/V2、冻结记录及 2024—2025 留出期结果均未修改。本轮只使用原开发期，属于查看过旧留出期之后的事后探索，不能称为独立确认或新的完全未见样本检验；本轮没有读取或运行留出期。

## 补充诊断

原规则按“更高的有符号开发期平均 Rank IC”选择 V2：V2 为 {f(v2.mean_rank_ic)}，V1 更低；这不是按绝对值选择。即使旧留出期出现负值，也没有翻转方向。五日标签逐日重叠，相邻观测共享四个收益日，普通 iid 标准误会偏乐观；表中采用 Newey–West/Bartlett、4 阶滞后的 95% 区间。V2 开发期 IC={f(v2.mean_ic)} [{f(v2.ic_ci_low)}, {f(v2.ic_ci_high)}]，Rank IC={f(v2.mean_rank_ic)} [{f(v2.rank_ic_ci_low)}, {f(v2.rank_ic_ci_high)}]。

每日横截面秩相关显示，探索因子与 20 日动量/5 日反转的平均相关分别为 {f(rc.loc['momentum_20d','mean'])} / {f(rc.loc['reversal_5d','mean'])}；完整分布见 CSV。线性残差化没有消除秩意义下的反转暴露，反而留下较强的负秩相关，因此不能宣称已解决冗余。年度结果为：{concentration}。符号跨年变化，表现不稳定，不能只看全期均值。

## 新增探索

Codex 真实提出三项假设并在数值诊断前选择 E1。早期设计段仅允许一次修改；修改后的开发期 IC={f(new.mean_ic)} [{f(new.ic_ci_low)}, {f(new.ic_ci_high)}]，Rank IC={f(new.mean_rank_ic)} [{f(new.rank_ic_ci_low)}, {f(new.rank_ic_ci_high)}]，Q5-Q1={f(new.q5_minus_q1)}，最高组成员换手={f(new.mean_top_group_turnover)}。后期验证段（仍是已知历史开发数据）Rank IC={f(new_late.mean_rank_ic)} [{f(new_late.rank_ic_ci_low)}, {f(new_late.rank_ic_ci_high)}]。区间跨零、分组差为负且换手偏高，本轮没有形成支持新因子的稳健证据；结果照实保留，未继续搜索。

缓存没有历史市值与可靠的点时行业分类，因此没有强行补齐暴露诊断。换手是相邻信号日最高五分位成员替换率，不是含权重与成本的真实组合换手。

## 复现与监督

`python run_posthoc_extension.py` 每次自动创建新的时间戳目录并引用冻结的历史交互来源。数据加载、时间对齐、横截面处理、IC、HAC、分组收益、换手与报告表由脚本自动完成；三项假设和一次反馈建议来自 2026-10-08 的历史 Codex 交互，本次复现不会把它们标记为新生成。人工仍需监督经济含义、数据来源和研究边界。
"""


def make_claims() -> str:
    return """# 主张核对表

|主张|分类|状态/限制|
|---|---|---|
|原 V1/V2 和冻结记录保持不变|原实验|是；本轮只新增文件|
|旧 2024—2025 结果是独立确认|原实验|只对原冻结实验按其原流程成立；本轮不得复用该称谓|
|V2 按更高的有符号 Rank IC 选择|补充诊断|是；未按绝对值，未事后翻转方向|
|HAC 区间消除所有依赖问题|补充诊断|否；4 阶 Newey–West 仅缓解五日重叠导致的序列相关|
|新因子通过全新未见样本验证|新增探索|否；后期段是预划分的开发期内部验证，历史整体已知|
|行业和规模暴露已充分控制|新增探索|否；缺历史市值和点时行业，明确省略|
|结果可代表可交易收益|全部|否；未计成本、冲击、停牌、涨跌停和完整组合约束|
|为得到正 IC 持续搜索|新增探索|否；三选一、一次修改、随后停止|
"""


def main() -> None:
    global OUT
    argparse.ArgumentParser(description="Run the bounded post-hoc extension in a new isolated directory.").parse_args()
    validate_historical_interactions()
    OUT = allocate_run_directory()
    prepare()
    early_design()
    shutil.copy2(REVISION_SOURCE, OUT / "03_historical_revision_interaction_source.json")
    final_analysis()
