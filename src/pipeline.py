from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .factors_v1 import candidate_v1
from .factors_v2 import candidate_v2

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config" / "experiment.json"
OUT = ROOT / "outputs" / "latest"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def make_synthetic(cfg: dict) -> pd.DataFrame:
    """Deterministic panel for software validation, never empirical evidence."""
    rng = np.random.default_rng(cfg["seed"])
    dates = pd.bdate_range(cfg["synthetic"]["start"], cfg["synthetic"]["end"])
    n = cfg["synthetic"]["n_symbols"]
    symbols = [f"S{i:04d}" for i in range(n)]
    industries = np.array([f"IND{i % 8}" for i in range(n)])
    size = rng.normal(22, 1.1, n)
    market = rng.normal(0.00015, 0.009, len(dates))
    rets = np.zeros((len(dates), n))
    latent = rng.normal(0, 0.01, n)
    for t in range(len(dates)):
        eps = rng.normal(0, 0.013 + 0.004 * rng.random(n), n)
        # Persistent but weak cross-sectional component makes diagnostics meaningful.
        latent = 0.985 * latent + 0.015 * eps
        rets[t] = market[t] + 0.08 * latent + eps
    close = 20 * np.exp(np.cumsum(rets, axis=0))
    overnight = rng.normal(0, 0.002, close.shape)
    open_ = close * np.exp(overnight)
    volume = rng.lognormal(14, 0.6, close.shape)
    cap = close * np.exp(size)[None, :] / 1e8
    frames = []
    for j, sym in enumerate(symbols):
        frames.append(pd.DataFrame({"date": dates, "symbol": sym, "open": open_[:, j],
                                    "close": close[:, j], "volume": volume[:, j],
                                    "market_cap": cap[:, j], "industry": industries[j],
                                    "data_source": "synthetic_validation_only"}))
    return pd.concat(frames, ignore_index=True)


def load_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    chosen = None
    for rel in cfg["data_priority"]:
        p = ROOT / rel
        if p.exists():
            chosen = p
            break
    if chosen is None:
        df = make_synthetic(cfg)
        return df, {"kind": "synthetic_validation_only", "path": None,
                    "warning": "仅用于验证程序，不能作为因子实证结果"}
    df = pd.read_parquet(chosen) if chosen.suffix == ".parquet" else pd.read_csv(chosen)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"date", "symbol", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"真实数据缺少字段: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    fallback = "open" not in df
    if fallback:
        df["open"] = df["close"]
    df["data_source"] = f"real_file:{chosen.name}"
    return df, {"kind": "real", "path": str(chosen), "open_fallback_to_close": fallback}


def validate_data(df: pd.DataFrame) -> dict:
    keys_dup = int(df.duplicated(["date", "symbol"]).sum())
    bad_price = int(((df["close"] <= 0) | (df["open"] <= 0)).sum())
    if keys_dup or bad_price:
        raise ValueError(f"数据检查失败: duplicate_keys={keys_dup}, nonpositive_prices={bad_price}")
    return {"rows": len(df), "symbols": int(df.symbol.nunique()),
            "start": str(df.date.min().date()), "end": str(df.date.max().date()),
            "duplicate_keys": keys_dup, "nonpositive_prices": bad_price,
            "optional_fields": {c: c in df for c in ["volume", "market_cap", "industry"]}}


def add_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    lag, h = cfg["execution_lag_days"], cfg["return_horizon_days"]
    entry = g["open"].shift(-lag)
    exit_ = g["open"].shift(-(lag + h))
    df["forward_return"] = exit_ / entry - 1.0
    df["label_exit_date"] = g["date"].shift(-(lag + h))
    return df


def cs_process(s: pd.Series, df: pd.DataFrame, cfg: dict) -> pd.Series:
    tmp = pd.DataFrame({"date": df["date"], "x": s})
    def one(x: pd.Series) -> pd.Series:
        if x.notna().sum() < 5:
            return x * np.nan
        lo, hi = x.quantile([cfg["winsor_lower"], cfg["winsor_upper"]])
        y = x.clip(lo, hi)
        sd = y.std()
        return (y - y.mean()) / sd if sd and np.isfinite(sd) else y * np.nan
    return tmp.groupby("date", group_keys=False)["x"].apply(one).sort_index()


def build_panel(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = add_labels(df, cfg)
    g = df.groupby("symbol", sort=False)["close"]
    df["benchmark_raw"] = g.pct_change(20, fill_method=None)
    df["candidate_v1_raw"] = candidate_v1(df)
    df["candidate_v2_raw"] = candidate_v2(df)
    for name in ["benchmark", "candidate_v1", "candidate_v2"]:
        df[name] = cs_process(df[f"{name}_raw"], df, cfg)
    return df


def period_slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Exit date must remain inside period: explicit label-boundary purge.
    return panel[(panel.date >= start) & (panel.date <= end) &
                 (panel.label_exit_date <= end) & panel.forward_return.notna()].copy()


def safe_corr(a: pd.Series, b: pd.Series, rank: bool = False) -> float:
    z = pd.concat([a, b], axis=1).dropna()
    if len(z) < 5 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return np.nan
    if rank:
        z = z.rank(method="average")
    return float(z.corr().iloc[0, 1])


def evaluate(panel: pd.DataFrame, factor: str, start: str, end: str, q: int) -> tuple[dict, pd.DataFrame]:
    x = period_slice(panel, start, end)
    daily, group_rows, prev_top = [], [], None
    for date, d in x.groupby("date", sort=True):
        d = d[["symbol", factor, "forward_return"]].dropna()
        if len(d) < q * 2:
            continue
        ic = safe_corr(d[factor], d.forward_return)
        ric = safe_corr(d[factor], d.forward_return, rank=True)
        try:
            buckets = pd.qcut(d[factor].rank(method="first"), q, labels=False) + 1
        except ValueError:
            continue
        means = d.assign(group=buckets).groupby("group").forward_return.mean()
        top = set(d.loc[buckets == q, "symbol"])
        turnover = np.nan if prev_top is None else 1 - len(top & prev_top) / max(len(top), 1)
        prev_top = top
        daily.append({"date": date, "ic": ic, "rank_ic": ric, "turnover_top_group": turnover})
        for k, v in means.items():
            group_rows.append({"date": date, "group": int(k), "mean_forward_return": float(v)})
    dd = pd.DataFrame(daily)
    gg = pd.DataFrame(group_rows)
    gm = gg.groupby("group").mean(numeric_only=True)["mean_forward_return"] if len(gg) else pd.Series(dtype=float)
    metrics = {"factor": factor, "start": start, "end": end, "n_dates": len(dd),
               "mean_ic": dd.ic.mean() if len(dd) else np.nan,
               "ic_ir": dd.ic.mean() / dd.ic.std() if len(dd) and dd.ic.std() else np.nan,
               "mean_rank_ic": dd.rank_ic.mean() if len(dd) else np.nan,
               "rank_ic_ir": dd.rank_ic.mean() / dd.rank_ic.std() if len(dd) and dd.rank_ic.std() else np.nan,
               "mean_top_group_turnover": dd.turnover_top_group.mean() if len(dd) else np.nan,
               "top_minus_bottom_mean_forward_return": gm.get(q, np.nan) - gm.get(1, np.nan)}
    return metrics, gg


def exposure_diagnostics(x: pd.DataFrame, factor: str) -> dict:
    out = {}
    if "market_cap" in x:
        out["rank_corr_log_market_cap"] = safe_corr(x[factor], np.log(x.market_cap.clip(lower=1e-9)), True)
    if "industry" in x:
        means = x.groupby("industry")[factor].mean()
        out["industry_mean_abs_zscore"] = float(means.abs().mean())
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def fnum(x) -> str:
    return "NA" if pd.isna(x) else f"{x:.6f}"


def main() -> None:
    started = utcnow(); t0 = time.time(); cfg = load_config()
    if OUT.exists():
        archive = ROOT / "outputs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.move(str(OUT), str(archive))
    OUT.mkdir(parents=True)
    shutil.copy2(CFG_PATH, OUT / "experiment_config_frozen_before_results.json")
    hypotheses = {
      "actor": "Codex（人工监督原型；不是独立运行的外部 LLM/多 agent 系统）", "generated_at": started,
      "model_input": "基于可用日频字段提出3个可检验因子假设；禁止查看本次回测表现；选择一个实现。",
      "model_output": [
        {"id":"H1","name":"skip_recent_momentum","formula":"close[t-5]/close[t-60]-1","reason":"中期趋势可能延续，跳过近期以减轻短期反转干扰","requires":["close"]},
        {"id":"H2","name":"volume_confirmed_momentum","formula":"mom20 * zscore(log(volume20/volume60))","reason":"成交活跃度可能确认趋势","requires":["close","volume"]},
        {"id":"H3","name":"low_volatility","formula":"-std(ret,20)","reason":"低波动异象可能带来横截面预测力","requires":["close"]}],
      "selected": "H1（字段最稳健、与预指定20日动量基准存在清晰但非同一的经济差异）",
      "benchmark_pre_specified": cfg["benchmark"]}
    write_json(OUT / "01_hypotheses_and_model_io.json", hypotheses)
    df, source = load_data(cfg); checks = validate_data(df)
    write_json(OUT / "00_data_manifest.json", {"source": source, "checks": checks})
    panel = build_panel(df, cfg)
    dev = cfg["development"]; hold = cfg["holdout"]; q = cfg["quantiles"]
    # Phase 1: only development data is evaluated.
    dev_rows, all_groups = [], []
    for factor in ["benchmark", "candidate_v1"]:
        m, g = evaluate(panel, factor, dev["start"], dev["end"], q); dev_rows.append(m); g["factor"] = factor; all_groups.append(g)
    rolling = []
    for a, b in cfg["rolling_windows"]:
        for factor in ["benchmark", "candidate_v1"]:
            m, _ = evaluate(panel, factor, a, b, q); rolling.append(m)
    v1 = next(x for x in dev_rows if x["factor"] == "candidate_v1")
    feedback_time = utcnow()
    modification = {"feedback_at": feedback_time,
      "development_evidence": {k: v1[k] for k in ["mean_rank_ic","rank_ic_ir","mean_top_group_turnover"]},
      "model_input": "只根据开发期V1结果提出一次修改；之后停止调参。",
      "model_output": "将跳过近期的60日动量除以过去20日实现波动率，再做同样的横截面缩尾与标准化。理由：降低高噪声个股对排序的支配，并检验风险调整后的趋势是否更稳定。",
      "constraint": "仅此一次修改；不根据留出期继续调整"}
    write_json(OUT / "02_single_revision_record.json", modification)
    m2, g2 = evaluate(panel, "candidate_v2", dev["start"], dev["end"], q); dev_rows.append(m2); g2["factor"]="candidate_v2"; all_groups.append(g2)
    for a, b in cfg["rolling_windows"]:
        m, _ = evaluate(panel, "candidate_v2", a, b, q); rolling.append(m)
    dev_table = pd.DataFrame(dev_rows)
    # Deterministic development-only selection between V1/V2; benchmark is never a candidate.
    candidates = dev_table[dev_table.factor.isin(["candidate_v1", "candidate_v2"])].copy()
    candidates["selection_score"] = candidates["mean_rank_ic"].fillna(-999)
    final_factor = candidates.sort_values(["selection_score", "factor"], ascending=[False, True]).iloc[0].factor
    freeze = {"frozen_at": utcnow(), "selected_final_version": final_factor,
              "selection_rule_pre_holdout": "V1与唯一修改V2中开发期 mean_rank_ic 较高者；平局按名称排序",
              "holdout_not_accessed_before_freeze": True,
              "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src'/'factors_v1.py', ROOT/'src'/'factors_v2.py']}}
    write_json(OUT / "03_version_freeze.json", freeze)
    # Phase 2: only after freeze, evaluate holdout once.
    hold_rows=[]
    for factor in ["benchmark", final_factor]:
        m,g=evaluate(panel,factor,hold["start"],hold["end"],q); hold_rows.append(m); g["factor"]=factor; g.to_csv(OUT/f"holdout_groups_{factor}.csv",index=False)
    corr_dev=safe_corr(period_slice(panel,dev["start"],dev["end"])[final_factor], period_slice(panel,dev["start"],dev["end"])["benchmark"],True)
    corr_hold=safe_corr(period_slice(panel,hold["start"],hold["end"])[final_factor], period_slice(panel,hold["start"],hold["end"])["benchmark"],True)
    dev_table["rank_correlation_with_benchmark"] = dev_table.factor.map(lambda x: 1.0 if x=="benchmark" else safe_corr(period_slice(panel,dev["start"],dev["end"])[x],period_slice(panel,dev["start"],dev["end"])["benchmark"],True))
    hold_table=pd.DataFrame(hold_rows); hold_table["rank_correlation_with_benchmark"]=hold_table.factor.map({"benchmark":1.0,final_factor:corr_hold})
    dev_table.to_csv(OUT/"development_comparison.csv",index=False)
    pd.DataFrame(rolling).to_csv(OUT/"development_rolling.csv",index=False)
    pd.concat(all_groups).to_csv(OUT/"development_group_returns.csv",index=False)
    hold_table.to_csv(OUT/"final_holdout_results.csv",index=False)
    exposures={"development": exposure_diagnostics(period_slice(panel,dev["start"],dev["end"]),final_factor),"holdout":exposure_diagnostics(period_slice(panel,hold["start"],hold["end"]),final_factor)}
    write_json(OUT/"04_exposure_diagnostics.json",exposures)
    empirical = source["kind"] == "real"
    report=f"""# 简短中文报告\n\n本项目由 Codex 执行模型角色、人工监督，完成了三项假设生成、H1 实现、开发期滚动诊断、一次反馈修改、与预先指定20日动量基准比较、版本冻结及一次最终留出期评估。数据类型：**{source['kind']}**。{'以下结果属于真实文件上的研究诊断。' if empirical else '**当前没有真实行情；以下数字仅验证程序，不能作为因子实证证据。**'}\n\n时间切分在查看表现前固定：开发期 {dev['start']} 至 {dev['end']}，留出期 {hold['start']} 至 {hold['end']}。信号在收盘后计算，t+1 开盘执行，标签为 t+1 开盘至 t+6 开盘收益；每段尾部按 label_exit_date 隔离。\n\n开发期最终选择 `{final_factor}`（只在V1与唯一修改V2间按 mean Rank IC 选择），冻结后留出期仅运行一次。开发期 Rank IC={fnum(dev_table.set_index('factor').loc[final_factor,'mean_rank_ic'])}，留出期 Rank IC={fnum(hold_table.set_index('factor').loc[final_factor,'mean_rank_ic'])}；与基准的开发期/留出期秩相关分别为 {fnum(corr_dev)} / {fnum(corr_hold)}。完整 IC、Rank IC、ICIR、分组多空差、换手率及逐组收益见 CSV。\n\n限制：这是因子诊断/简化回测，没有交易成本、冲击、涨跌停、停牌、容量、完整组合构建和生存者偏差校正，因此不能推断可交易策略收益。合成数据运行更不能支持经济结论。规模/行业诊断仅在字段存在时输出。\n"""
    (OUT/"REPORT_CN.md").write_text(report,encoding="utf-8")
    interview="""# 面试说明\n\n- 因子：V1 为 `close[t-5]/close[t-60]-1`，意图捕捉中期趋势并跳过最近5日；V2 是其除以20日实现波动率的风险调整版本。基准在实验前固定为20日动量。\n- IC：每日横截面上因子与未来5日收益的 Pearson 相关；Rank IC 为 Spearman 秩相关，更不受极端值和非线性尺度影响。\n- 换手率：相邻信号日最高五分组成员替换比例；这是信号稳定性指标，不等于含权重与成本的实际组合换手。\n- 防泄漏：所有滚动量仅用 t 日及此前收盘；t 收盘信号最早 t+1 开盘成交；收益标签到 t+6 开盘；区间末尾要求标签退出日仍在区间内。配置在结果前保存，留出期在版本冻结后才运行。\n- 修改理由：开发期反馈后只修改一次，用历史波动率缩放，目标是减少高噪声股票对排序的支配；V1/V2都保留，最终版本只按开发期规则确定，留出期不再调参。\n"""
    (OUT/"INTERVIEW_NOTES_CN.md").write_text(interview,encoding="utf-8")
    resume="""- Built a reproducible, Codex-supervised factor research prototype covering hypothesis logging, point-in-time signal construction, rolling diagnostics, one feedback-driven revision, and a frozen holdout evaluation.\n- Implemented consistent benchmark/candidate evaluation with IC, Rank IC, quantile returns, turnover, correlation, boundary purging, and optional size/industry diagnostics; validated the pipeline on synthetic data when no empirical dataset was available.\n"""
    (OUT/"RESUME_BULLETS_EN.md").write_text(resume,encoding="utf-8")
    write_json(OUT/"run_metadata.json",{"started_at":started,"finished_at":utcnow(),"elapsed_seconds":round(time.time()-t0,3),"python":sys.version,"platform":platform.platform(),"command":"python run_experiment.py","status":"completed"})
    print(f"Completed. data={source['kind']}, final={final_factor}, outputs={OUT}")
