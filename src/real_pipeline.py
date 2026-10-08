from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import (ROOT, add_labels, cs_process, evaluate, exposure_diagnostics,
                       period_slice, safe_corr, utcnow, validate_data, write_json)
from .factors_v1 import candidate_v1
from .factors_v2 import candidate_v2

CFG = ROOT / "config" / "experiment_real.json"
OUT = ROOT / "outputs" / "real_data_20260930"


def cfg() -> dict:
    return json.loads(CFG.read_text(encoding="utf-8"))


def source_files(c: dict) -> list[Path]:
    files = []
    for p in sorted(Path(c["source_raw_dir"]).glob("*.csv")):
        columns = set(pd.read_csv(p, nrows=0).columns)
        if {"date", "symbol", "open", "close"} <= columns:
            files.append(p)
    if not files:
        raise FileNotFoundError("No source CSV files found")
    return files


def load_real(c: dict) -> pd.DataFrame:
    parts = []
    for p in source_files(c):
        d = pd.read_csv(p, dtype={"symbol": str})
        d["symbol"] = d["symbol"].str.zfill(6)
        parts.append(d)
    x = pd.concat(parts, ignore_index=True)
    x["date"] = pd.to_datetime(x["date"])
    for column in ["open", "high", "low", "close"]:
        if column in x:
            x.loc[x[column] <= 0, column] = np.nan
    x["data_source"] = "existing_AkShare_cache_read_only"
    return x.sort_values(["symbol", "date"]).reset_index(drop=True)


def panel(c: dict) -> pd.DataFrame:
    x = add_labels(load_real(c), c)
    g = x.groupby("symbol", sort=False)["close"]
    x["benchmark_raw"] = g.pct_change(20, fill_method=None)
    x["candidate_v1_raw"] = candidate_v1(x)
    x["candidate_v2_raw"] = candidate_v2(x)
    for n in ["benchmark", "candidate_v1", "candidate_v2"]:
        x[n] = cs_process(x[f"{n}_raw"], x, c)
    return x


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError(f"Real output already exists; refusing overwrite: {OUT}")
    OUT.mkdir(parents=True)
    shutil.copy2(CFG, OUT / "00_config_frozen_before_results.json")
    c = cfg(); files = source_files(c); x = load_real(c); check = validate_data(x)
    sample_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files[:5] + files[-5:]}
    raw_nonpositive = 0
    for p in files:
        z = pd.read_csv(p, usecols=["open", "close"])
        raw_nonpositive += int(((z["open"] <= 0) | (z["close"] <= 0)).sum())
    manifest = {
      "audited_at": utcnow(), "read_only_source": c["source_raw_dir"],
      "source_identity": "AkShare A-share daily history cache; Eastmoney primary/Tencent fallback according to source loader",
      "is_real_market_data": True, "adjustment": c["price_basis"],
      "fields": list(x.columns), "checks": check, "file_count": len(files),
      "raw_nonpositive_open_or_close_rows_converted_to_nan": raw_nonpositive,
      "sample_sha256_first_last_5": sample_hashes,
      "source_evidence_files": [c["source_project_config"], c["source_loader"]],
      "excluded_non_stock_csv": ["benchmark_csi500.csv (no symbol column)"],
      "limitations": [c["universe_limitation"], "CSV rows lack embedded vendor and adjustment metadata; provenance relies on source code/config", "qfq open is used for return alignment, not a literal executable historical price", "nonpositive OHLC placeholders are converted to missing without forward fill"]}
    write_json(OUT / "01_real_data_manifest.json", manifest)
    provenance = {
      "recorded_at": utcnow(),
      "hypothesis_generation": {"type":"pre_registered_existing_logic","dynamic_agent_generation":False,
        "truthful_description":"Three hypotheses and H1 selection originated in the earlier Codex-supervised prototype. This real run does not claim a new model call or regenerate them after viewing data.",
        "hypotheses":["H1 skip-recent 60-day momentum","H2 volume-confirmed momentum","H3 low volatility"],"implemented":"H1"},
      "benchmark": {"type":"human_requested_pre_specification","formula":c["benchmark"]["formula"]},
      "revision_policy": {"type":"pre_registered_formula_family","dynamic_decision_pending":True,"maximum":1,
        "note":"V1 development metrics will be inspected once by Codex; the actual feedback record must exist before V2 runs."}}
    write_json(OUT / "02_provenance_before_results.json", provenance)
    write_json(OUT / "03_pipeline_audit.json", {
      "label_alignment":"signal at t close; entry qfq open t+1; exit qfq open t+6",
      "rolling_grouping":"all shifts/pct_change/rolling operations grouped by symbol",
      "missing_values":"rolling warm-up and absent future labels remain NaN; evaluation drops only factor/label missing rows per date",
      "turnover":"1 - intersection(top-quintile symbols_t, symbols_t-1)/size(top_t)",
      "boundary_leakage":"period_slice requires label_exit_date <= period end; start uses only trailing raw history",
      "execution_limitation":"qfq open and no suspension/limit/cost model; factor diagnostic only"})
    print(f"prepared {OUT}; rows={len(x)}, symbols={x.symbol.nunique()}")


def development_v1() -> None:
    c=cfg(); required=OUT/"02_provenance_before_results.json"
    if not required.exists(): raise RuntimeError("Run prepare first")
    if (OUT/"04_development_v1.csv").exists(): raise RuntimeError("V1 already evaluated")
    p=panel(c); d=c["development"]; rows=[]
    for f in ["benchmark","candidate_v1"]:
        m,g=evaluate(p,f,d["start"],d["end"],c["quantiles"]); rows.append(m); g["factor"]=f; g.to_csv(OUT/f"04_groups_{f}.csv",index=False)
    rolling=[]
    for a,b in c["rolling_windows"]:
        for f in ["benchmark","candidate_v1"]:
            m,_=evaluate(p,f,a,b,c["quantiles"]); rolling.append(m)
    pd.DataFrame(rows).to_csv(OUT/"04_development_v1.csv",index=False)
    pd.DataFrame(rolling).to_csv(OUT/"04_development_v1_rolling.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def revision_v2_and_freeze() -> None:
    c=cfg(); feedback=OUT/"05_actual_revision_feedback.json"
    if not feedback.exists(): raise RuntimeError("Actual post-V1 feedback record is required before V2")
    if (OUT/"07_version_freeze_before_holdout.json").exists(): raise RuntimeError("Already frozen")
    p=panel(c); d=c["development"]; m,g=evaluate(p,"candidate_v2",d["start"],d["end"],c["quantiles"])
    pd.DataFrame([m]).to_csv(OUT/"06_development_v2.csv",index=False); g.to_csv(OUT/"06_groups_candidate_v2.csv",index=False)
    rolling=[]
    for a,b in c["rolling_windows"]:
        z,_=evaluate(p,"candidate_v2",a,b,c["quantiles"]); rolling.append(z)
    pd.DataFrame(rolling).to_csv(OUT/"06_development_v2_rolling.csv",index=False)
    v1=pd.read_csv(OUT/"04_development_v1.csv"); both=pd.concat([v1,pd.DataFrame([m])],ignore_index=True)
    candidates=both[both.factor.isin(["candidate_v1","candidate_v2"])].copy()
    selected=candidates.sort_values(["mean_rank_ic","factor"],ascending=[False,True]).iloc[0].factor
    both.to_csv(OUT/"06_development_comparison.csv",index=False)
    write_json(OUT/"07_version_freeze_before_holdout.json",{"frozen_at":utcnow(),"selected":selected,
      "rule":"higher development mean_rank_ic between V1 and the single V2 revision; tie by factor name",
      "holdout_evaluated":False,"v1_sha256":hashlib.sha256((ROOT/'src/factors_v1.py').read_bytes()).hexdigest(),
      "v2_sha256":hashlib.sha256((ROOT/'src/factors_v2.py').read_bytes()).hexdigest()})
    print(both.to_string(index=False)); print("FROZEN",selected)


def holdout_once() -> None:
    c=cfg(); freeze_path=OUT/"07_version_freeze_before_holdout.json"
    if not freeze_path.exists(): raise RuntimeError("Freeze required")
    if (OUT/"08_final_holdout_results.csv").exists(): raise RuntimeError("Holdout already run; no rerun allowed")
    freeze=json.loads(freeze_path.read_text(encoding="utf-8")); final=freeze["selected"]; p=panel(c); h=c["holdout"]; rows=[]
    hx=period_slice(p,h["start"],h["end"])
    for f in ["benchmark",final]:
        m,g=evaluate(p,f,h["start"],h["end"],c["quantiles"]); rows.append(m); g["factor"]=f; g.to_csv(OUT/f"08_holdout_groups_{f}.csv",index=False)
    tab=pd.DataFrame(rows); corr=safe_corr(hx[final],hx["benchmark"],True)
    tab["rank_correlation_with_benchmark"]=tab.factor.map({"benchmark":1.0,final:corr}); tab.to_csv(OUT/"08_final_holdout_results.csv",index=False)
    write_json(OUT/"08_exposure_diagnostics.json",exposure_diagnostics(hx,final))
    dev=pd.read_csv(OUT/"06_development_comparison.csv").set_index("factor"); ht=tab.set_index("factor")
    report=f"""# 真实数据验证报告\n\n找到 500 个字段合格的 AkShare 股票缓存文件，其中 601112.csv 为空，最终有效样本为 499 只股票、670,233 行，覆盖 2020-01-02 至 2025-12-31。字段包括前复权 OHLC、成交量、成交额和换手率。旧项目加载器以东方财富为主、腾讯为回退；CSV 未内嵌来源和复权元数据，证据来自旧项目代码及配置。样本主要源于当前存续/当前成分股票，历史退市覆盖不完整，存在幸存者偏差。\n\n配置在结果前冻结：开发期 {c['development']['start']}—{c['development']['end']}，留出期 {h['start']}—{h['end']}。基准为预先指定的20日动量；V1=`close[t-5]/close[t-60]-1`；唯一修改V2=V1/过去20日收益波动率。最终按开发期平均 Rank IC 冻结 `{final}`，之后留出期只运行一次。\n\n开发期 `{final}`：IC={dev.loc[final,'mean_ic']:.6f}，Rank IC={dev.loc[final,'mean_rank_ic']:.6f}，最高组换手率={dev.loc[final,'mean_top_group_turnover']:.6f}，Q5-Q1五日收益差={dev.loc[final,'top_minus_bottom_mean_forward_return']:.6f}。留出期：IC={ht.loc[final,'mean_ic']:.6f}，Rank IC={ht.loc[final,'mean_rank_ic']:.6f}，最高组换手率={ht.loc[final,'mean_top_group_turnover']:.6f}，Q5-Q1={ht.loc[final,'top_minus_bottom_mean_forward_return']:.6f}，与基准秩相关={corr:.6f}。\n\n这只是因子诊断：以 t 日收盘信息计算、t+1 前复权开盘到 t+6 前复权开盘作为标签，区间末尾按退出日隔离。未建模停牌、涨跌停、冲击、成本和公司行动现金流；前复权开盘不是可直接成交的历史报价，因此不能宣称可交易策略收益。无可靠行业和市值字段，规模/行业暴露诊断省略。假设并非本轮动态 agent 新生成；来源与真实的一次反馈修改记录见 JSON。\n"""
    (OUT/"REPORT_REAL_CN.md").write_text(report,encoding="utf-8")
    freeze["holdout_evaluated"]=True; freeze["holdout_run_at"]=utcnow(); write_json(OUT/"09_completed_run_record.json",freeze)
    print(tab.to_string(index=False))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["prepare","development_v1","revision_v2_and_freeze","holdout_once"]); a=ap.parse_args()
    globals()[a.phase]()
