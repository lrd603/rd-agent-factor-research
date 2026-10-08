# 简历主张核对表

状态口径：**已完成**表示有真实数据产物和可核查实现；**部分完成**表示只覆盖该能力的一部分；**未完成**表示缺少必要流程或数据，不能写成已实现。

| 主张 | 状态 | 可安全使用的表述 | 限制与证据 |
|---|---|---|---|
| Hypothesis generation | 部分完成 | 记录了三个候选假设，并选择跳过近期的中期动量作为 V1 | 假设来自早期 Codex 监督原型；真实数据实验没有新的动态模型调用。不能声称构建了自主 LLM 假设生成 agent。证据：`outputs/real_data_20260930/02_provenance_before_results.json`、`src/pipeline.py` |
| Factor implementation | 已完成 | 实现了 V1 跳过近期动量和 V2 波动率缩放版本，并统一进行横截面缩尾和标准化 | V1：`close[t-5]/close[t-60]-1`；V2：V1 除以 20 日实现波动率。证据：`src/factors_v1.py`、`src/factors_v2.py`、`src/real_pipeline.py` |
| Diagnostics | 已完成（核心指标） | 实现 IC、Rank IC、ICIR、五分组收益差、最高组换手率、与基准秩相关和年度窗口稳定性诊断 | 属于因子诊断，不是含成本的可交易组合回测。证据：`04_development_v1.csv`、`06_development_comparison.csv`、`08_final_holdout_results.csv` |
| Industry/size exposure diagnostics | 未完成 | 不应写入已完成能力 | 真实缓存没有可靠的行业和市值字段，暴露输出为空对象。证据：`01_real_data_manifest.json`、`08_exposure_diagnostics.json` |
| Iterative refinement | 已完成（单次、受约束） | 根据 V1 开发期诊断进行一次真实 Codex 反馈修改，生成 V2，之后停止调参 | 公式族在 V1 前已经允许；这不是开放式多轮 agent 优化。证据：`05_actual_revision_feedback.json`、`06_development_v2.csv` |
| Human-designed baselines | 已完成 | 使用人工预先指定的 20 日动量作为统一基准 | 基准公式为 `close[t]/close[t-20]-1`，并非模型自动发现。证据：`00_config_frozen_before_results.json`、`04_development_v1.csv`、`08_final_holdout_results.csv` |
| Rolling-window evaluation | 已完成 | 在开发期的四个年度窗口中检查因子指标的跨期稳定性 | 窗口是固定年度诊断，没有逐窗训练、冻结、向前测试。证据：`04_development_v1_rolling.csv`、`06_development_v2_rolling.csv` |
| Walk-forward validation | 未完成 | 不应声称完成完整 walk-forward；最多写“年度滚动窗口稳定性诊断” | 缺少多轮“训练/选择 → 下一未见窗口测试 → 向前推进”的序列，也没有逐窗重新冻结模型 |
| Isolated holdout | 已完成 | 在版本冻结后，对 2024—2025 年留出期只评估一次，未依据留出期继续调参 | 时间戳和冻结记录支持阶段顺序。证据：`07_version_freeze_before_holdout.json`、`08_final_holdout_results.csv`、`09_completed_run_record.json` |
| Point-in-time execution alignment | 已完成（简化口径） | 信号在 `t` 收盘后计算，标签使用 `t+1` 开盘至 `t+6` 开盘，并隔离区间尾部跨界标签 | 使用前复权开盘价，未建模停牌、涨跌停和真实成交约束。证据：`03_pipeline_audit.json`、`src/pipeline.py` |
| Transaction-cost/portfolio backtest | 未完成 | 不应声称完成可交易策略回测或获得可实现收益 | 未建模费用、冲击、容量、组合权重和完整公司行动现金流 |
| Reproducibility/audit trail | 已完成 | 保存冻结配置、数据清单、单次反馈、源码哈希、版本冻结和最终完成记录 | 数据 CSV 本身不嵌入供应商和复权元数据；来源依赖旧项目代码及配置。证据：`00_config_frozen_before_results.json` 至 `09_completed_run_record.json` |

## 建议的简历表述

> Built a reproducible, Codex-supervised A-share factor research prototype with point-in-time signal alignment, IC/rank-IC and quantile diagnostics, one development-driven revision, annual rolling stability checks, a human-designed momentum baseline, and a version-frozen isolated holdout; documented negative empirical results and data limitations without holdout retuning.

不建议使用 “autonomous hypothesis-generation agent”“full walk-forward backtest”“industry/size neutralization” 或 “profitable alpha strategy”等表述，因为现有证据不支持这些主张。
