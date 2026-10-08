# RD-Agent-inspired Factor Research MVP

这是一个由 Codex 执行、人工监督的最小可复现因子研究原型。它完成：假设记录、候选因子实现、开发期滚动诊断、一次反馈修改、与预先指定的 20 日动量基准比较、版本冻结和最终留出期评估。

## 一键运行

```powershell
python run_experiment.py
```

## 事后有限扩展（2026-10-08）

以下入口只运行 2020-04-01 至 2023-12-29 原开发期，不读取或运行 2024—2025 留出期。它复现补充诊断、一次真实 Codex 假设/反馈记录、一次修改以及开发期内部的早期/后期切分：

```powershell
python run_posthoc_extension.py
```

每次执行都会在 `outputs/posthoc_runs/run_YYYYMMDD_HHMMSS_microseconds/` 下原子创建新的运行目录；同一时间戳发生碰撞时自动追加序号，已有审计目录永不覆盖，也不需要手工归档。

脚本自动完成数据加载、时间对齐、因子计算、IC/HAC、分组收益、换手、相关性诊断和报告落盘。三项假设及唯一一次修改建议不是复现脚本现场调用模型生成的：它们是 2026-10-08 Codex 交互形成的历史来源记录，固定保存在 `provenance/model_interactions/`，每次运行只复制快照、校验哈希并明确标记 `generated_in_this_run=false`。经济含义、来源可靠性和研究边界仍需人工监督。该分析属于事后探索，不是新的独立确认。

默认在没有真实数据时生成合成数据，仅验证程序，不作为实证结果。真实数据放到 `data/raw/market.parquet` 或 `data/raw/market.csv` 后重跑。最低字段：`date,symbol,close`；推荐另有 `open,volume,market_cap,industry`。若无 `open`，程序会把次日收盘作为执行价并在报告中披露。

输出位于 `outputs/latest/`，包括配置、输入输出审计记录、开发期与留出期指标、滚动结果、分组收益、版本冻结记录、中文报告和面试说明。

## 已完成的真实数据验证

真实实验使用 `D:/quant_project/multi_factor_stock_selection/data/raw` 的只读 AkShare 缓存，结果独立保存在 `outputs/real_data_20260930/`，没有覆盖合成验证结果。分阶段入口为：

```powershell
python run_real_experiment.py prepare
python run_real_experiment.py development_v1
# 由 Codex 查看 V1 后保存真实反馈记录，且只允许一次修改
python run_real_experiment.py revision_v2_and_freeze
python run_real_experiment.py holdout_once
```

阶段闸门防止留出期在版本冻结前运行，并拒绝重复运行留出期。真实实验的报告见 `outputs/real_data_20260930/REPORT_REAL_CN.md`。

## 口径

- 信号在 t 日收盘后计算，最早以 t+1 日开盘执行。
- 5 日标签为 t+1 开盘到 t+6 开盘收益。
- 每个评估区间的尾部 6 个交易日被隔离，避免标签跨边界。
- 横截面 1%/99% 缩尾、标准化、五分组等口径对候选和基准一致。
- 输出是因子诊断/简化回测，不是可交易策略收益；未计交易成本、冲击和完整组合约束。
