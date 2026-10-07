# OperatorForge · 踩坑库（症状 / 根因 / 修法）

> 三段式：症状 → 根因 → 修法。由 `random-ai-system-delivery` SOP 在交付过程中沉淀。

## P1 · Darcy 2D 五点差分右端项重复除 h²（指纹 288≈289）
- **症状**：`test_manufactured_solution_is_recovered_with_small_residual` 失败，实测 `relative_error ≈ 288.0`，与理论 ≤1e-6 相差 5 个数量级。
- **根因**：`data/pde.py` 的 `darcy_finite_difference` 中矩阵对角线/非对角项已含 `1/h²`，右端项又对 `forcing_field` 重复除 `h_squared`，整体被放大 (n-1)² 倍；本例 n=18→289，与 288 吻合。
- **修法**：右端项去掉 `/ h_squared`，仅保留 `forcing_field[sample_index, 1:-1, 1:-1]`。修复后双绿（≤1e-6 且残差≤1e-10）。

## P2 · OpFuse 全关三组件≡FNO 的硬契约不可达
- **症状**：`test_fully_disabled_opfuse_is_numerically_equivalent_to_fno` 报 `TypeError: no_gate`，无法表达 A/B/C 三组件全关。
- **根因**：`operators/opfuse.py` 原仅支持单选字符串 `ablation`（"no_gate"/"no_markov"/"no_highfreq"），不能组合，导致"全关≡FNO"契约无法构造。
- **修法**：新增 `no_gate / no_markov / no_highfreq` 三个布尔参数；`_resolve_components()` 改为 `use_x = (not no_x) and (ablation is None or ablation != "no_x")`；全关路径用严格恒等退化（gain=1、markov gain=0、highfreq 权重=0，注释标明硬契约），与 FNO 逐位一致。

## P3 · benchmark 缺失可选后端聚合出 NaN（伪造指标）
- **症状**：`test_missing_optional_backend_is_marked_skipped_without_fake_metrics` 失败，空数组 `float(values.mean())` 得 `NaN`，违反"缺失后端标 skipped 不伪造"。
- **根因**：`pipeline/benchmark.py` 对后端不可用分支仍走 `mean()` 聚合，产出 NaN。
- **修法**：后端不可用则写 `status="skipped"` + reason，聚合阶段显式跳过 NaN/非有限值。

## P4 · 谱尾误差归一化口径爆炸（1e12）
- **症状**：`spectral_tail_error` 在高频尾能量趋零时归一化到近零分母 → 值爆炸到 ~1e12，不可比。
- **根因**：`eval/metrics.py` 原归一化到"目标高频尾能量"（近零），数值不稳定。
- **修法**：改为归一化到目标总谱能量 `||target_full||`（无量纲有界），并加 `isfinite` 守卫；修复后谱尾误差在 1e-7 量级、经典方法 ≈1e-16。

## P5 · 超分门禁为结构性恒等式（G2 非判别）
- **症状**：OpFuse 与 FNO 在 train@64→test@256 零样本超分误差**逐位相同**（0.00566… / 0.44057…）。
- **根因**：OpFuse 的离网格转移路径是与之结构相同的 FNO 对角谱算子，超分误差必然逐位相等 → G2 门禁在此分辨率下无判别力。
- **修法**：诚实重定义 G2——标记 `structural_identity=True`，真实架构比较改用 in-grid 口径（OpFuse 0.1711 vs FNO 0.2220）；不伪造、不降格硬推。

## P6 · 消融三组件全为负贡献（增益来自容量而非命名组件）
- **症状**：burgers1d 消融 arms：full=0.3261 > no_gate=0.3207 > no_markov=0.3168 > no_highfreq=0.3037（去掉组件反而更好）。
- **根因**：在有限数据/预算下，A 自适应频谱门控 / B 混叠修正 / C 高频残差分支并非普适有益；旗舰相对 FNO 的增益实际来自 `LiftedSpectralMap` 的特征提升带来的**容量**增加，而非命名组件。
- **修法**：补做等参数量 / 等预算 / 同初始化三组对照证明；诚实改写旗舰声明（OpFuse=更强谱算子主干+容量提升，组件为可选增强而非必需），如实报告 darcy2d 上 OpFuse(0.1872) 负于 FNO(0.1788)。不掩盖、不重定义数字。

## P7 · ruff 收口防乒乓（check 与 format 必须同轮）
- **症状**：工程师自报"全绿"，独立复跑 `ruff check` 仍 1 error（Yoda 条件 SIM300）+ `ruff format` 6 文件待重排。
- **根因**：format 改动可能引入新 lint，单跑其一会互相触发；且人工漏报。
- **修法**：同一轮内顺序执行 `ruff check . --fix` → `ruff format .` → 复跑 `ruff check .` + `ruff format --check .` 收口，最终 All checks passed + 42 files formatted。结论：自测须独立复跑，不轻信单一报告。

## P8 · 平台 429 限流导致 agent 中断
- **症状**：software-engineer / software-qa-engineer / software-engineer-2 三次因 `429 使用量超出频率限制` 中断。
- **根因**：并发 agent 触发平台频率上限。
- **修法**：重置后重派 software-engineer-3 串行续作；交付脚本 `gh_push.py` 的 `Gh` 封装对 502/503/reset 做退避重试（2s×(attempt+1)，最多 5 次）。
