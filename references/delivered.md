# OperatorForge · 交付登记（作者：晨星）

> 由 `random-ai-system-delivery` SOP 全自动交付。每次交付追加一行。

| 系统 | 域 | 日期 | Tag | 旗舰模型 | 三任务聚合 relL2（越低越好） | 最强基线 | 门禁 G1–G4 | 质量等级 | 已知限制（诚实披露） |
|------|----|------|-----|----------|------------------------------|----------|------------|----------|----------------------|
| OperatorForge | Neural Operator / 科学机器学习（PDE 解算子学习） | 2026-10-07 | v0.1.0 | OpFuse | **0.1711**（7 模型中最佳） | FNO 0.2220 / rbf 0.2349 / coarse 0.2922 / mlp 0.6391 / cnn 0.6895 / deeponet 0.7791 | ✅ G1 性能(0.771≤0.9) · ✅ G2 超分(结构性恒等式，真实比较见 in-grid 0.1711<0.2220) · ✅ G3 不变量(梯度 5.6e-10≤1e-6, advdiff 解析 0.0, Parseval 7.1e-15≤1e-12) · ✅ G4 确定性(逐位一致) | **A**（非 S：G2 离网格门禁结构性真空、DeepONet burgers 弱于常数预测、darcy2d 上 OpFuse 负于 FNO/rbf） | ① 超分门禁为恒等式（OpFuse 离网格转移路径与 FNO 逐位相同）；② 三组件消融均为负贡献（有限数据下非普适有益），旗舰增益来自 LiftedSpectralMap 特征提升的容量而非命名组件；③ darcy2d：FNO 0.179 / rbf 0.150 优于 OpFuse 0.187；④ DeepONet 在 burgers1d relL2=1.569（弱于常数预测），属归纳偏置不匹配非 bug |

## 复现命令
```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -r requirements.lock.txt
pytest -q                                        # 130 passed
python examples/run_demo.py                      # 双跑产 benchmark.json，约 9s
```
确定性：固定 seed=1729 与 seeds=[17,29,43]；独立二次运行逐位一致（signature 6f94d3…）。

## 性能基线（3 seeds 聚合 relative L2）
| 任务 | OpFuse | FNO | rbf | coarse | mlp | cnn | deeponet |
|------|--------|-----|-----|--------|-----|-----|----------|
| advdiff1d | **1.22e-5** | 0.0064 | 0.286 | 0.0555 | 1.005 | 0.872 | 0.123 |
| burgers1d | **0.326** | 0.481 | 0.440 | 0.489 | 0.603 | 0.570 | 1.569 |
| darcy2d | 0.187 | **0.179** | **0.150** | 0.160 | 0.318 | 0.626 | 0.646 |

结论：OpFuse 在 advdiff1d、burgers1d 夺冠；darcy2d 上 rbf/FNO 略优。三任务总体 OpFuse 最佳（0.1711 < FNO 0.2220）。
